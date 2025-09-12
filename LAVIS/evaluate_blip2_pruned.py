"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

 Modified by Zimeng Wu in 2025
 - Added support for evaluating structured pruned models.
"""

import argparse
import json
import logging
import random
from functools import partial
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

from lavis.models.eva_vit import Attention as EvaVitAttention
from lavis.peft import PeftModel
from lavis.peft.tuners.prunelora.layer import Linear as PruneLoraLinear

from lavis.runners.runner_base import RunnerBase
from lavis.common.config import Config
from lavis.common.dist_utils import get_rank
from lavis.common.logger import setup_logger
import lavis.tasks as tasks
# imports modules for registration
from lavis.datasets.builders import *
from lavis.models import *
from lavis.processors import *
from lavis.tasks import *

from utils import *


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def main(args):

    cfg = Config(args)
    cfg.run_cfg.distributed = False

    setup_seeds(cfg)

    # set after init_distributed_mode() to only log on master.
    setup_logger()
    logger = logging.getLogger("my_logger")
    stream_handler = logging.StreamHandler()
    logger.addHandler(stream_handler)

    # cfg.pretty_print()

    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for name, module in model.named_modules():
        if isinstance(module, EvaVitAttention):
            qkv = module.qkv
            in_feat, out_feat = qkv.in_features, qkv.out_features//3
            q, k, v = nn.Linear(in_feat, out_feat, bias=True).to(device), nn.Linear(in_feat, out_feat, bias=False).to(device), nn.Linear(in_feat, out_feat, bias=True).to(device)
            q.weight.data = qkv.weight.data[:out_feat, :]
            k.weight.data = qkv.weight.data[out_feat:out_feat*2, :]
            v.weight.data = qkv.weight.data[out_feat*2:, :]
            q.bias.data = module.q_bias.data
            v.bias.data = module.v_bias.data
            module.q = q
            module.k = k
            module.v = v
            del module.qkv
            del module.q_bias
            del module.v_bias
            module.forward = partial(decoupled_visual_SA, module)
    
    if args.peft_ckpt is None:
        if args.pruned_ckpt is not None:
            logger.info("Load from Pruned Model: {}".format(args.pruned_ckpt))
            pruned_dict = torch.load(args.pruned_ckpt, map_location='cpu')
            pruned_model = pruned_dict['model']
        elif args.full_ckpt is not None:
            logger.info("Load from Full Param Finetuned Model: {}".format(args.full_ckpt))
            full_dict = torch.load(args.full_ckpt, map_location='cpu')
            pruned_model = full_dict['model']
    else:
        # for evaluating finetuned model
        pruned_dict = torch.load(args.pruned_ckpt, map_location='cpu')
        pruned_model = pruned_dict['model']
        pruned_model.config = None
        pruned_model = PeftModel.from_pretrained(pruned_model, args.peft_ckpt)
        if args.pruned_mask is not None:
            logger.info("Load from Pruned Mask: {}".format(args.pruned_mask))
            data_ = json.load(open(args.pruned_mask, 'r'))
            pruned_mask = {}
            for k in data_.keys(): # transfer to tensor
                pruned_mask[k] = [torch.tensor(mask, dtype=torch.bool) for mask in data_[k]]
            for _name, module in pruned_model.named_modules():
                if isinstance(module, PruneLoraLinear):
                    name = _name.replace('base_model.model.', '')
                    full_module = get_module_by_name(model, name)
                    if full_module is None:
                        continue
                    masks = pruned_mask[name+'.weight']
                    if module.input_base_layer is not None:
                        module.input_base_layer.weight.data = full_module.weight.data[masks[0]][:, ~masks[1]].clone()
                    if module.output_base_layer is not None:
                        module.output_base_layer.weight.data = full_module.weight.data[~masks[0]][:, masks[1]].clone()   
    
    if hasattr(model, 'visual_encoder'):
        logger.info("setting pruned visual encoder...")
        model.visual_encoder.blocks = pruned_model.visual_encoder.blocks
    if hasattr(model, 't5_model'):
        logger.info("setting pruned t5 model...")
        model.t5_model = pruned_model.t5_model
    if hasattr(model, 'opt_model'):
        logger.info("setting pruned opt model...")
        model.opt_model = pruned_model.opt_model
    if args.use_totally_loaded:
        model.ln_vision = pruned_model.ln_vision
        model.Qformer = pruned_model.Qformer
        if hasattr(model, 't5_proj'):
            model.t5_proj = pruned_model.t5_proj
        if hasattr(model, 'opt_proj'):
            model.opt_proj = pruned_model.opt_proj
    
    model.to(device)
    runner = RunnerBase(
        cfg=cfg, job_id=args.job_id, task=task, model=model, datasets=datasets
    )
    runner.evaluate(skip_reload=True)
    logger.info("[FINISH] - Finish Evaluating Model")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Evaluating")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--options", nargs="+", help="override some settings in the used config, the key-value pair in xxx=yyy format will be merged into config file (deprecate), change to --cfg-options instead.")

    parser.add_argument("--job_id", type=str, default="finetune", help="The id of the Job")
    
    parser.add_argument("--pruned_ckpt", type=str, default=None, help="The checkpoint path of pruned model")
    
    parser.add_argument("--peft_ckpt", type=str, default=None, help="The checkpoint path of pruned model")
    
    parser.add_argument("--full_ckpt", type=str, default=None, help="The checkpoint path of full param finetuned model")
    
    parser.add_argument("--pruned_mask", type=str, default=None, help="The json path of pruned mask")
    
    parser.add_argument('--use_totally_loaded', action='store_true', help='For example preserve the loaded Qformer')
    
    args = parser.parse_args()

    main(args)
