'''
Refer to
https://github.com/tloen/alpaca-lora/blob/main/finetune.py

 Inherited from finetune.py
 Modified by Zimeng Wu in 2025
 - Added support for finetuning from pruned models.
'''

import re
import json
import argparse
from pathlib import Path
import random
import numpy as np

from copy import deepcopy

import torch
import torch.nn as nn
import transformers

import torch.backends.cudnn as cudnn

import lavis.tasks as tasks
from lavis.common.config import Config
from lavis.common.dist_utils import get_rank, init_distributed_mode
from lavis.common.logger import setup_logger

# imports modules for registration
from lavis.datasets.builders import *
from lavis.models import *
from lavis.processors import *
from lavis.runners.runner_base import RunnerBase
from lavis.tasks import *

from lavis.common.logger import LoggerWithDepth

from lavis.peft import (
    PruneLoraConfig,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    # prepare_model_for_int8_training,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)

from functools import partial

from lavis.models.eva_vit import Attention as EvaVitAttention
from lavis.models.eva_vit import Block as EvaVitBlock
from lavis.models.blip2_models.modeling_t5 import T5Attention, T5LayerFF, T5Block

from utils import *
from lavis.peft.tuners.prunelora.layer import Linear as PruneLoraLinear

from lavis.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)

def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    
def forward_save_feature_hook(module, input, output):
    if isinstance(output, tuple):
        module.feature = output[0]
    else:
        module.feature = output
        
def register_hook(model):
    hook_list = []
    for name, module in model.named_modules():
        if isinstance(module, EvaVitBlock):
            match = re.search(r'blocks.(\d+)', name)
            num = int(match.group(1))
            if num % 2 == 0: 
                hook_list.append(module.register_forward_hook(forward_save_feature_hook))
        elif isinstance(module, T5Block):
            match = re.search(r'block.(\d+)', name)
            num = int(match.group(1))
            if num % 2 == 1: 
                hook_list.append(module.register_forward_hook(forward_save_feature_hook))
    return hook_list

def main(args):

    cfg = Config(args)
    init_distributed_mode(cfg.run_cfg)
    setup_seeds(cfg)
    
    logger = LoggerWithDepth(
        env_name="{}".format(args.save_ckpt_log_name), 
        config=args.__dict__,
        root_dir='tuned_checkpoint',
        setup_sublogger=True,
        sublogger_name=args.job_id
    )
    setup_logger()
    
    task = tasks.setup_task(cfg)
    
    hook_list = []
    if args.distill_mode:
        logger.log("Load full model as teacher...")
        full_model = task.build_teacher_model(cfg)
        full_model.to(args.device)
        for param in full_model.parameters():
            param.requires_grad = False
        
        hook_list += register_hook(full_model)
    else:
        full_model = task.build_model(cfg)
        
    for name, module in full_model.named_modules():
        if isinstance(module, EvaVitAttention):
            qkv = module.qkv
            in_feat, out_feat = qkv.in_features, qkv.out_features//3
            q, k, v = nn.Linear(in_feat, out_feat, bias=True).to(args.device), nn.Linear(in_feat, out_feat, bias=False).to(args.device), nn.Linear(in_feat, out_feat, bias=True).to(args.device)
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
        
    # Load Pruned Model
    logger.log("Load from Pruned Model: {}".format(args.pruned_ckpt))
    pruned_dict = torch.load(args.pruned_ckpt, map_location='cpu')
    model = pruned_dict['model']
    
    if args.pruned_mask is not None:
        logger.log("Load from Pruned Mask: {}".format(args.pruned_mask))
        data_ = json.load(open(args.pruned_mask, 'r'))
        pruned_mask = {}
        for k in data_.keys(): # transfer to tensor
            pruned_mask[k] = [torch.tensor(mask, dtype=torch.bool) for mask in data_[k]]
            
        pruned_linear_features = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                pruned_linear_features[name] = [(len(l)-int(sum(l))) for l in data_[name+'.weight']]
    
    # Prepare For LoRA
    model = prepare_model_for_kbit_training(model)
    
    logger.log("Freeze Backbone Dropouts...")
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            module.p = 0
        elif isinstance(module, T5Attention):
            module.dropout = 0
    for name, module in full_model.named_modules():
        if isinstance(module, nn.Dropout):
            module.p = 0
        elif isinstance(module, T5Attention):
            module.dropout = 0

    if args.distill_mode:
        logger.log("Adding Hook for Distillation...")
        hook_list += register_hook(model)

    if args.wr_lora:
        config = PruneLoraConfig(
            r=args.lora_r,
            pruned_r=args.lora_pruned_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules.split(","),
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="PREFIX_DERIVATIVE",
            init_lora_weights=True,
            pruned_features=pruned_linear_features,
        )
        model.config = {}
        model = get_peft_model(model, config).to(args.device)

        for _name, module in model.named_modules():
            if isinstance(module, PruneLoraLinear):
                name = _name.replace('base_model.model.', '').replace('.base_layer','')
                full_module = get_module_by_name(full_model, name)
                masks = pruned_mask[name+'.weight']
                if module.input_base_layer is not None:
                    module.input_base_layer.weight.data = full_module.weight.data[masks[0]][:, ~masks[1]].clone()
                if module.output_base_layer is not None:
                    module.output_base_layer.weight.data = full_module.weight.data[~masks[0]][:, masks[1]].clone()
    else:
        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules.split(","),
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="PREFIX_DERIVATIVE",
            init_lora_weights=True,
        )
        model.config = {}
        model = get_peft_model(model, config).to(args.device)
    logger.log(model)
    
    model.print_trainable_parameters()  
   
    old_state_dict = model.state_dict
    task.feature_norm = True
    
    if args.distill_mode:
        train_step_list = [task.train_step_low_visual, task.train_step_high_visual, task.train_step_low_encoder, task.train_step_high_encoder, task.train_step_low_decoder, task.train_step_high_decoder, task.train_step_top]
    else:
        train_step_list = [task.loss_fn]
    
    full_datasets = deepcopy(task.build_datasets(cfg))
    total_num = 0
    import random
    for key in full_datasets.keys():
        random.shuffle(full_datasets[key]['train'].annotation)
        total_num = len(full_datasets[key]['train'].annotation)

    if args.distill_mode:
        data_num = [[0, 20000], [20000, 40000], [40000, 120000], [120000, 160000], [160000, 240000], [240000, 320000], [320000, total_num]]
    else:
        data_num = [[0, total_num]]
        
    for i, loss_fn in enumerate(train_step_list):
        datasets = deepcopy(full_datasets)
        for key in datasets.keys():
            datasets[key]['train'].annotation = datasets[key]['train'].annotation[data_num[i][0]:data_num[i][1]]
        model.state_dict = (
            lambda self, *_, **__: get_peft_model_state_dict(
                self, old_state_dict()
            )
        ).__get__(model, type(model))
        task.train_step = loss_fn
        runner = RunnerBase(cfg=cfg, job_id=args.job_id, task=task, model=model, datasets=datasets)
        runner.train()
        del runner
    
    for hook_handle in hook_list:
        hook_handle.remove()
    
    logger.log("Saving to path: {}".format(logger.checkpoint_path))
    model.state_dict = old_state_dict
    model.save_pretrained(logger.sub_dir)
    model.save_pretrained(logger.log_dir)
    
    logger.log("[FINISH] - Finish Finetuning Model")
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Tuning Pruned Model')
    
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    # Model Type&Path
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--pruned_ckpt", type=str, default=None, help="The checkpoint path of pruned model")
    parser.add_argument("--job_id", type=str, default=None, help="The id of the Job")
    parser.add_argument('--save_ckpt_log_name', type=str, default="llama_finetune", help='the path for save the checkpoint and the log.')

    # Lora Configuration
    parser.add_argument('--lora_r', type=int, default=8, help='lora r')
    parser.add_argument('--lora_alpha', type=int, default=16, help='lora alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='lora dropout')
    parser.add_argument('--lora_target_modules', type=str, default="qkv,attn.proj,fc1,fc2,q,k,v,o,wi_0,wi_1,wo", help='lora target modules')

    #ddp
    parser.add_argument('--local_rank', type=int, default=-1) # Not Implemented
    parser.add_argument('--device', type=str, default="cuda", help='device')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    
    # modified
    parser.add_argument('--distill_mode', action='store_true', help='if use full model distillation')
    parser.add_argument('--wr_lora', action='store_true', help='if use new lora structure')
    parser.add_argument("--pruned_mask", type=str, default=None, help="The json path of pruned mask")
    parser.add_argument('--lora_pruned_r', type=int, default=8, help='r for the weight recalling branch')
    
    args = parser.parse_args()
    args.torch_version = int(torch.__version__.split('.')[1])

    main(args)
