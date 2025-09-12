import argparse
import json
import os
import random
from functools import partial

import lavis.compression.torch_pruning as tp
import lavis.tasks as tasks
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fvcore.nn import FlopCountAnalysis, flop_count_table
from lavis.common.config import Config
from lavis.common.dist_utils import init_distributed_mode
from lavis.common.logger import LoggerWithDepth
from lavis.compression.pruners import mask_pruner as mask_pruner
from lavis.compression.pruners.mask_pruner import (
    layer_norm_mask_pruner, linear_mask_pruner, param_mask_pruner,
    t5_attention_head_mask_pruner, t5_layer_norm_mask_pruner)
from lavis.datasets.builders import *
from lavis.datasets.data_utils import prepare_sample
from lavis.models import *
from lavis.models.blip2_models.modeling_t5 import (T5Attention,
                                                   T5ForConditionalGeneration,
                                                   T5LayerNorm)
from lavis.models.eva_vit import Attention as EvaVitAttention
from lavis.processors import *
from lavis.runners.runner_base import RunnerBase
from lavis.tasks import *
from utils import *


def forward_fn(model, example_input):
    return model(example_input)['loss']

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
def main(args):
    set_random_seed(args.seed)
    cfg = Config(args)
    init_distributed_mode(cfg.run_cfg)
    logger = LoggerWithDepth(
        env_name="{}".format(args.save_ckpt_log_name), 
        config=args.__dict__,
        root_dir='pruned_checkpoint',
        setup_sublogger=True,
        sublogger_name=args.job_id
    )
    train_task = tasks.setup_task(cfg)
    model = train_task.build_model(cfg)
    
    calibration_bs = args.calibration_bs
    cfg.run_cfg.batch_size_train = calibration_bs
    cfg.run_cfg.batch_size_eval = calibration_bs
    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    model.to(args.device)
    
    runner = RunnerBase(
        cfg=cfg, job_id=None, task=task, model=model, datasets=datasets
    )
    data_loader = runner.train_loader

    batch_data = next(iter(data_loader))
    example_input = get_one_from_batch(batch_data, 0)
    raw_flop = FlopCountAnalysis(model, example_input)
    logger.log(flop_count_table(raw_flop, max_depth=2, show_param_shapes=True))
    logger.log("Total Flops: "+str(raw_flop.total() / 1e9))

    pruner_type = args.pruner_type.lower()
    if pruner_type == 'taylor':
        imp = mask_pruner.TaylorImportance(group_reduction=args.grouping_strategy, normalizer=args.imp_normalizer, taylor=args.taylor, model=model)
    elif pruner_type == "taylor+knowledge":
        imp1 = mask_pruner.TaylorImportance(group_reduction=args.grouping_strategy, normalizer=args.imp_normalizer, taylor=args.taylor)
        imp2 = mask_pruner.KnowledgeImportance(group_reduction="mean", normalizer=None)
        imp = [imp1, imp2]
    else:
        raise NotImplementedError
    
    # reconstruct qkv params in EVAVitAttention, for easier process
    for name, module in model.named_modules():
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
    torch.cuda.empty_cache()

    for param in model.parameters():
        param.requires_grad_(True)

    before_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    before_pruning_visual_parameters = sum(p.numel() for p in model.visual_encoder.parameters() if p.requires_grad)
    before_pruning_language_parameters = sum(p.numel() for p in model.t5_model.parameters() if p.requires_grad)
    raw_param_counts = count_param(model)
    
    for name, param in model.named_parameters():
        param.global_name = name # used for specific operation in pruners
        
    if args.num_examples == 0:
        args.num_examples = len(data_loader)
        
    if args.pruning_ratio <= 0.2: # for more fine-grained pruning
        args.channel_per_step = 100
    
    if args.granularity == "channel":
        raise NotImplementedError
    elif args.granularity == "block":
        model.t5_model.encoder.block[0].layer[0].SelfAttention.has_relative_attention_bias = False
        model.t5_model.decoder.block[0].layer[0].SelfAttention.has_relative_attention_bias = False
        kwargs = {
            "importance": imp,
            "global_pruning": args.global_pruning,
            "iterative_steps": args.iterative_steps,
            "ch_sparsity": args.pruning_ratio, 
            "ignored_layers":[],
            "max_pruning_ratio": args.max_pruning_ratio,
            "customized_pruners": {
                nn.Linear: linear_mask_pruner,
                nn.LayerNorm: layer_norm_mask_pruner,
                T5LayerNorm: t5_layer_norm_mask_pruner,
            },
            "root_module_types": [], 
            "channel_per_step": args.channel_per_step,
            'prune_num_heads': True,
            'prune_head_dims': False,
            'multimodal': args.multimodal,
        }    
        if args.imp_normalizer is None:
            kwargs['group_collect'] = 'sum'
        num_heads = {}
        for name, module in model.named_modules():
            if isinstance(module, T5Attention):
                num_heads[module.q] = module.n_heads
            if isinstance(module, EvaVitAttention):
                num_heads[module.q] = module.num_heads
        kwargs['num_heads'] = num_heads
        
        root_instances = [model.visual_encoder.blocks[i].attn.q for i in range(0, 39)] + \
                         [model.visual_encoder.blocks[i].mlp.fc1 for i in range(0, 39)] + \
                         [model.t5_model.encoder.block[i].layer[0].SelfAttention.q for i in range(0, 24)] + \
                         [model.t5_model.encoder.block[i].layer[1].DenseReluDense.wi_0 for i in range(0, 24)] + \
                         [model.t5_model.encoder.block[i].layer[1].DenseReluDense.wi_1 for i in range(0, 24)] + \
                         [model.t5_model.decoder.block[i].layer[0].SelfAttention.q for i in range(0, 24)] + \
                         [model.t5_model.decoder.block[i].layer[1].EncDecAttention.q for i in range(0, 24)] + \
                         [model.t5_model.decoder.block[i].layer[2].DenseReluDense.wi_0 for i in range(0, 24)] + \
                         [model.t5_model.decoder.block[i].layer[2].DenseReluDense.wi_1 for i in range(0, 24)] 
        kwargs['root_instances'] = root_instances
        
        mask_operation = {
            linear_mask_pruner.prune_in_channels: linear_mask_pruner.operate_in_masks, linear_mask_pruner.prune_out_channels: linear_mask_pruner.operate_out_masks,
            layer_norm_mask_pruner.prune_in_channels: layer_norm_mask_pruner.operate_in_masks, layer_norm_mask_pruner.prune_out_channels: layer_norm_mask_pruner.operate_out_masks,
            param_mask_pruner.prune_in_channels: param_mask_pruner.operate_in_masks, param_mask_pruner.prune_out_channels: param_mask_pruner.operate_out_masks,
            t5_layer_norm_mask_pruner.prune_in_channels: t5_layer_norm_mask_pruner.operate_in_masks, t5_layer_norm_mask_pruner.prune_out_channels: t5_layer_norm_mask_pruner.operate_out_masks,
        }
        kwargs['mask_operation'] = mask_operation

        trigger_prune_module = {}
        for i in range(24):
            tmp = model.t5_model.encoder.block[i].layer[0].SelfAttention
            trigger_prune_module[tmp.q] = (tmp, t5_attention_head_mask_pruner.operate_out_masks)
            tmp = model.t5_model.decoder.block[i].layer[0].SelfAttention
            trigger_prune_module[tmp.q] = (tmp, t5_attention_head_mask_pruner.operate_out_masks)
        kwargs['trigger'] = trigger_prune_module
        
        logger.log("Getting Pruner")
        pruner = tp.pruner.MaskPruner(
            model,
            example_input,
            forward_fn=forward_fn,
            **kwargs
        )
        
        model.t5_model.encoder.block[0].layer[0].SelfAttention.has_relative_attention_bias = True
        model.t5_model.decoder.block[0].layer[0].SelfAttention.has_relative_attention_bias = True
        
        if args.select_loss:
            model.t5_model.forward = partial(selected_t5_forward, model.t5_model)
        
        model.zero_grad()
        for param in model.parameters():
            param.grad = None
            
        if args.entropy_importance:    
            def new_process_imp_list(self, group, imp_list, ch_groups, remain_channels):
                _is_attn, qkv_layers = self._is_attn_group(group)
                group_size = len(imp_list[0]) // ch_groups
                if _is_attn and self.prune_num_heads:
                    for i in range(len(imp_list)):
                        if imp_list[i] is None: continue
                        if self.group_collect == 'mean':
                            imp_list[i] = imp_list[i].view(ch_groups, -1).mean(1) 
                        elif self.group_collect == 'sum':
                            imp_list[i] = imp_list[i].view(ch_groups, -1).sum(1)
                if self.is_visual_part(group):
                    imp = imp_list[0]
                else:
                    imp = imp_list[0]*imp_list[1]

                if _is_attn and self.prune_num_heads:
                    remain_channels = remain_channels.view(ch_groups, -1)[:,0].view(-1)
                imp[remain_channels==0] = float('inf')
                return imp, group_size
            pruner.process_imp_list = partial(new_process_imp_list, pruner)

            def forward_projection_save_hook(module, input, output):
                weights = module.weight.t()
                x_flat = input[0].view(-1, input[0].size(-1))
                weights_norm = F.normalize(weights, p=2, dim=0)
                x_norm = F.normalize(x_flat, p=2, dim=1)
                sim = torch.matmul(x_norm, weights_norm)
                module.out_dim_vals.append(sim)

            logger.log("Start Computing knowledge importance")
            with torch.no_grad():
                for module_name_list in [['.q', '.k', '.v', 'wi_0', 'wi_1']]:
                    module_list, hook_list = [], []
                    for name, module in model.t5_model.named_modules():
                        if not isinstance(module, nn.Linear):
                            continue
                        for subname in module_name_list:
                            if subname in name:
                                module_list.append(module)
                                module.out_dim_vals = []
                                hook_list.append(module.register_forward_hook(forward_projection_save_hook))
                                break
                    loader = iter(data_loader)
                    for i in range(args.num_examples // calibration_bs):
                        batch_data = next(loader)              
                        inputs = prepare_sample(batch_data, args.device)
                        model(inputs)
                    for module in module_list:
                        out_dim_vals = torch.cat(module.out_dim_vals, dim=0).to(torch.float)
                        entropies = []
                        for d in range(out_dim_vals.shape[-1]):
                            hist = torch.histc(out_dim_vals[:,d], bins=100, min=-1, max=1)
                            prob = hist / hist.sum(dim=0, keepdim=True)
                            entropy = -torch.sum(prob * torch.log(prob + 1e-12), dim=0)
                            entropies.append(entropy)
                        entropies = torch.stack(entropies)
                        module.out_dim_entropy = entropies
                        del out_dim_vals, module.out_dim_vals
                        torch.cuda.empty_cache()
                    for hook_handle in hook_list:
                        hook_handle.remove()
                    torch.cuda.empty_cache()
            torch.cuda.empty_cache()
        model.zero_grad()
        logger.log("Start Computing Grad...")
        i, total_token_cnt, used_token_cnt = 0, 0, 0
        loader = iter(data_loader)
        for i in range(args.num_examples // calibration_bs):
            batch_data = next(loader)
            # for idx in range(len(batch_data['image_path'])):
            #     print("data", i, batch_data['text_input'][idx] + "*" + batch_data['text_output'][idx])
            inputs = prepare_sample(batch_data, args.device)
            out = model(inputs)
            used_token_cnt += out['logits'].shape[0]
            total_token_cnt += out['labels'].shape[-1]
            loss = out['loss']
            logger.log("Loss = {}".format(loss))
            loss.backward()
            i += 1
        logger.log("Total forward data: {}, forward tokens: {}, used tokens: {}".format(i, total_token_cnt, used_token_cnt))   
        
        logger.log("Start Pruning...")
        if args.global_pruning:
            iter_steps, pruned_ratio = 0, 0.0
            while pruned_ratio < args.pruning_ratio:
                if pruned_ratio > 0.9*args.pruning_ratio:
                    pruner.channel_per_step = 100
                iter_steps += 1
                pruner.step()
                after_pruning_parameters = sum(torch.prod(torch.tensor([mask.sum() for mask in p.preserve_masks])) for p in model.parameters())
                pruned_ratio = 1 - 1.0*after_pruning_parameters / before_pruning_parameters
                after_pruning_visual_parameters = sum(torch.prod(torch.tensor([mask.sum() for mask in p.preserve_masks])) for p in model.visual_encoder.parameters())
                visual_pruned_ratio = 1 - 1.0*after_pruning_visual_parameters / before_pruning_visual_parameters
                after_pruning_language_parameters = sum(torch.prod(torch.tensor([mask.sum() for mask in p.preserve_masks])) for p in model.t5_model.parameters())
                language_pruned_ratio = 1 - 1.0*after_pruning_language_parameters / before_pruning_language_parameters
                logger.log("After Iter {}, #parameters: {}, #ratio: {:.4f}, #vision ratio: {:.2f}, #language ratio: {:.2f}".format(iter_steps, after_pruning_parameters, pruned_ratio*100, visual_pruned_ratio*100, language_pruned_ratio*100))
        else:
            pruner.step()
            after_pruning_parameters = sum(torch.prod(torch.tensor([mask.sum() for mask in p.preserve_masks])) for p in model.parameters())
            pruned_ratio = 1 - 1.0*after_pruning_parameters / before_pruning_parameters
            after_pruning_visual_parameters = sum(torch.prod(torch.tensor([mask.sum() for mask in p.preserve_masks])) for p in model.visual_encoder.parameters())
            visual_pruned_ratio = 1 - 1.0*after_pruning_visual_parameters / before_pruning_visual_parameters
            after_pruning_language_parameters = sum(torch.prod(torch.tensor([mask.sum() for mask in p.preserve_masks])) for p in model.t5_model.parameters())
            language_pruned_ratio = 1 - 1.0*after_pruning_language_parameters / before_pruning_language_parameters
            logger.log("After 1 Iter Local Pruning, #parameters: {}, #ratio: {:.4f}, #vision ratio: {:.2f}, #language ratio: {:.2f}".format(after_pruning_parameters, pruned_ratio*100, visual_pruned_ratio*100, language_pruned_ratio*100))
       
        model.zero_grad()
        for name, module in model.named_parameters():
            if 'weight' in name:
                module.grad = None
        
        # Finally compress weight matrix with masks
        pruner.compress_matrix()
        del pruner
        
        for name, module in model.named_modules():
            if isinstance(module, T5Attention):
                module.n_heads = module.q.weight.data.shape[0] // module.key_value_proj_dim
                module.inner_dim = module.q.weight.data.shape[0]
            if isinstance(module, EvaVitAttention):
                module.num_heads = module.q.weight.data.shape[0] // module.head_dim

        logger.log("Saving Prune Masks...")
        mask_dict = {}
        for name, param in model.named_parameters():
            if hasattr(param, 'preserve_masks'):
                mask_dict[name] = [tensor.cpu().numpy().tolist() for tensor in param.preserve_masks]
                del param.preserve_masks
        json.dump(mask_dict, open(os.path.join(logger.sub_dir, 'prune_masks.json'), "w"))
                
    else:
        raise NotImplementedError

    after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log("#Param before: {}, #Param after: {}, #Ratio = {:.4f}%".format(before_pruning_parameters, after_pruning_parameters,  100.0*after_pruning_parameters/before_pruning_parameters))

    if args.select_loss:
        model.t5_model.forward = partial(T5ForConditionalGeneration.forward, model.t5_model)

    logger.log("Saving Pruned Model...")
    torch.save({'model': model}, logger.checkpoint_path)
    new_param_counts = count_param(model)
    save_dic = {
        'raw': raw_param_counts,
        'new': new_param_counts
    }
    logger.log("Saving Param Ratios...")
    json.dump(save_dic, open(os.path.join(logger.sub_dir, 'param_counts.json'), "w"))
        
    logger.log('Model After Pruning:')
    logger.log(model)
    new_flop = FlopCountAnalysis(model, example_input)
    logger.log(flop_count_table(new_flop, max_depth=2, show_param_shapes=True))
    logger.log("Total Flops: "+str(new_flop.total() / 1e9))
    logger.log("FLOPs pruning ratio: {:.4f}%".format(new_flop.total()*100.0/raw_flop.total()))
    logger.log("[FINISH] - Finish Pruning Model")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='UKMP')
    
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    
    parser.add_argument('--seed', type=int, default=42, help='seed')
    
    parser.add_argument("--job_id", type=str, default=None, help="The id of the Job")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")

    parser.add_argument('--save_ckpt_log_name', type=str, default="llama_prune", help='the path for save the checkpoint and the log')
    parser.add_argument('--pruning_ratio', type=float, default=0.5, help='pruning ratio')
    parser.add_argument('--max_pruning_ratio', type=float, default=0.9, help='max pruning ratio of each group')
    parser.add_argument('--pruner_type', type=str, default='taylor', help='pruner type')
    
    parser.add_argument('--granularity', type=str, default='channel', help='prune granularity')
    parser.add_argument('--imp_normalizer', type=str, default=None, help='importance normalizer')

    parser.add_argument('--iterative_steps', type=int, default=1, help="Iteration step for baseline pruning. Disabled for the proposed method.")
    parser.add_argument('--channel_per_step', type=int, default=1000, help="Channels per step, for iterative pruning.")
    
    parser.add_argument('--grouping_strategy', type=str, default='sum', help='Reduce method for grouping.')
    parser.add_argument('--global_pruning', action='store_true', help='Whether global pruning.')
    parser.add_argument('--taylor', type=str, default='param_first', help='choose from [vectorize, param_second, param_first, param_mix]')
    parser.add_argument('--num_examples', type=int, default=1000, help='Calibration dataset size.')

    parser.add_argument('--device', type=str, default="cuda", help='device')

    parser.add_argument('--select_loss', action='store_true', help='Whether use selected loss')
    parser.add_argument('--entropy_importance', action='store_true', help='Whether use entropy importance')
    
    parser.add_argument('--calibration_bs', type=int, default=1, help='batch size of calibration dataset')
    parser.add_argument('--multimodal', action='store_true', help='Whether normalize imp for multimodal pruning.')
    
    args = parser.parse_args()
    args.torch_version = float('.'.join(torch.__version__.split('.')[:2]))
    
    main(args)
