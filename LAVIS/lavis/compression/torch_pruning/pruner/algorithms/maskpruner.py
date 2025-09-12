"""
 Inherited from lavis/compression/torch_pruning/pruner/algorithms/metapruner.py
 Modified by Zimeng Wu in 2025
 - Added support for the UKMP method.
""" 

import typing
from copy import deepcopy
from functools import partial

import torch
import torch.nn as nn

from ... import ops
from .. import function
from .metapruner import MetaPruner
from .scheduler import linear_scheduler


class MaskPruner(MetaPruner):
    def __init__(
        self,
        # Basic
        model: nn.Module, # a simple pytorch model
        example_inputs: torch.Tensor, # a dummy input for graph tracing. Should be on the same 
        importance: typing.Callable, # tp.importance.Importance for group importance estimation
        global_pruning: bool = False, # https://pytorch.org/tutorials/intermediate/pruning_tutorial.html#global-pruning.
        pruning_ratio: float = 0.5,  # channel/dim pruning ratio, also known as pruning ratio
        pruning_ratio_dict: typing.Dict[nn.Module, float] = None, # layer-specific pruning ratio, will cover pruning_ratio if specified
        max_pruning_ratio: float = 1.0, # maximum pruning ratio. useful if over-pruning happens.
        iterative_steps: int = 1,  # for iterative pruning
        iterative_pruning_ratio_scheduler: typing.Callable = linear_scheduler, # scheduler for iterative pruning.
        ignored_layers: typing.List[nn.Module] = None, # ignored layers
        round_to: int = None,  # round channels to the nearest multiple of round_to

        # Advanced
        in_channel_groups: typing.Dict[nn.Module, int] = dict(), # The number of channel groups for layer input
        out_channel_groups: typing.Dict[nn.Module, int] = dict(), # The number of channel groups for layer output
        num_heads: typing.Dict[nn.Module, int] = dict(), # The number of heads for multi-head attention
        prune_num_heads: bool = False, # remove entire heads in multi-head attention
        prune_head_dims: bool = True, # remove head dimensions in multi-head attention
        head_pruning_ratio: float = 0.0, # head pruning ratio
        head_pruning_ratio_dict: typing.Dict[nn.Module, float] = None, # layer-specific head pruning ratio
        customized_pruners: typing.Dict[typing.Any, function.BasePruningFunc] = None, # pruners for customized layers. E.g., {nn.Linear: my_linear_pruner}
        unwrapped_parameters: typing.Dict[nn.Parameter, int] = None, # unwrapped nn.Parameters & pruning_dims. For example, {ViT.pos_emb: 0}
        root_module_types: typing.List = [ops.TORCH_CONV, ops.TORCH_LINEAR, ops.TORCH_LSTM],  # root module for each group
        root_instances: typing.List = None, # added in llm-pruner
        forward_fn: typing.Callable = None, # a function to execute model.forward
        output_transform: typing.Callable = None, # a function to transform network outputs
        consecutive_groups: typing.Dict[nn.Module, int] = dict(), # for consecutive channels

        # deprecated
        channel_groups: typing.Dict[nn.Module, int] = dict(), # channel grouping
        ch_sparsity: float = None,
        ch_sparsity_dict: typing.Dict[nn.Module, float] = None, 
        
        # support UKMP
        channel_per_step: int = 0,
        trigger: dict = {},
        save_imp: bool = False,
        mask_operation: dict = None,
        group_collect: str = "mean",
        multimodal: bool = False,
        pre_imp: bool = False,
        imp_save_dir: str = None,
        multimodal_norm_type: str = "avg",
    ):
        
        super(MaskPruner, self).__init__(
            model=model,
            example_inputs=example_inputs,
            importance=importance,
            global_pruning=global_pruning,
            pruning_ratio=pruning_ratio,
            pruning_ratio_dict=pruning_ratio_dict,
            max_pruning_ratio=max_pruning_ratio,
            iterative_steps=iterative_steps,
            iterative_pruning_ratio_scheduler=iterative_pruning_ratio_scheduler,
            ignored_layers=ignored_layers,
            round_to=round_to,
            in_channel_groups=in_channel_groups,
            out_channel_groups=out_channel_groups,
            num_heads=num_heads,
            prune_num_heads=prune_num_heads,
            prune_head_dims=prune_head_dims,
            head_pruning_ratio=head_pruning_ratio,
            head_pruning_ratio_dict=head_pruning_ratio_dict,
            customized_pruners=customized_pruners,
            unwrapped_parameters=unwrapped_parameters,
            root_module_types=root_module_types,
            root_instances=root_instances,
            forward_fn=forward_fn,
            output_transform=output_transform,
            consecutive_groups=consecutive_groups,
            channel_groups=channel_groups,
            ch_sparsity=ch_sparsity,
            ch_sparsity_dict=ch_sparsity_dict,
            channel_per_step=channel_per_step,
            trigger=trigger,
            save_imp=save_imp,
            group_collect=group_collect,
            multimodal=multimodal,
            pre_imp=pre_imp,
        )
        
        self.mask_operation = mask_operation
        self.imp_save_dir = imp_save_dir
        self.multimodal_norm_type = multimodal_norm_type
        
        # # build masks for pruning
        for name, param in model.named_parameters():
            param.preserve_masks = []
            for i in param.size():
                param.preserve_masks.append(torch.ones(i, dtype=torch.int).to(param.device))   
        for group in self.DG.get_all_groups(ignored_layers=self.ignored_layers, root_module_types=self.root_module_types, root_instances=self.root_instances):
            for dep, idxs in group:
                dep.handler.__self__.pruning_dim = dep.target.pruning_dim
                layer = dep.target.module
                prune_fn = dep.handler     
                if prune_fn in self.mask_operation.keys():
                    self.mask_operation[prune_fn](layer, mode='init')  
        # to get remain channels
        from lavis.compression.torch_pruning.dependency import Node
        def masked_get_out_channels(self, module_or_node):
            if isinstance(module_or_node, Node):
                module = module_or_node.module
                pruning_dim = module_or_node.pruning_dim
            else:
                module = module_or_node
                pruning_dim = self.module2node[module].pruning_dim
            p = self.get_pruner_of_module(module)
            p.pruning_dim = pruning_dim
            if p is None:
                return None
            if hasattr(p, 'masked_get_out_channels'):
                return p.masked_get_out_channels(module)
            else:
                return None
        def masked_get_in_channels(self, module_or_node):
            if isinstance(module_or_node, Node):
                module = module_or_node.module
                pruning_dim = module_or_node.pruning_dim
            else:
                module = module_or_node
                pruning_dim = self.module2node[module].pruning_dim
            p = self.get_pruner_of_module(module)
            p.pruning_dim = pruning_dim
            if p is None:
                return None
            if hasattr(p, 'masked_get_in_channels'):
                return p.masked_get_in_channels(module)
            else:
                return None
        self.DG.get_out_channels = partial(masked_get_out_channels, self.DG)
        self.DG.get_in_channels = partial(masked_get_in_channels, self.DG)
    
    def compress_matrix(self):
        for group in self.DG.get_all_groups(ignored_layers=self.ignored_layers, root_module_types=self.root_module_types, root_instances=self.root_instances):
            for dep, idxs in group:
                prune_fn = dep.handler
                if dep.target.type == ops.OPTYPE.PARAMETER: 
                    old_parameter = dep.target.module
                    self.DG._param_to_name.pop(old_parameter)
                    name = old_parameter.global_name
                    dep.handler.__self__.pruning_dim = dep.target.pruning_dim
                    pruned_parameter = self.mask_operation[prune_fn](old_parameter, mode='compress')  
                    path = name.split('.')
                    module = self.model
                    for p in path[:-1]:
                        module = getattr(module, p)
                    setattr(module, path[-1], pruned_parameter)
                elif prune_fn in self.mask_operation.keys():
                    layer = dep.target.module
                    self.mask_operation[prune_fn](layer, mode='compress') 
                    
                layer = dep.target.module
                if layer in self.trigger.keys():
                    self.trigger[layer][1](self.trigger[layer][0], 'compress', (layer.weight.preserve_masks[0]==0).nonzero().view(-1))
    
    def prune_global_step(self) -> typing.Generator:
        ##############################################
        # 1. Pre-compute importance for each group
        ##############################################
        global_importance = []
        for group in self.DG.get_all_groups(ignored_layers=self.ignored_layers, root_module_types=self.root_module_types, root_instances=self.root_instances):
            if self._check_pruning_ratio(group):
                group = self._downstream_node_as_root_if_attention(group) # use a downstream node as the root node for attention layers
                ch_groups = self._get_channel_groups(group)
                imp = self.estimate_importance(group) # raw importance score
                remain_channels = self.get_remain_channels(group)
                raw_imp = deepcopy(imp)
                
                if isinstance(imp, list):
                    imp, group_size = self.process_imp_list(group, imp, ch_groups, remain_channels)
                    if imp is None:
                        continue
                else:
                    group_size = len(imp) // ch_groups
                    if imp is None:
                        continue

                    _is_attn, qkv_layers = self._is_attn_group(group)
                    if _is_attn and self.prune_num_heads:
                        if self.group_collect == 'mean':
                            imp = imp.view(ch_groups, -1).mean(1) # average importance by head. # TODO 
                        elif self.group_collect == 'sum':
                            imp = imp.view(ch_groups, -1).sum(1)
                        remain_channels = remain_channels.view(ch_groups, -1)[:,0].view(-1)
                    elif ch_groups > 1:
                        raise NotImplementedError
                    
                    imp[remain_channels==0] = float('inf')
                    
                global_importance.append((group, ch_groups, group_size, raw_imp, imp))
        
        ##############################################
        # 2. Thresholding by concatenating all importance scores
        ##############################################
        # Find the threshold for global pruning
        concat_imp = torch.cat([local_imp[-1][~torch.isinf(local_imp[-1])] for local_imp in global_importance], dim=0)
        topk_imp, _ = torch.topk(concat_imp, k=self.channel_per_step, largest=False)
        thres = topk_imp[-1]
        
        ##############################################
        # 3. Prune
        ##############################################
        for group, ch_groups, group_size, raw_imp, imp in global_importance:
            module = group[0].dep.target.module
            pruning_fn = group[0].dep.handler
            get_channel_fn = self.DG.get_out_channels if self.DG.is_out_channel_pruning_fn(pruning_fn) else self.DG.get_in_channels
            
            pruning_indices = []
            _is_attn, qkv_layers = self._is_attn_group(group)
            if _is_attn and self.prune_num_heads:
                head_pruning_indices = (imp <= thres).nonzero().view(-1)
                if len(head_pruning_indices)>0:
                    for head_id in head_pruning_indices:
                        pruning_indices.append( torch.arange(head_id*group_size, (head_id+1)*group_size, device=imp.device) )
                    
            elif ch_groups > 1:
                n_pruned_per_group = len((imp <= thres).nonzero().view(-1))
                if n_pruned_per_group>0:
                    if self.round_to:
                        n_pruned_per_group = self._round_to(n_pruned_per_group, group_size, self.round_to)

                    for chg in range(ch_groups):
                        sub_group_imp = raw_imp[chg*group_size: (chg+1)*group_size]
                        sub_imp_argsort = torch.argsort(sub_group_imp)
                        pruning_indices.append(sub_imp_argsort[:n_pruned_per_group]+chg*group_size)
            else:
                _pruning_indices = (imp <= thres).nonzero().view(-1)
                imp_argsort = torch.argsort(imp)
                if len(_pruning_indices)>0:
                    if self.round_to: 
                        n_pruned = len(_pruning_indices)
                        current_channels = get_channel_fn(module)
                        n_pruned = self._round_to(n_pruned, current_channels, self.round_to)
                        _pruning_indices = imp_argsort[:n_pruned]
                    pruning_indices.append(_pruning_indices)
            if len(pruning_indices)==0: continue
            pruning_indices = torch.unique(torch.cat(pruning_indices, 0)).tolist()
            # create pruning group
            group = self.DG.get_pruning_group(
                module, pruning_fn, pruning_indices)
            yield group
    
    def get_remain_channels(self, group):
        for dep, _ in group:
            module = dep.target.module
            prune_fn = dep.handler
            if hasattr(module, 'weight'):
                if self.DG.is_out_channel_pruning_fn(prune_fn):
                    return module.weight.preserve_masks[0]
                else:
                    return module.weight.preserve_masks[1]
        print(group)
        raise NotImplementedError 
        return None
    
    def process_imp_list(self, group, imp_list, ch_groups, remain_channels):
        _is_attn, qkv_layers = self._is_attn_group(group)
        grouped_imp = []
        for sub_imp in imp_list:
            if sub_imp is None:
                break
            group_size = len(sub_imp) // ch_groups
            if _is_attn and self.prune_num_heads:
                if self.group_collect == 'mean':
                    sub_imp = sub_imp.view(ch_groups, -1).mean(1) # average importance by head. # TODO 
                elif self.group_collect == 'sum':
                    sub_imp = sub_imp.view(ch_groups, -1).sum(1)
            elif ch_groups > 1:
                raise NotImplementedError
            grouped_imp.append(sub_imp)
        mean0, std0 = torch.mean(grouped_imp[0]), torch.std(grouped_imp[0])
        mean1, std1 = torch.mean(grouped_imp[1]), torch.std(grouped_imp[1])
        imp = (grouped_imp[0]-mean0)/std0 + (grouped_imp[1]-mean1)/std1
        if _is_attn and self.prune_num_heads:
            remain_channels = remain_channels.view(ch_groups, -1)[:,0].view(-1)
        imp[remain_channels==0] = float('inf')
        return imp, group_size
    
    def prune_global_step_multimodal(self) -> typing.Generator:
        ##############################################
        # 1. Pre-compute importance for each group
        ##############################################
        visual_global_importance, language_global_importance = [], []
        names_list, imps_list = [], []
        for group in self.DG.get_all_groups(ignored_layers=self.ignored_layers, root_module_types=self.root_module_types, root_instances=self.root_instances):
            if self._check_pruning_ratio(group):    
                module = group[0].dep.target.module
                group = self._downstream_node_as_root_if_attention(group) # use a downstream node as the root node for attention layers
                ch_groups = self._get_channel_groups(group)
                if (not self.pre_imp) or (self.pre_imp and (module not in self.pre_imp_dict.keys())):
                    imp = self.estimate_importance(group) # original importance score
                    remain_channels = self.get_remain_channels(group)
                    raw_imp = None # deepcopy(imp) # to reduce CUDA memory
                    if isinstance(imp, list):
                        imp, group_size = self.process_imp_list(group, imp, ch_groups, remain_channels)
                        if imp is None:
                            continue
                    else:
                        group_size = len(imp) // ch_groups
                        if imp is None:
                            continue

                        _is_attn, qkv_layers = self._is_attn_group(group)
                        if _is_attn and self.prune_num_heads:
                            if self.group_collect == 'mean':
                                imp = imp.view(ch_groups, -1).mean(1) # average importance by head. # TODO 
                            elif self.group_collect == 'sum':
                                imp = imp.view(ch_groups, -1).sum(1)
                            remain_channels = remain_channels.view(ch_groups, -1)[:,0].view(-1)
                        elif ch_groups > 1:
                            raise NotImplementedError
                        
                        imp[remain_channels==0] = float('inf')
                    if self.pre_imp:
                        self.pre_imp_dict[module] = (imp, group_size)
                else:
                    raw_imp = None
                    ch_groups = self._get_channel_groups(group)
                    imp, group_size = self.pre_imp_dict[module]
                    remain_channels = self.get_remain_channels(group)
                    _is_attn, qkv_layers = self._is_attn_group(group)
                    if _is_attn and self.prune_num_heads:
                        remain_channels = remain_channels.view(ch_groups, -1)[:,0].view(-1)
                    imp[remain_channels==0] = float('inf')
                    
                if self.is_visual_part(group):  
                    visual_global_importance.append((group, ch_groups, group_size, raw_imp, imp))
                else:
                    language_global_importance.append((group, ch_groups, group_size, raw_imp, imp))
        
        ##############################################
        # 2. Thresholding by concatenating all importance scores
        ##############################################
        # Find the threshold for global pruning
        concat_visual_imp = torch.cat([local_imp[-1][~torch.isinf(local_imp[-1])] for local_imp in visual_global_importance], dim=0)
        concat_language_imp = torch.cat([local_imp[-1][~torch.isinf(local_imp[-1])] for local_imp in language_global_importance], dim=0)
        global_importance = []
        
        if self.multimodal_norm_type == 'avg':
            visual_avg, language_avg = torch.mean(concat_visual_imp), torch.mean(concat_language_imp)
            for item in visual_global_importance:
                *other_elements, last_tensor = item
                global_importance.append((*other_elements, (last_tensor) / (visual_avg)))
            for item in language_global_importance:
                *other_elements, last_tensor = item
                global_importance.append((*other_elements, (last_tensor) / (language_avg)))
        elif self.multimodal_norm_type == 'minmax':
            visual_min, visual_max = torch.min(concat_visual_imp), torch.max(concat_visual_imp)
            language_min, language_max = torch.min(concat_language_imp), torch.max(concat_language_imp)
            for item in visual_global_importance:
                *other_elements, last_tensor = item
                global_importance.append((*other_elements, (last_tensor - visual_min) / (visual_max - visual_min)))
            for item in language_global_importance:
                *other_elements, last_tensor = item
                global_importance.append((*other_elements, (last_tensor - language_min) / (language_max - language_min)))
        else: 
            raise NotImplementedError
        
        concat_imp = torch.cat([local_imp[-1] for local_imp in global_importance], dim=0)
        topk_imp, _ = torch.topk(concat_imp, k=self.channel_per_step, largest=False)
        thres = topk_imp[-1]
        
        ##############################################
        # 3. Prune
        ##############################################
        for group, ch_groups, group_size, raw_imp, imp in global_importance:
            module = group[0].dep.target.module
            pruning_fn = group[0].dep.handler
            get_channel_fn = self.DG.get_out_channels if self.DG.is_out_channel_pruning_fn(pruning_fn) else self.DG.get_in_channels
            
            pruning_indices = []
            
            ###################
            _is_attn, qkv_layers = self._is_attn_group(group)
            if _is_attn and self.prune_num_heads:
                head_pruning_indices = (imp <= thres).nonzero().view(-1)
                if len(head_pruning_indices)>0:
                    for head_id in head_pruning_indices:
                        pruning_indices.append( torch.arange(head_id*group_size, (head_id+1)*group_size, device=imp.device) )       
            elif ch_groups > 1:
                raise NotImplementedError
            else:
                _pruning_indices = (imp <= thres).nonzero().view(-1)
                if len(_pruning_indices)>0:
                    pruning_indices.append(_pruning_indices)

            if len(pruning_indices)==0: continue
            pruning_indices = torch.unique(torch.cat(pruning_indices, 0)).tolist()
            group = self.DG.get_pruning_group(
                module, pruning_fn, pruning_indices)
            yield group
        del global_importance
        
    def prune_local(self) -> typing.Generator:
        assert self.iterative_steps == 1
        for group in self.DG.get_all_groups(ignored_layers=self.ignored_layers, root_module_types=self.root_module_types, root_instances=self.root_instances):
            if self._check_pruning_ratio(group): # check pruning ratio
                ##################################
                # Compute raw importance score
                ##################################
                group = self._downstream_node_as_root_if_attention(group)
                module = group[0].dep.target.module
                pruning_fn = group[0].dep.handler
                ch_groups = self._get_channel_groups(group)
                imp = self.estimate_importance(group) # raw importance score
                group_size = len(imp) // ch_groups
                if imp is None: continue
                
                _is_attn, qkv_layers = self._is_attn_group(group)
                if _is_attn and self.prune_num_heads:
                    if self.group_collect == 'mean':
                        imp = imp.view(ch_groups, -1).mean(1) # average importance by head. # TODO 
                    elif self.group_collect == 'sum':
                        imp = imp.view(ch_groups, -1).sum(1)
                elif ch_groups > 1:
                    raise NotImplementedError

                target_pruning_ratio = self.get_target_pruning_ratio(module)
                n_pruned = int(len(imp)*target_pruning_ratio)
                
                imp_argsort = torch.argsort(imp)
                pruning_indices = []
                if _is_attn and self.prune_num_heads:
                    head_pruning_indices = imp_argsort[:n_pruned]
                    for head_id in head_pruning_indices:
                        pruning_indices.append( torch.arange(head_id*group_size, (head_id+1)*group_size, device=imp.device) )
                elif ch_groups > 1:
                    raise NotImplementedError
                else:
                    pruning_indices.append(imp_argsort[:n_pruned])
                if len(pruning_indices) == 0: continue
                
                pruning_indices = torch.unique(torch.cat(pruning_indices, 0)).tolist()
                group = self.DG.get_pruning_group(module, pruning_fn, pruning_indices)
                yield group
                
    def compute_the_sparsity_per_group(self, total_parameters_to_keep, group_scores, group_num_parameters, max_sparsity_per_layer=0.8):
        scores = torch.FloatTensor(list(group_scores.values()))
        num_parameters = torch.LongTensor(list(group_num_parameters.values()))
        
        parameters_to_keep_per_group = torch.zeros_like(scores, dtype=int)
        
        parameters_to_keep_per_group += torch.ceil(num_parameters * (1 - max_sparsity_per_layer)).int() # to gaurantee the max_sparsity
        
        while parameters_to_keep_per_group.sum() < total_parameters_to_keep:
            total_ratio = torch.sum(scores)
            
            rest_total_parameters_to_keep = total_parameters_to_keep - parameters_to_keep_per_group.sum()
            
            parameters_to_add = torch.ceil((scores / total_ratio) * rest_total_parameters_to_keep)
            
            parameters_to_keep_per_group = parameters_to_keep_per_group + parameters_to_add
            
            scores[parameters_to_keep_per_group >= num_parameters] = 0 # make sure they are not going to add more parameters
            
            parameters_to_keep_per_group = torch.clamp(parameters_to_keep_per_group, max=num_parameters) # remove the extra parameters

            # they are to make sure the sum of parameters_to_keep_per_group is EXACTLY the same as total_parameters_to_keep
            if parameters_to_add.sum() == 0: # for some reason the algo cannot add more parameters
                # the algo stuck
                current_sum = parameters_to_keep_per_group.sum()
                if current_sum < total_parameters_to_keep:
                    num_need_to_add = total_parameters_to_keep - current_sum
                    
                    while num_need_to_add > 0:
                        # distributed the parameters to the rest of groups
                        for index in torch.where(scores > 0)[0]:
                            parameters_can_add = min(
                                num_need_to_add, num_parameters[index] - parameters_to_keep_per_group[index]
                            )
                            parameters_to_keep_per_group[index] += parameters_can_add
                            
                            num_need_to_add -= parameters_can_add
                            
                            if num_need_to_add == 0:
                                break
                            
            if parameters_to_keep_per_group.sum() > total_parameters_to_keep: # for some reason the algo cannot add more parameters
                # the algo stuck
                current_sum = parameters_to_keep_per_group.sum()

                num_need_to_remove = current_sum - total_parameters_to_keep
                
                while num_need_to_remove > 0:
                    # remove the parameters from full groups
                    for index in torch.argsort(parameters_to_keep_per_group, descending=True, stable=True):
                        parameters_can_remove = min(
                            num_need_to_remove, 
                            parameters_to_keep_per_group[index] - (num_parameters[index] * (1 - max_sparsity_per_layer)).int() # extra parameters
                        )
                        parameters_to_keep_per_group[index] += parameters_can_remove
                        
                        num_need_to_remove -= parameters_can_remove
                        
                        if num_need_to_remove == 0:
                            break
                        
        # convert the group parameters to keep to sparsity    
        group_sparsity = {}
        
        for k, param_to_keep, group_max_param in zip(group_num_parameters.keys(), parameters_to_keep_per_group, num_parameters):
            group_sparsity[k] = torch.clamp(1 - param_to_keep / group_max_param, min=0, max=1).item()
            
        return group_sparsity
                
    def pre_estimate_ratio(self, mode, param_to_prune, group_dic):
        assert mode == 'ecoflap' # TODO: only support structured ECoFLaP now!
        group_scores, group_num_parameters = {k: 0 for k in group_dic.values()}, {k: 0 for k in group_dic.values()}
        param_sum = 0
        for group in self.DG.get_all_groups(ignored_layers=self.ignored_layers, root_module_types=self.root_module_types, root_instances=self.root_instances):
            group = self._downstream_node_as_root_if_attention(group) # use a downstream node as the root node for attention layers
            module0 = group[0].dep.target.module
            group_idx = group_dic[module0]
            imp = self.estimate_importance(group) # raw importance score
            imp = imp.sum()
            group_scores[group_idx] += imp
            param_cnt = 0
            for dep, idx in group:
                module = dep.target.module
                param_cnt += sum(p.numel() for p in module.parameters() if p.requires_grad)
            group_num_parameters[group_idx] += param_cnt
            param_sum += param_cnt   

        total_param_to_keep = param_sum - param_to_prune
        
        for k, v in group_scores.items():
            print(k, v)

        sparse_dic = self.compute_the_sparsity_per_group(total_param_to_keep, group_scores, group_num_parameters, round(1.0 - self.pruning_ratio + 0.1, 1)) 
        for k, v in sparse_dic.items():
            print(k, v)
        for group in self.DG.get_all_groups(ignored_layers=self.ignored_layers, root_module_types=self.root_module_types, root_instances=self.root_instances):
            module0 = group[0].dep.target.module
            self.pruning_ratio_dict[module0] = [0, sparse_dic[group_dic[module0]]]
            