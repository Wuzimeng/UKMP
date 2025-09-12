"""
 Inherited from Torch-Pruning.
 Modified by Zimeng Wu in 2025
 - Added support for the UKMP method.
"""

from copy import deepcopy
from typing import Callable, Sequence, Tuple, Dict

import torch
import torch.nn as nn

import lavis.compression.torch_pruning as tp
from lavis.compression.torch_pruning import BasePruningFunc, ops

from abc import ABC, abstractclassmethod


##############################
# Pruners
##############################
class BaseMaskPruningFunc(ABC):
    TARGET_MODULES = ops.TORCH_OTHERS  # None

    def __init__(self, pruning_dim=1):
        self.pruning_dim = pruning_dim

    @abstractclassmethod
    def prune_out_channels(self, layer: nn.Module, idxs: Sequence[int]):
        raise NotImplementedError

    @abstractclassmethod
    def prune_in_channels(self, layer: nn.Module, idxs: Sequence[int]):
        raise NotImplementedError

    @abstractclassmethod
    def get_out_channels(self, layer: nn.Module):
        raise NotImplementedError

    @abstractclassmethod
    def get_in_channels(self, layer: nn.Module):
        raise NotImplementedError
    
    @abstractclassmethod
    def masked_get_out_channels(self, layer: nn.Module):
        raise NotImplementedError

    @abstractclassmethod
    def masked_get_in_channels(self, layer: nn.Module):
        raise NotImplementedError

    def check(self, layer, idxs, to_output):
        if self.TARGET_MODULES is not None:
            assert isinstance(layer, self.TARGET_MODULES), 'Mismatched pruner {} and module {}'.format(
                self.__str__, layer)
        if to_output:
            prunable_channels = self.get_out_channels(layer)
        else:
            prunable_channels = self.get_in_channels(layer)
        if prunable_channels is not None:
            assert all(idx < prunable_channels and idx >=
                       0 for idx in idxs), "All pruning indices should fall into [{}, {})".format(0, prunable_channels)

    def __call__(self, layer: nn.Module, idxs: Sequence[int], to_output: bool = True, inplace: bool = True, dry_run: bool = False) -> Tuple[nn.Module, int]:
        idxs.sort()
        self.check(layer, idxs, to_output)
        pruning_fn = self.prune_out_channels if to_output else self.prune_in_channels
        if not inplace:
            layer = deepcopy(layer)
        layer = pruning_fn(layer, idxs)
        return layer

    def get_in_channel_groups(self, layer):
        return 1
    
    def get_out_channel_groups(self, layer):
        return 1

    def _prune_parameter_and_grad(self, weight, idxs, pruning_dim):
        weight.preserve_masks[pruning_dim][idxs] = 0
        return
    
    def _compress_parameter(self, weight, keep_idxs, pruning_dim):
        pruned_weight = torch.nn.Parameter(torch.index_select(weight, pruning_dim, keep_idxs))
        if hasattr(weight, 'global_name'):
            pruned_weight.global_name = weight.global_name
        if hasattr(weight, 'preserve_masks'):
            pruned_weight.preserve_masks = weight.preserve_masks
        return pruned_weight.to(weight.device)

class LinearMaskPruner(BaseMaskPruningFunc):
    TARGET_MODULES = ops.TORCH_LINEAR

    def prune_out_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        if "qkv" in layer.weight.global_name: # TODO
            l = layer.out_features // 3
            new_idxs = []
            for i in idxs:
                new_idxs.append(i)
                new_idxs.append(i+l)
                new_idxs.append(i+l*2)
                idxs = new_idxs
        idxs = list(set(idxs))
        idxs.sort()
        self._prune_parameter_and_grad(layer.weight, idxs, 0)
        if layer.bias is not None:
            self._prune_parameter_and_grad(layer.bias, idxs, 0)
        return layer

    def prune_in_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        idxs.sort()
        self._prune_parameter_and_grad(layer.weight, idxs, 1)
        return layer

    def get_out_channels(self, layer):
        if "qkv" in layer.weight.global_name: # TODO
            return layer.out_features // 3
        return layer.out_features
    
    def masked_get_out_channels(self, layer):
        if "qkv" in layer.weight.global_name: # TODO
            return layer.weight.preserve_masks[0].sum() // 3
        return layer.weight.preserve_masks[0].sum()

    def get_in_channels(self, layer):
        return layer.in_features
    
    def masked_get_in_channels(self, layer):
        return layer.weight.preserve_masks[1].sum()
    
    def operate_out_masks(self, layer, mode):
        if mode == 'init':
            layer.weight.preserve_masks[0].fill_(1)
        elif mode == 'compress':
            keep_idxs = layer.weight.preserve_masks[0].nonzero().view(-1)
            layer.weight = self._compress_parameter(layer.weight, keep_idxs, 0)
            if layer.bias is not None:
                layer.bias = self._compress_parameter(layer.bias, keep_idxs, 0)
            layer.out_features = len(keep_idxs)
        else:
            raise NotImplementedError
        return layer
    
    def operate_in_masks(self, layer, mode):
        if mode == 'init':
            layer.weight.preserve_masks[1].fill_(1)
        elif mode == 'compress':
            keep_idxs = layer.weight.preserve_masks[1].nonzero().view(-1)           
            layer.weight = self._compress_parameter(layer.weight, keep_idxs, 1)
            layer.in_features = len(keep_idxs)
        else:
            raise NotImplementedError
        return layer

class LayernormMaskPruner(BaseMaskPruningFunc):
    TARGET_MODULES = ops.TORCH_LAYERNORM

    def __init__(self, metrcis=None, pruning_dim=-1):
        super().__init__(metrcis)
        self.pruning_dim = pruning_dim

    def check(self, layer, idxs):
        layer.dim = self.pruning_dim

    def prune_out_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        pruning_dim = self.pruning_dim
        if len(layer.normalized_shape) < -pruning_dim:
            return layer
        idxs.sort()
        if layer.elementwise_affine:
            self._prune_parameter_and_grad(layer.weight, idxs, pruning_dim)
            if layer.bias is not None:
                self._prune_parameter_and_grad(layer.bias, idxs, pruning_dim)
        return layer

    prune_in_channels = prune_out_channels
    
    def operate_out_masks(self, layer, mode):
        if mode == 'init':
            layer.weight.preserve_masks[self.pruning_dim].fill_(1)
        elif mode == 'compress':
            keep_idxs = layer.weight.preserve_masks[self.pruning_dim].nonzero().view(-1)
            if layer.elementwise_affine:
                layer.weight = self._compress_parameter(layer.weight, keep_idxs, self.pruning_dim)
                if layer.bias is not None:
                    layer.bias = self._compress_parameter(layer.bias, keep_idxs, self.pruning_dim)
            if self.pruning_dim != -1:  
                layer.normalized_shape = layer.normalized_shape[:self.pruning_dim] + (
                    keep_idxs.size(0), ) + layer.normalized_shape[self.pruning_dim+1:]
            else:
                layer.normalized_shape = layer.normalized_shape[:self.pruning_dim] + (
                    keep_idxs.size(0), )
        else:
            raise NotImplementedError
        return layer
    operate_in_masks = operate_out_masks

    def get_out_channels(self, layer):
        return layer.normalized_shape[self.pruning_dim]
    
    def masked_get_out_channels(self, layer):
        return layer.weight.preserve_masks[self.pruning_dim].sum()

    def get_in_channels(self, layer):
        return layer.normalized_shape[self.pruning_dim]
    
    def masked_get_in_channels(self, layer):
        return layer.weight.preserve_masks[self.pruning_dim].sum()
    
class ParameterMaskPruner(BaseMaskPruningFunc):
    TARGET_MODULES = ops.TORCH_PARAMETER
    def __init__(self, pruning_dim=-1):
        super().__init__(pruning_dim=pruning_dim)
        
    def prune_out_channels(self, tensor, idxs: list) -> nn.Module:
        idxs.sort()
        self._prune_parameter_and_grad(tensor, idxs, self.pruning_dim)
        return tensor

    prune_in_channels = prune_out_channels

    def get_out_channels(self, parameter):
        return parameter.shape[self.pruning_dim]
    
    def masked_get_out_channels(self, parameter):
        return parameter.preserve_masks[self.pruning_dim].sum()

    def get_in_channels(self, parameter):
        return parameter.shape[self.pruning_dim]
    
    def masked_get_in_channels(self, parameter):
        return parameter.preserve_masks[self.pruning_dim].sum()

    def operate_out_masks(self, parameter, mode):
        if mode == 'init':
            parameter.preserve_masks[self.pruning_dim].fill_(1)
        elif mode == 'compress':
            keep_idxs = parameter.preserve_masks[self.pruning_dim].nonzero().view(-1)
            pruned_parameter = self._compress_parameter(parameter, keep_idxs, self.pruning_dim)
            return pruned_parameter
        else:
            raise NotImplementedError
    operate_in_masks = operate_out_masks

class T5LayerNormMaskPruner(BaseMaskPruningFunc):

    def prune_out_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        idxs.sort()
        self._prune_parameter_and_grad(layer.weight, idxs, 0)
        return layer

    prune_in_channels = prune_out_channels

    def get_out_channels(self, layer):
        return layer.weight.size(0)
    
    def masked_get_out_channels(self, layer):
        return layer.weight.preserve_masks[0].sum()

    def get_in_channels(self, layer):
        return layer.weight.size(0)   
    
    def masked_get_in_channels(self, layer):
        return layer.weight.preserve_masks[0].sum()
    
    def operate_out_masks(self, layer, mode):
        if mode == 'init':
            layer.weight.preserve_masks[0].fill_(1)
        elif mode == 'compress':
            keep_idxs = layer.weight.preserve_masks[0].nonzero().view(-1)
            layer.weight = self._compress_parameter(layer.weight, keep_idxs, 0)
        else:
            raise NotImplementedError
        return layer
    operate_in_masks = operate_out_masks
    
class T5AttentionHeadMaskPruner(BaseMaskPruningFunc):

    def prune_out_channels(self, layer: nn.Module, idxs: Sequence[int]) -> nn.Module:
        raise NotImplementedError("Can't be called")
        prune_head_idxs = list(set(idx//layer.key_value_proj_dim for idx in idxs))
        remain_head_idxs = [i for i in range(32) if i not in layer.pruned_heads]
        for idx in prune_head_idxs:
            layer.pruned_heads.add(remain_head_idxs[idx])
        return layer

    prune_in_channels = prune_out_channels

    def get_out_channels(self, layer):
        raise NotImplementedError("Only a triggered function, can't be called")

    def get_in_channels(self, layer):
        raise NotImplementedError("Only a triggered function, can't be called")
    
    def masked_get_out_channels(self, layer):
        raise NotImplementedError("Only a triggered function, can't be called")

    def masked_get_in_channels(self, layer):
        raise NotImplementedError("Only a triggered function, can't be called")
    
    def operate_out_masks(self, layer, mode, prune_idxs):
        if mode == 'init':
            raise NotImplementedError("Can't be called")
        elif mode == 'compress':
            layer.pruned_heads = set(list((idx//layer.key_value_proj_dim).item() for idx in prune_idxs))
        else:
            raise NotImplementedError
  
linear_mask_pruner = LinearMaskPruner()
layer_norm_mask_pruner = LayernormMaskPruner()

from ..torch_pruning.pruner.function import PrunerBox
from ..torch_pruning.pruner import function
PrunerBox[ops.OPTYPE.PARAMETER] = ParameterMaskPruner()
param_mask_pruner = PrunerBox[ops.OPTYPE.PARAMETER]
function.prune_parameter_in_channels = param_mask_pruner.prune_in_channels
function.prune_parameter_out_channels = param_mask_pruner.prune_out_channels

t5_layer_norm_mask_pruner = T5LayerNormMaskPruner() 
t5_attention_head_mask_pruner = T5AttentionHeadMaskPruner()

##############################
# Importance
##############################
class TaylorImportance(tp.importance.Importance):
    def __init__(self, group_reduction="sum", normalizer=None, taylor=None, model=None):
        self.group_reduction = group_reduction
        self.normalizer = normalizer
        self.taylor = taylor
        self.model = model

    def _reduce(self, group_imp):
        if self.group_reduction == "sum":
            group_imp = group_imp.sum(dim=0)
        elif self.group_reduction == "mean":
            group_imp = group_imp.mean(dim=0)
        elif self.group_reduction == "max":
            group_imp = group_imp.max(dim=0)[0]
        elif self.group_reduction == "prod":
            group_imp = torch.prod(group_imp, dim=0)
        elif self.group_reduction=='first':
            group_imp = group_imp[0]
        elif self.group_reduction=='second':
            group_imp = group_imp[1]
        elif self.group_reduction is None:
            group_imp = group_imp
        else: 
            raise NotImplementedError
        return group_imp
    
    def _normalize(self, group, group_imp):
        if self.normalizer == "param":
            params = []
            for dep, idxs in group:
                idxs.sort()
                layer = dep.target.module
                prune_fn = dep.handler
                # Linear out_channels
                if prune_fn in [linear_mask_pruner.prune_out_channels]:
                    if "qkv" in layer.weight.global_name: # TODO
                        params.append(len(layer.weight.preserve_masks[1])*3)
                    else:
                        params.append(len(layer.weight.preserve_masks[1]))

                # Linear in_channels
                elif prune_fn in [linear_mask_pruner.prune_in_channels]:
                    params.append(len(layer.weight.preserve_masks[0]))

                elif prune_fn in [t5_layer_norm_mask_pruner.prune_out_channels, t5_layer_norm_mask_pruner.prune_in_channels]:
                    params.append(1)

            return group_imp / sum(params)
        elif self.normalizer is not None:
            raise NotImplementedError
        else:
            return group_imp

    @torch.no_grad()
    def __call__(self, group, ch_groups=1, consecutive_groups=1):
        group_imp = []
        for dep, idxs in group:
            idxs.sort()
            layer = dep.target.module
            prune_fn = dep.handler

            if prune_fn not in [
                linear_mask_pruner.prune_in_channels, linear_mask_pruner.prune_out_channels,
            ]:
                continue

            salience = layer.weight * layer.weight.grad

            if self.taylor in ['param_second']:
                salience = layer.weight * layer.weight.grad_acc2 * layer.weight
            elif self.taylor in ['param_mix']: 
                salience = salience - 0.5 * layer.weight * layer.weight.grad_acc2 * layer.weight
            
            # Linear out_channels
            if prune_fn in [linear_mask_pruner.prune_out_channels]:
                # salience[layer.weight.preserve_masks[0]==0, :] = 0
                if self.taylor == 'vectorize':
                    local_norm = salience.sum(1).abs()
                elif 'param' in self.taylor:
                    local_norm = salience.abs().sum(1)
                else:
                    raise NotImplementedError
                # local_norm[layer.weight.preserve_masks[0]!=0]+=(1e-12)
                group_imp.append(local_norm)

            # Linear in_channels
            elif prune_fn in [linear_mask_pruner.prune_in_channels]:
                # salience[:, layer.weight.preserve_masks[1]==0] = 0
                if self.taylor == 'vectorize':
                    local_norm = salience.sum(0).abs()
                elif 'param' in self.taylor:
                    local_norm = salience.abs().sum(0)
                else:
                    raise NotImplementedError
                # local_norm[layer.weight.preserve_masks[1]!=0]+=(1e-12)
                group_imp.append(local_norm)

            else:
                raise NotImplementedError
       
        if len(group_imp)==0:
            return None
        min_imp_size = min([len(imp) for imp in group_imp])
        aligned_group_imp = []
        for imp in group_imp:
            if len(imp)>min_imp_size and len(imp)%min_imp_size==0:
                imp = imp.view(len(imp) // min_imp_size, min_imp_size).sum(0)
                aligned_group_imp.append(imp)
            elif len(imp)==min_imp_size:
                aligned_group_imp.append(imp)
        group_imp = torch.stack(aligned_group_imp, dim=0)
        group_imp = self._reduce(group_imp)
        group_imp = self._normalize(group, group_imp)
            
        return group_imp


class WandaImportance(tp.importance.Importance):
    def __init__(self, group_reduction="sum"):
        self.group_reduction = group_reduction

    def _reduce(self, group_imp):
        if self.group_reduction == "sum":
            group_imp = group_imp.sum(dim=0)
        elif self.group_reduction == "mean":
            group_imp = group_imp.mean(dim=0)
        elif self.group_reduction == "max":
            group_imp = group_imp.max(dim=0)[0]
        elif self.group_reduction == "prod":
            group_imp = torch.prod(group_imp, dim=0)
        elif self.group_reduction=='first':
            group_imp = group_imp[0]
        elif self.group_reduction=='second':
            group_imp = group_imp[1]
        elif self.group_reduction is None:
            group_imp = group_imp
        else: 
            raise NotImplementedError
        return group_imp
    
    def _normalize(self, group, group_imp):
        return group_imp

    @torch.no_grad()
    def __call__(self, group, ch_groups=1, consecutive_groups=1):
        group_imp = []
        for dep, idxs in group:
            idxs.sort()
            layer = dep.target.module
            prune_fn = dep.handler

            if prune_fn not in [
                linear_mask_pruner.prune_in_channels, linear_mask_pruner.prune_out_channels,
            ]:
                continue

            salience = layer.weight * layer.scaler_row
            # salience[layer.weight.preserve_masks[0]==0, :] = 0
            # salience[:, layer.weight.preserve_masks[1]==0] = 0
            
            # Linear out_channels
            if prune_fn in [linear_mask_pruner.prune_out_channels]:
                local_norm = salience.abs().sum(1)
                group_imp.append(local_norm)

            # Linear in_channels
            elif prune_fn in [linear_mask_pruner.prune_in_channels]:
                local_norm = salience.abs().sum(0)
                group_imp.append(local_norm)
            else:
                raise NotImplementedError
       
        if len(group_imp)==0:
            return None
        min_imp_size = min([len(imp) for imp in group_imp])
        aligned_group_imp = []
        for imp in group_imp:
            if len(imp)>min_imp_size and len(imp)%min_imp_size==0:
                imp = imp.view(len(imp) // min_imp_size, min_imp_size).sum(0)
                aligned_group_imp.append(imp)
            elif len(imp)==min_imp_size:
                aligned_group_imp.append(imp)
        group_imp = torch.stack(aligned_group_imp, dim=0)
        group_imp = self._reduce(group_imp)
        group_imp = self._normalize(group, group_imp)
            
        return group_imp

    
class KnowledgeImportance(tp.importance.Importance):
    def __init__(self, group_reduction="sum", normalizer=None, model=None):
        self.group_reduction = group_reduction
        self.normalizer = normalizer
        self.model = model

    def _reduce(self, group_imp):
        if self.group_reduction == "sum":
            group_imp = group_imp.sum(dim=0)
        elif self.group_reduction == "mean":
            group_imp = group_imp.mean(dim=0)
        elif self.group_reduction == "max":
            group_imp = group_imp.max(dim=0)[0]
        elif self.group_reduction == "prod":
            group_imp = torch.prod(group_imp, dim=0)
        elif self.group_reduction=='first':
            group_imp = group_imp[0]
        elif self.group_reduction=='second':
            group_imp = group_imp[1]
        elif self.group_reduction is None:
            group_imp = group_imp
        else: 
            raise NotImplementedError
        return group_imp
    
    def _normalize(self, group, group_imp):
        if self.normalizer == "param":
            params = []
            for dep, idxs in group:
                layer = dep.target.module
                prune_fn = dep.handler
                # Linear out_channels
                if prune_fn in [linear_mask_pruner.prune_out_channels]:
                    if "qkv" in layer.weight.global_name: # TODO
                        params.append(torch.sum(layer.weight.preserve_masks[1])*3)
                    else:
                        params.append(torch.sum(layer.weight.preserve_masks[1]))

                # Linear in_channels
                elif prune_fn in [linear_mask_pruner.prune_in_channels]:
                    params.append(torch.sum(layer.weight.preserve_masks[0]))

                elif prune_fn in [t5_layer_norm_mask_pruner.prune_out_channels, t5_layer_norm_mask_pruner.prune_in_channels]:
                    params.append(1)
            return group_imp / sum(params)
        elif self.normalizer is not None:
            raise NotImplementedError
        else:
            return group_imp

    @torch.no_grad()
    def __call__(self, group, ch_groups=1, consecutive_groups=1):
        group_imp = []
        for dep, idxs in group:
            idxs.sort()
            layer = dep.target.module
            prune_fn = dep.handler

            # Linear out_channels
            if prune_fn in [linear_mask_pruner.prune_out_channels] and hasattr(layer, 'out_dim_entropy'):
                imp = layer.out_dim_entropy.clone()
                group_imp.append(imp)

            # Linear in_channels
            elif prune_fn in [linear_mask_pruner.prune_in_channels] and hasattr(layer, 'in_dim_entropy'):
                imp = layer.in_dim_entropy.clone()
                group_imp.append(imp)
            else:
                continue
       
        if len(group_imp)==0:
            return None
        min_imp_size = min([len(imp) for imp in group_imp])
        aligned_group_imp = []
        for imp in group_imp:
            if len(imp)>min_imp_size and len(imp)%min_imp_size==0:
                imp = imp.view(len(imp) // min_imp_size, min_imp_size).sum(0)
                aligned_group_imp.append(imp)
            elif len(imp)==min_imp_size:
                aligned_group_imp.append(imp)
        group_imp = torch.stack(aligned_group_imp, dim=0)
        group_imp = self._reduce(group_imp)
        group_imp = self._normalize(group, group_imp)
            
        return group_imp