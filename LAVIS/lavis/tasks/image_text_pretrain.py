"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 
 Modified by Zimeng Wu in 2025
 - Added support of progressive distillation for the UKMP method.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask

from lavis.datasets.data_utils import prepare_sample

from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask

from lavis.common.registry import registry
from lavis.datasets.data_utils import prepare_sample


@registry.register_task("image_text_pretrain")
class ImageTextPretrainTask(BaseTask):
    def __init__(self):
        super().__init__()

    def evaluation(self, model, data_loader, cuda_enabled=True):
        pass

    def get_data_derivative(self, model, data_loader, num_data=128, power=2, num_logits=1, vision_weight=0.0, cuda_enabled=False):
        gradients_dict = {}

        if power == 1:
            grad_method = torch.abs
        elif power == 2:
            grad_method = torch.square
        else:
            raise ValueError(f"power in `get_data_derivative` can only be 1 or 2, but got {power}")

        for name, param in model.named_parameters():
            gradients_dict[name] = 0

        idx = 0

        no_grad_list = set()

        for samples in data_loader:
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)

            loss_dict = model.forward_with_vision_auxloss(samples)
            loss = loss_dict["loss"] + loss_dict["vision_auxloss"] * vision_weight
            loss.backward()

            for name, param in model.named_parameters():

                if param.grad is not None:
                    gradients_dict[name] += grad_method(param.grad.cpu().data) / num_data
                else:
                    no_grad_list.add(name)
            
            model.zero_grad()

            idx += 1

            if idx >= num_data:
                break

        for k in no_grad_list:
            print(f"{k} has no grad")

        return gradients_dict

mse = nn.MSELoss()

@registry.register_task("image_text_pretrain_distill")
class ImageTextPretrainDistillTask(ImageTextPretrainTask):
    def __init__(self):
        super().__init__()
        self.full_model = None
        self.mse = nn.MSELoss()
        self.feature_norm = False
        self.loss_ratio = None
    
    def loss_fn(self, model, samples):
        return model(samples)['loss']
        
    def build_teacher_model(self, cfg):
        model_config = cfg.model_cfg

        model_cls = registry.get_model_class(model_config.arch)
        self.full_model = model_cls.from_config(model_config)
        return self.full_model
    
    def set_full_model(self, full_model):
        self.full_model = full_model
        
    def get_params(self, module):
        return sum(p.numel() for name, p in module.named_parameters() if "lora" not in name)
    
    def get_loss_ratio(self, part, idx):
        if self.loss_ratio is None:
            return 1.0
        return self.loss_ratio[part][idx]
        
    def set_loss_ratio(self, model):
        assert self.full_model
        loss_ratio = {
            'visual': [],
            'language_encoder': [],
            'language_decoder': [],
        }
        for i in range(39):
            tmp = self.get_params(model.visual_encoder.blocks[i]) * 1.0 / self.get_params(self.full_model.visual_encoder.blocks[i])
            tmp = 0 if tmp < 0.5 else tmp
            loss_ratio['visual'].append(tmp)
            
        for i in range(24):
            tmp = self.get_params(model.t5_model.encoder.block[i]) * 1.0 / self.get_params(self.full_model.t5_model.encoder.block[i])
            tmp = 0 if tmp < 0.5 else tmp
            loss_ratio['language_encoder'].append(tmp)
            
        for i in range(24):
            tmp = self.get_params(model.t5_model.decoder.block[i]) * 1.0 / self.get_params(self.full_model.t5_model.decoder.block[i])
            tmp = 0 if tmp < 0.5 else tmp
            loss_ratio['language_decoder'].append(tmp)
        self.loss_ratio = loss_ratio

    def train_step_low_visual(self, model, samples):
        pruned_output, full_output = model.module.base_model.model(samples), self.full_model(samples)
        vision_loss = 0.0
        for i in range(0, 23, 2):
            feat1 = model.module.base_model.model.visual_encoder.blocks[i].feature
            feat2 = self.full_model.visual_encoder.blocks[i].feature
            ratio = self.get_loss_ratio('visual', i)
            if self.feature_norm:
                vision_loss += self.mse(feat1/torch.norm(feat1, p=2), feat2/torch.norm(feat2, p=2))*ratio
            else:
                vision_loss += self.mse(feat1, feat2)*ratio
        return vision_loss*1e6

    def train_step_high_visual(self, model, samples):
        pruned_output, full_output = model.module.base_model.model(samples), self.full_model(samples)
        vision_loss = 0.0
        for i in range(0, 39, 2):
            feat1 = model.module.base_model.model.visual_encoder.blocks[i].feature
            feat2 = self.full_model.visual_encoder.blocks[i].feature
            ratio = self.get_loss_ratio('visual', i)
            if self.feature_norm:
                vision_loss += self.mse(feat1/torch.norm(feat1, p=2), feat2/torch.norm(feat2, p=2))*ratio
            else:
                vision_loss += self.mse(feat1, feat2)*ratio
        return vision_loss*1e6

    def train_step_low_encoder(self, model, samples):
        pruned_output, full_output = model.module.base_model.model(samples), self.full_model(samples)
        loss = 0.0
        vision_loss = 0.0
        for i in range(0, 39, 2):
            feat1 = model.module.base_model.model.visual_encoder.blocks[i].feature
            feat2 = self.full_model.visual_encoder.blocks[i].feature
            ratio = self.get_loss_ratio('visual', i)
            if self.feature_norm:
                vision_loss += self.mse(feat1/torch.norm(feat1, p=2), feat2/torch.norm(feat2, p=2))*ratio
            else:
                vision_loss += self.mse(feat1, feat2)*ratio
        loss += vision_loss*1e6
        encoder_loss = 0.0
        for i in range(1, 12, 2):
            feat1 = model.module.base_model.model.t5_model.encoder.block[i].feature
            feat2 = self.full_model.t5_model.encoder.block[i].feature
            ratio = self.get_loss_ratio('language_encoder', i)
            if self.feature_norm:
                encoder_loss += self.mse(feat1/torch.norm(feat1, p=2), feat2/torch.norm(feat2, p=2))*ratio
            else:
                encoder_loss += self.mse(feat1, feat2)*ratio
        loss += encoder_loss*1e9
        return loss

    def train_step_high_encoder(self, model, samples):
        pruned_output, full_output = model.module.base_model.model(samples), self.full_model(samples)
        loss = 0.0
        vision_loss = 0.0
        for i in range(0, 39, 2):
            feat1 = model.module.base_model.model.visual_encoder.blocks[i].feature
            feat2 = self.full_model.visual_encoder.blocks[i].feature
            ratio = self.get_loss_ratio('visual', i)
            if self.feature_norm:
                vision_loss += self.mse(feat1/torch.norm(feat1, p=2), feat2/torch.norm(feat2, p=2))*ratio
            else:
                vision_loss += self.mse(feat1, feat2)*ratio
        loss += vision_loss*1e6
        encoder_loss = 0.0
        for i in range(1, 24, 2):
            feat1 = model.module.base_model.model.t5_model.encoder.block[i].feature
            feat2 = self.full_model.t5_model.encoder.block[i].feature
            ratio = self.get_loss_ratio('language_encoder', i)
            if self.feature_norm:
                encoder_loss += self.mse(feat1/torch.norm(feat1, p=2), feat2/torch.norm(feat2, p=2))*ratio
            else:
                encoder_loss += self.mse(feat1, feat2)*ratio
        loss += encoder_loss*1e9
        return loss

    def train_step_low_decoder(self, model, samples):
        pruned_output, full_output = model.module.base_model.model(samples), self.full_model(samples)
        loss = 0.0
        vision_loss = 0.0
        for i in range(0, 39, 2):
            feat1 = model.module.base_model.model.visual_encoder.blocks[i].feature
            feat2 = self.full_model.visual_encoder.blocks[i].feature
            ratio = self.get_loss_ratio('visual', i)
            if self.feature_norm:
                vision_loss += self.mse(feat1/torch.norm(feat1, p=2), feat2/torch.norm(feat2, p=2))*ratio
            else:
                vision_loss += self.mse(feat1, feat2)*ratio
        loss += vision_loss*1e6
        encoder_loss = 0.0
        for i in range(1, 24, 2):
            feat1 = model.module.base_model.model.t5_model.encoder.block[i].feature
            feat2 = self.full_model.t5_model.encoder.block[i].feature
            ratio = self.get_loss_ratio('language_encoder', i)
            if self.feature_norm:
                encoder_loss += self.mse(feat1/torch.norm(feat1, p=2), feat2/torch.norm(feat2, p=2))*ratio
            else:
                encoder_loss += self.mse(feat1, feat2)*ratio
        loss += encoder_loss*1e9
        decoder_loss = 0.0
        for i in range(1, 12, 2):
            feat1 = model.module.base_model.model.t5_model.decoder.block[i].feature
            feat2 = self.full_model.t5_model.decoder.block[i].feature
            ratio = self.get_loss_ratio('language_decoder', i)
            if self.feature_norm:
                decoder_loss += self.mse(feat1/torch.norm(feat1, p=2), feat2/torch.norm(feat2, p=2))*ratio
            else:
                decoder_loss += self.mse(feat1, feat2)*ratio
        loss += decoder_loss*1e8
        return loss

    def train_step_high_decoder(self, model, samples):
        pruned_output, full_output = model.module.base_model.model(samples), self.full_model(samples)
        loss = 0.0
        vision_loss = 0.0
        for i in range(0, 39, 2):
            feat1 = model.module.base_model.model.visual_encoder.blocks[i].feature
            feat2 = self.full_model.visual_encoder.blocks[i].feature
            ratio = self.get_loss_ratio('visual', i)
            if self.feature_norm:
                vision_loss += self.mse(feat1/torch.norm(feat1, p=2), feat2/torch.norm(feat2, p=2))*ratio
            else:
                vision_loss += self.mse(feat1, feat2)*ratio
        loss += vision_loss*1e6
        encoder_loss = 0.0
        for i in range(1, 24, 2):
            feat1 = model.module.base_model.model.t5_model.encoder.block[i].feature
            feat2 = self.full_model.t5_model.encoder.block[i].feature
            ratio = self.get_loss_ratio('language_encoder', i)
            if self.feature_norm:
                encoder_loss += self.mse(feat1/torch.norm(feat1, p=2), feat2/torch.norm(feat2, p=2))*ratio
            else:
                encoder_loss += self.mse(feat1, feat2)*ratio
        loss += encoder_loss*1e9
        decoder_loss = 0.0
        for i in range(1, 24, 2):
            feat1 = model.module.base_model.model.t5_model.decoder.block[i].feature
            feat2 = self.full_model.t5_model.decoder.block[i].feature
            ratio = self.get_loss_ratio('language_decoder', i)
            if self.feature_norm:
                decoder_loss += self.mse(feat1/torch.norm(feat1, p=2), feat2/torch.norm(feat2, p=2))*ratio
            else:
                decoder_loss += self.mse(feat1, feat2)*ratio
        loss += decoder_loss*1e8
        return loss

    def train_step_top(self, model, samples):
        pruned_output, full_output = model.module.base_model.model(samples), self.full_model(samples)
        loss = pruned_output['loss']
        T = 1
        p, q = pruned_output['logits'], full_output['logits']
        p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        kl_loss = F.kl_div(F.log_softmax(p/T, dim=-1), F.softmax(q/T, dim=-1), reduction='sum')/p.shape[0]
        loss += kl_loss
        return loss
    
    def train_step(self, model, samples):
        return model(samples)['loss']