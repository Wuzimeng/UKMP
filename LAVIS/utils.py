import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput

def count_param(model):
    param_counts = {'out': {}, 'in': {}, 'total': {}}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            param_counts['out'][name] = module.weight.shape[0]
            param_counts['in'][name] = module.weight.shape[1]
            param_counts['total'][name] = module.weight.numel()
    return param_counts

def get_one_from_batch(batch, idx):
    res = {}
    for k, v in batch.items():
        res[k] = [v[idx]] if isinstance(v, list) else v[idx].unsqueeze(0)
    return res


def isattn(name):
    for s in ['.qkv', '.q', '.k', '.v', '.proj', '.o']:
        if s in name:
            return True
    return False

def forward_fn(model, example_input):
    return model(example_input)

def decoupled_visual_SA(self, x, rel_pos_bias=None, output_attn=False):
    B, N, C = x.shape
    q = self.q(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    k = self.k(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    
    q = q * self.scale
    attn = (q @ k.transpose(-2, -1))
    attn_before_softmax = None
    # attn_before_softmax = attn.clone()

    if self.relative_position_bias_table is not None:
        relative_position_bias = \
            self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1] + 1,
                self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

    if rel_pos_bias is not None:
        attn = attn + rel_pos_bias
    
    # attn_before_softmax = attn.clone()
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)

    x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    x = self.proj(x)
    x = self.proj_drop(x)

    return (x, attn_before_softmax) if output_attn else (x, None)

def decoupled_visual_FFN(self, x):
    x = self.fc1(x)
    x = self.act(x)
    # x = self.drop(x)
    # commit this for the orignal BERT implement 
    x = self.fc2(x)
    x = self.drop(x)
    return x

def decoupled_language_Attn(
    self,
    hidden_states,
    mask=None,
    key_value_states=None,
    position_bias=None,
    past_key_value=None,
    layer_head_mask=None,
    query_length=None,
    use_cache=False,
    output_attentions=False,
):
    batch_size, seq_length = hidden_states.shape[:2]

    real_seq_length = seq_length

    if past_key_value is not None:
        assert (
            len(past_key_value) == 2
        ), f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
        real_seq_length += (
            past_key_value[0].shape[2] if query_length is None else query_length
        )

    key_length = (
        real_seq_length if key_value_states is None else key_value_states.shape[1]
    )

    def shape(states):
        """projection"""
        return states.view(
            batch_size, -1, self.n_heads, self.key_value_proj_dim
        ).transpose(1, 2)

    def unshape(states):
        """reshape"""
        return (
            states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
        )

    def project(hidden_states, proj_layer, key_value_states, past_key_value):
        """projects hidden states correctly to key/query states"""
        if key_value_states is None:
            hidden_states = shape(proj_layer(hidden_states))
        elif past_key_value is None:
            hidden_states = shape(proj_layer(key_value_states))

        if past_key_value is not None:
            if key_value_states is None:
                hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
            else:
                hidden_states = past_key_value
        return hidden_states
    
    hidden_states_q = hidden_states
    hidden_states_k = hidden_states
    hidden_states_v = hidden_states
    key_value_states_k = key_value_states
    key_value_states_v = key_value_states
        
    query_states = shape(
        self.q(hidden_states_q)
    )  # (batch_size, n_heads, seq_length, dim_per_head)

    # get key/value states
    key_states = project(
        hidden_states_k,
        self.k,
        key_value_states_k,
        past_key_value[0] if past_key_value is not None else None,
    )
    value_states = project(
        hidden_states_v,
        self.v,
        key_value_states_v,
        past_key_value[1] if past_key_value is not None else None,
    )

    # compute scores
    scores = torch.matmul(
        query_states, key_states.transpose(3, 2)
    )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

    if position_bias is None:
        if not self.has_relative_attention_bias:
            position_bias = torch.zeros(
                (1, self.n_heads, real_seq_length, key_length),
                device=scores.device,
                dtype=scores.dtype,
            )
            if self.gradient_checkpointing and self.training:
                position_bias.requires_grad = True
        else:
            position_bias = self.compute_bias(
                real_seq_length, key_length, device=scores.device
            )

        # if key and values are already calculated
        # we want only the last query position bias
        if past_key_value is not None:
            position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

        if mask is not None:
            position_bias = (
                position_bias + mask
            )  # (batch_size, n_heads, seq_length, key_length)

    if self.pruned_heads:
        mask = torch.ones(position_bias.shape[1])
        mask[list(self.pruned_heads)] = 0
        position_bias_masked = position_bias[:, mask.bool()]
    else:
        position_bias_masked = position_bias

    if (not self.pruned_heads) and scores.shape[1] != position_bias_masked.shape[1]:
        position_bias_masked = position_bias_masked[:,0:1,:,:].repeat(1, scores.shape[1], 1, 1)

    scores += position_bias_masked
    attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
        scores
    )  # (batch_size, n_heads, seq_length, key_length)
    attn_weights = nn.functional.dropout(
        attn_weights, p=self.dropout, training=self.training
    )  # (batch_size, n_heads, seq_length, key_length)

    # Mask heads if we want to
    if layer_head_mask is not None:
        attn_weights = attn_weights * layer_head_mask

    attn_output = unshape(
        torch.matmul(attn_weights, value_states)
    )  # (batch_size, seq_length, dim)
    attn_output = self.o(attn_output)

    present_key_value_state = (
        (key_states, value_states) if (self.is_decoder and use_cache) else None
    )
    outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

    if output_attentions:
        outputs = outputs + (attn_weights,)
    return outputs

def decoupled_language_FFN(self, hidden_states):
    forwarded_states = self.layer_norm(hidden_states)
    forwarded_states = self.DenseReluDense(forwarded_states)
    hidden_states = hidden_states + self.dropout(forwarded_states)
    return hidden_states

def get_module_by_name(model, name):
    parts = name.split('.')
    module = model
    for part in parts:
        if hasattr(module, part):
            module = getattr(module, part)
        else:
            return None
    return module

def selected_t5_forward(
    self,
    input_ids = None,
    attention_mask = None,
    decoder_input_ids = None,
    decoder_attention_mask = None,
    head_mask = None,
    decoder_head_mask = None,
    cross_attn_head_mask = None,
    encoder_outputs = None,
    past_key_values = None,
    inputs_embeds = None,
    decoder_inputs_embeds = None,
    labels = None,
    use_cache = None,
    output_attentions = None,
    output_hidden_states = None,
    return_dict = None,
    reduction = "mean",
):
    r"""
    rewrite forward function from lavis.models.blip2_models.modeling_t5.T5ForConditionalGeneration
    ```"""
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
    # if head_mask is not None and decoder_head_mask is None:
    #     if self.config.num_layers == self.config.num_decoder_layers:
    #         warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
    #         decoder_head_mask = head_mask

    # Encode if needed (training, first prediction pass)
    if encoder_outputs is None:
        # Convert encoder inputs in embeddings if needed
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
        encoder_outputs = BaseModelOutput(
            last_hidden_state=encoder_outputs[0],
            hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
            attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
        )

    hidden_states = encoder_outputs[0]

    if self.model_parallel:
        torch.cuda.set_device(self.decoder.first_device)

    if (
        labels is not None
        and decoder_input_ids is None
        and decoder_inputs_embeds is None
    ):
        # get decoder inputs from shifting lm labels to the right
        decoder_input_ids = self._shift_right(labels)

    # Set device for model parallelism
    if self.model_parallel:
        torch.cuda.set_device(self.decoder.first_device)
        hidden_states = hidden_states.to(self.decoder.first_device)
        if decoder_input_ids is not None:
            decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.decoder.first_device)
        if decoder_attention_mask is not None:
            decoder_attention_mask = decoder_attention_mask.to(
                self.decoder.first_device
            )

    # Decode
    decoder_outputs = self.decoder(
        input_ids=decoder_input_ids,
        attention_mask=decoder_attention_mask,
        inputs_embeds=decoder_inputs_embeds,
        past_key_values=past_key_values,
        encoder_hidden_states=hidden_states,
        encoder_attention_mask=attention_mask,
        head_mask=decoder_head_mask,
        cross_attn_head_mask=cross_attn_head_mask,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    sequence_output = decoder_outputs[0]

    # Set device for model parallelism
    if self.model_parallel:
        torch.cuda.set_device(self.encoder.first_device)
        self.lm_head = self.lm_head.to(self.encoder.first_device)
        sequence_output = sequence_output.to(self.lm_head.weight.device)

    if self.config.tie_word_embeddings:
        # Rescale output before projecting on vocab
        # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
        sequence_output = sequence_output * (self.model_dim**-0.5)

    lm_logits = self.lm_head(sequence_output)
    
    sm_p = F.softmax(lm_logits, dim=2)
    max_values, _ = torch.max(sm_p, dim=2)

    thresh = max(0.4, torch.min(max_values).item())
    indices = (max_values <= thresh).nonzero(as_tuple=True)
    lm_logits = lm_logits[indices[0], indices[1], :]
    labels = labels[indices[0], indices[1]]
    
    loss = None
    if labels is not None:
        loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='mean')
        loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        if reduction == "none":
            loss = loss.view(lm_logits.size(0), -1).sum(1)

    if not return_dict:
        output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
        return ((loss,) + output) if loss is not None else output

    return Seq2SeqLMOutput(
        loss=loss,
        logits=lm_logits,
        past_key_values=decoder_outputs.past_key_values,
        decoder_hidden_states=decoder_outputs.hidden_states,
        decoder_attentions=decoder_outputs.attentions,
        cross_attentions=decoder_outputs.cross_attentions,
        encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        encoder_hidden_states=encoder_outputs.hidden_states,
        encoder_attentions=encoder_outputs.attentions,
    )