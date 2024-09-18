import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:4096' # do this before importing pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import EsmModel
import torch
import numpy as np
from lightning.pytorch import seed_everything
from typing import Tuple
import torch
import gc
from torch.optim.lr_scheduler import _LRScheduler
from transformers import EsmModel, PreTrainedModel
from configuration import MetaLATTEConfig
from urllib.parse import urljoin
seed_everything(42)
   
class GELU(nn.Module):
    """Implementation of the gelu activation function.

    For information: OpenAI GPT's gelu is slightly different
    (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1) # x: B, L, H, hidden # x1: B, L, H, hidden // 2
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin):
    # Assuming x has shape (B, L, H, HIDDEN_DIM)
    # cos and sin have shape (1, L, HIDDEN_DIM)
    cos = cos.unsqueeze(2)  # (1, L, 1, HIDDEN_DIM)
    sin = sin.unsqueeze(2)  # (1, L, 1, HIDDEN_DIM)
    return (x * cos) + (rotate_half(x) * sin)


class RotaryEmbedding(torch.nn.Module):
    """
    The rotary position embeddings from RoFormer_ (Su et. al).
    A crucial insight from the method is that the query and keys are
    transformed by rotation matrices which depend on the relative positions.
    Other implementations are available in the Rotary Transformer repo_ and in
    GPT-NeoX_, GPT-NeoX was an inspiration
    .. _RoFormer: https://arxiv.org/abs/2104.09864
    .. _repo: https://github.com/ZhuiyiTechnology/roformer
    .. _GPT-NeoX: https://github.com/EleutherAI/gpt-neox
    .. warning: Please note that this embedding is not registered on purpose, as it is transformative
        (it does not create the embedding dimension) and will likely be picked up (imported) on a ad-hoc basis
    """

    def __init__(self, dim: int, *_, **__):
        super().__init__()
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_tables(self, x, seq_dimension=1):
        seq_len = x.shape[seq_dimension]

        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if seq_len != self._seq_len_cached or self._cos_cached.device != x.device:
            self._seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dimension], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq) # L， 256
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device) # L, 512

            self._cos_cached = emb.cos()[None, :, :] # 1, L, 512
            self._sin_cached = emb.sin()[None, :, :] # 1, L, 512

        return self._cos_cached, self._sin_cached

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self._cos_cached, self._sin_cached = self._update_cos_sin_tables(k)

        return (
            apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached), # B, L, H, hidden
            apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached),
        )


def macro_f1(y_true, y_pred, thresholds):
    y_pred_binary = (y_pred >= thresholds).float()
    tp = (y_true * y_pred_binary).sum(dim=0)
    fp = ((1 - y_true) * y_pred_binary).sum(dim=0)
    fn = (y_true * (1 - y_pred_binary)).sum(dim=0)
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    macro_f1 = f1.mean()
    return macro_f1

def safeguard_softmax(logits, dim=-1):
    # remove max number to prevent exp() to be INF
    max_logits, _ = logits.max(dim=dim, keepdim=True)
    exp_logits = torch.exp(logits - max_logits)
    exp_sum = exp_logits.sum(dim=dim, keepdim=True)
    probs = exp_logits / (exp_sum + 1e-7)  # Adding a small epsilon to prevent division by zero
    return probs

class PositionalAttentionHead(nn.Module):
    def __init__(self, hidden_dim, n_heads):
        super(PositionalAttentionHead, self).__init__()
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // n_heads
        self.preattn_ln = nn.LayerNorm(self.head_dim)
        self.Q = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.K = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.V = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.rot_emb = RotaryEmbedding(self.head_dim)

    def forward(self, x, attention_mask):
        batch_size, seq_len, _ = x.size() # B, L, H
        x = x.view(batch_size, seq_len, self.n_heads, self.head_dim)
        x = self.preattn_ln(x)

        q = self.Q(x)
        k = self.K(x)
        v = self.V(x)

        q, k = self.rot_emb(q, k)
        gc.collect()
        torch.cuda.empty_cache()

        attn_scores = torch.einsum('bqhd,bkhd->bhqk', q, k) / math.sqrt(self.head_dim)
        #print(attention_mask.unsqueeze(1).shape)
        #print(attention_mask.unsqueeze(1).unsqueeze(1).shape)
        attn_scores = attn_scores.masked_fill(torch.logical_not(attention_mask.unsqueeze(1).unsqueeze(1)), float("-inf")) # B, H, L, L

        attn_probs = safeguard_softmax(attn_scores, dim=-1)

        x = torch.einsum('bhqk,bkhd->bqhd', attn_probs, v)
        x = x.reshape(batch_size, seq_len, self.hidden_dim)  # B, L, H
        gc.collect()
        torch.cuda.empty_cache()
        return x, attn_probs

class CosineAnnealingWithWarmup(_LRScheduler):
    # Implement based on Llama paper's description
    # https://arxiv.org/abs/2302.13971
    def __init__(self, optimizer, warmup_steps, total_steps, eta_ratio=0.1, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_ratio = eta_ratio  # The ratio of minimum to maximum learning rate
        super(CosineAnnealingWithWarmup, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [base_lr * self.last_epoch / self.warmup_steps for base_lr in self.base_lrs]

        progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
        decayed_lr = (1 - self.eta_ratio) * cosine_decay + self.eta_ratio

        return [decayed_lr * base_lr for base_lr in self.base_lrs]
    
class RobertaLMHead(nn.Module):
    """Head for masked language modeling."""
    def __init__(self, embed_dim, output_dim, weight):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.weight = weight
        self.gelu = GELU()
        self.bias = nn.Parameter(torch.zeros(output_dim))
    def forward(self, features):
        x = self.dense(features)
        x = self.gelu(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        return x

       
class MultitaskProteinModel(PreTrainedModel):
    config_class = MetaLATTEConfig  
    base_model_prefix = "metalatte"
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.esm_model = EsmModel.from_pretrained(self.config.esm_model_name)      
        # layer freezing for the original esm model
        # first freeze all
        for param in self.esm_model.parameters():
            param.requires_grad = False
        # unfreeze the required layers
        for i in range(config.num_layers_to_finetune):
            for param in self.esm_model.encoder.layer[-i-1].parameters():
                param.requires_grad = True
        self.lm_head = RobertaLMHead(embed_dim = 1280, output_dim=33, weight=self.esm_model.embeddings.word_embeddings.weight)
        # esm_dim should be 1280
        self.attn_head = PositionalAttentionHead(self.config.hidden_size, self.config.num_attention_heads)
        self.attn_ln = nn.LayerNorm(self.config.hidden_size)
        self.attn_skip = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.linear_layers = nn.ModuleList()
        # Add linear layers after the attention head
        for _ in range(self.config.num_linear_layers):
            self.linear_layers.append(nn.Linear(self.config.hidden_size, self.config.hidden_size))
        self.reduction_layers = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_dim),
            GELU(),
            nn.Linear(self.config.hidden_dim, self.config.num_labels)
        )
        self.clf_ln = nn.LayerNorm(self.config.hidden_size)
        self.classification_thresholds = nn.Parameter(torch.tensor([0.5]*self.config.num_labels))

        # Initialize weights and apply final processing
        self.post_init()
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = kwargs.pop("config", None)
        if config is None:
            config = MetaLATTEConfig.from_pretrained(pretrained_model_name_or_path)
        
        model = cls(config)        
        #state_dict = torch.load(f"{pretrained_model_name_or_path}/pytorch_model.bin", map_location=torch.device('cpu'))['state_dict']
        try:
            state_dict_url = urljoin(f"https://huggingface.co/{pretrained_model_name_or_path}/resolve/main/", "pytorch_model.bin")
            state_dict = torch.hub.load_state_dict_from_url(
                state_dict_url,
                map_location=torch.device('cpu')
            )['state_dict']
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            raise RuntimeError(f"Error loading state_dict from {pretrained_model_name_or_path}/pytorch_model.bin: {e}")

        return model
    
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.esm_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        embeddings = outputs.last_hidden_state
        attention_masks = attention_mask

        x_pool, x_attns = self.attn_head(embeddings, attention_masks)
        x_pool = self.attn_ln(x_pool + self.attn_skip(x_pool))  # Added skip connection for the attention layer

        for linear_layer in self.linear_layers:
            residue = x_pool
            x_pool = linear_layer(x_pool)  # 1280 -> 1280
            x_pool = F.silu(x_pool)
            x_pool = x_pool + residue  # Skip connection

        x_weighted = torch.einsum('bhlk,bld->bhld', x_attns, x_pool)  # (B, H, L, 1280)
        x_combined = x_weighted.mean(dim=1)  # Average over heads: (B, L, 1280)
        x_combined = self.clf_ln(x_combined)

        mlm_logits = self.lm_head(x_combined)
        attention_masks = attention_masks.unsqueeze(-1).float()  # (B, L, 1)
        attention_sum = attention_masks.sum(dim=1, keepdim=True)  # (B, 1, 1)
        x_combined_masked = (x_combined * attention_masks).sum(dim=1) / attention_sum.squeeze(1)  # (B, 1280)

        # Compute classification logits
        x_pred = self.reduction_layers(x_combined_masked)
        gc.collect()
        torch.cuda.empty_cache()
        return x_pred, x_attns, x_combined_masked, mlm_logits

    def predict(self, input_ids, attention_mask=None):
        x_pred, _, _, _ = self.forward(input_ids, attention_mask)
        classification_output = torch.sigmoid(x_pred)
        predictions = (classification_output >= self.classification_thresholds).float()

        for i, pred in enumerate(predictions):
            if pred.sum() == 0:
                weighted_probs = classification_output[i]
                max_class = torch.argmax(weighted_probs)
                predictions[i, max_class] = 1.0

        return classification_output, predictions

    