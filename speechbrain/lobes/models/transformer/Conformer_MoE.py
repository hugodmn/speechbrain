"""
Conformer implementation (offline-only + utterance-level Switch-MoE for FFN1/FFN2).

Key changes vs. baseline SpeechBrain Conformer:
- Adds ConformerEncoderLayer_MoE with:
  - utterance-level router (one routing decision per utterance)
  - two expert banks (FFN1 experts + FFN2 experts)
  - same expert id used for FFN1 and FFN2 inside the layer
- Allows selecting which encoder layers are MoE via moe_idx_layer (list of layer indices)
- Explicitly disables streaming everywhere (NotImplementedError), per your requirement.
- No padding logic is introduced by MoE; it works on (B,T,C) as-is.
"""

import warnings
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import speechbrain as sb
from speechbrain.nnet.activations import Swish
from speechbrain.nnet.attention import (
    MultiheadAttention,
    PositionalwiseFeedForward,
    RelPosMHAXL,
    RoPEMHA,
)
from speechbrain.nnet.hypermixing import HyperMixing
from speechbrain.nnet.normalization import LayerNorm
from speechbrain.utils.dynamic_chunk_training import DynChunkTrainConfig


# =========================
# Streaming context (kept for API compatibility, but streaming is disabled)
# =========================

@dataclass
class ConformerEncoderLayerStreamingContext:
    mha_left_context_size: int
    mha_left_context: Optional[torch.Tensor] = None
    dcconv_left_context: Optional[torch.Tensor] = None


@dataclass
class ConformerEncoderStreamingContext:
    dynchunktrain_config: DynChunkTrainConfig
    layers: List[ConformerEncoderLayerStreamingContext]


# =========================
# Convolution Module (unchanged)
# =========================

class ConvolutionModule(nn.Module):
    def __init__(
        self,
        input_size,
        kernel_size=31,
        bias=True,
        activation=Swish,
        dropout=0.0,
        causal=False,
        dilation=1,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.causal = causal
        self.dilation = dilation

        if self.causal:
            self.padding = (kernel_size - 1) * 2 ** (dilation - 1)
        else:
            self.padding = (kernel_size - 1) * 2 ** (dilation - 1) // 2

        self.layer_norm = nn.LayerNorm(input_size)
        self.bottleneck = nn.Sequential(
            nn.Conv1d(input_size, 2 * input_size, kernel_size=1, stride=1, bias=bias),
            nn.GLU(dim=1),
        )

        self.conv = nn.Conv1d(
            input_size,
            input_size,
            kernel_size=kernel_size,
            stride=1,
            padding=self.padding,
            dilation=dilation,
            groups=input_size,
            bias=bias,
        )

        self.after_conv = nn.Sequential(
            nn.LayerNorm(input_size),
            activation(),
            nn.Linear(input_size, input_size, bias=bias),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        dynchunktrain_config: Optional[DynChunkTrainConfig] = None,
    ):
        # NOTE: you said "no streaming"; dynchunktrain_config is for training-time masking.
        # Keeping this logic intact for compatibility with upstream.
        if dynchunktrain_config is not None:
            assert not self.causal, "Chunked convolution not supported with causal padding"
            assert self.dilation == 1, "Current DynChunkTrain logic does not support dilation != 1"

            chunk_size = dynchunktrain_config.chunk_size
            batch_size = x.shape[0]

            if x.shape[1] % chunk_size != 0:
                final_right_padding = chunk_size - (x.shape[1] % chunk_size)
            else:
                final_right_padding = 0

            out = self.layer_norm(x)
            out = out.transpose(1, 2)
            out = self.bottleneck(out)

            out = F.pad(out, (self.padding, final_right_padding), value=0)
            out = out.unfold(2, size=chunk_size + self.padding, step=chunk_size)
            out = F.pad(out, (0, self.padding), value=0)
            out = out.transpose(1, 2)
            out = out.flatten(start_dim=0, end_dim=1)

            out = F.conv1d(
                out,
                weight=self.conv.weight,
                bias=self.conv.bias,
                stride=self.conv.stride,
                padding=0,
                dilation=self.conv.dilation,
                groups=self.conv.groups,
            )

            out = out.transpose(1, 2)
            out = self.after_conv(out)
            out = torch.unflatten(out, dim=0, sizes=(batch_size, -1))
            out = torch.flatten(out, start_dim=1, end_dim=2)

            if final_right_padding > 0:
                out = out[:, :-final_right_padding, :]
        else:
            out = self.layer_norm(x)
            out = out.transpose(1, 2)
            out = self.bottleneck(out)
            out = self.conv(out)

            if self.causal:
                out = out[..., : -self.padding]

            out = out.transpose(1, 2)
            out = self.after_conv(out)

        if mask is not None:
            out.masked_fill_(mask, 0.0)

        return out




class FFNExpert(nn.Module):
    """ FFN expert """

    def __init__(self, d_model: int, d_ffn: int, dropout: float, activation=Swish):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            PositionalwiseFeedForward(
                d_ffn=d_ffn,
                input_size=d_model,
                dropout=dropout,
                activation=activation,
            ),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,C)
        return self.net(x)




def moe_aux_loss_topk(probs: torch.Tensor, top_idx: torch.Tensor, top_w: torch.Tensor) -> torch.Tensor:
    """
    probs:   (B, N) softmax over experts
    top_idx: (B, K) selected expert ids
    top_w:   (B, K) normalized weights (sum over K = 1)
    """
    B, N = probs.shape 
    K = top_idx.shape[1]

    if top_idx.dtype != torch.long:
        top_idx = top_idx.long()

    f = torch.zeros(N, device=probs.device)
    f.scatter_add_(0, top_idx.flatten(), top_w.flatten())   # (N,)
    f = f / B  # empirical freq of expert selection

    p = probs.mean(dim=0)  # (N,)
    aux_loss = N * torch.sum(f * p)
    return aux_loss


def moe_apply_topk_utterance(
    x: torch.Tensor,
    experts: nn.ModuleList,
    top_idx: torch.Tensor,   # (B, K)
    top_w: torch.Tensor,     # (B, K) sum over K = 1
) -> torch.Tensor:
    """
    Utterance-level top-k mixture.
    - Routing decision is per utterance, not per token.
    - Experts operate on (B,T,C) slices.

    x:       (B,T,C)
    top_idx: (B,K) expert ids per utterance
    top_w:   (B,K) weights per utterance (normalized)
    returns: (B,T,C)
    """
    B, T, C = x.shape
    N = len(experts)
    K = top_idx.shape[1]

    if top_idx.dtype != torch.long:
        top_idx = top_idx.long()

    y = x.new_zeros(B, T, C)

    # For each expert, accumulate contributions from utterances that picked it
    for e in range(N):
        # mask over utterances x topk slots
        m = (top_idx == e)            # (B,K) bool
        if not m.any():
            continue

        b_idx, k_idx = torch.where(m) # indices in (B,K)

        # gather utterance chunks
        x_e = x[b_idx]                # (Be,T,C)

        # gather weights for those utterances for this expert occurrence
        w_e = top_w[b_idx, k_idx].view(-1, 1, 1)  # (Be,1,1)

        # apply expert on utterances, weighted sum back
        y_e = experts[e](x_e) * w_e   # (Be,T,C)
        y.index_add_(0, b_idx, y_e)

    return y


class Router(nn.Module):
    pass


# class GlobalRouter_attnpooling(Router):
#     """
#     Utterance-level router:
#     - attention pooling over time -> mean/std summary -> expert logits
#     Returns:
#       top_probs: (B,K)
#       idx:       (B,K)
#       probs:     (B,N)
#     """

#     def __init__(self, input_dim: int, experts_nb: int, top_k: int, attn_dim: int = 128):
#         super().__init__()
#         self.attn_proj = nn.Linear(input_dim, attn_dim)
#         self.attn_weights = nn.Parameter(torch.randn(attn_dim))
#         self.expert_proj = nn.Linear(2 * input_dim, experts_nb)
#         self.top_k = min(top_k, experts_nb)
#         self.experts_nb = experts_nb

#     def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None):
#         # x: (B,T,C)
#         scores = torch.tanh(self.attn_proj(x))            # (B,T,H)
#         scores = torch.matmul(scores, self.attn_weights)  # (B,T)

#         if padding_mask is not None : 
#             if padding_mask.dtype != torch.bool:
#                 padding_mask = padding_mask.to(torch.bool)

#             valid = (~padding_mask).sum(dim=1)                  # (B,)
#             if (valid == 0).any():
#                 raise RuntimeError("Router received an all-padding utterance")

#             scores = scores.masked_fill(padding_mask, float('-inf'))


#         alphas = F.softmax(scores, dim=-1).unsqueeze(-1)     # (B,T,1)


#         if padding_mask is not None:
#             alphas = alphas * (~padding_mask).unsqueeze(-1)     # (B,T,1)
#             wsum = alphas.sum(dim=1, keepdim=True)              # (B,1,1)
#             if (wsum < 1e-6).any():
#                 raise RuntimeError("Router attention collapsed (sum ~ 0)")
#             alphas = alphas / wsum
            
#         # weighted mean/std (prevent from padding bias)
#         eps = 1e-8
#         weighted_x = alphas * x                              # (B,T,C)
#         mean = weighted_x.sum(dim=1)                           # (B,C)
#         # weighted_x.mean(dim=1)                        # (B,C)
        
#         var = (alphas * (x - mean.unsqueeze(1))**2).sum(dim=1)  # (B,C)
#         std = torch.sqrt(var + eps)                          # (B,C)

#         gate_input = torch.cat([mean, std], dim=-1)          # (B,2C)
#         logits = self.expert_proj(gate_input)                # (B,N)
#         probs = F.softmax(logits, dim=-1)                    # (B,N)

#         top_probs, idx = torch.topk(probs, k=self.top_k, dim=-1)  # (B,K), (B,K)
#         top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)

#         return top_probs, idx, probs


class GlobalRouter_statpooling(Router):
    def __init__(self, input_dim: int, experts_nb: int, top_k: int):
        super().__init__()
        self.expert_proj = nn.Linear(2 * input_dim, experts_nb)
        self.top_k = min(top_k, experts_nb)

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None):
        # x: (B, T, C)

        if padding_mask is not None:
            if padding_mask.dtype != torch.bool:
                padding_mask = padding_mask.to(torch.bool)
            valid = ~padding_mask                                          # (B, T)
            lengths = valid.sum(dim=1, keepdim=True).float().clamp(min=1) # (B, 1)
            x_valid = x * valid.unsqueeze(-1)                             # zero out padding
            mean = x_valid.sum(dim=1) / lengths                           # (B, C)
            var = ((x - mean.unsqueeze(1)) ** 2 * valid.unsqueeze(-1)).sum(dim=1) / lengths
            std = torch.sqrt(var + 1e-8)                                  # (B, C)
        else:
            mean = x.mean(dim=1)                                          # (B, C)
            std = x.std(dim=1)                                            # (B, C)

        gate_input = torch.cat([mean, std], dim=-1)                       # (B, 2C)
        logits = self.expert_proj(gate_input)                             # (B, N)
        probs = F.softmax(logits, dim=-1)                                 # (B, N)
        top_probs, idx = torch.topk(probs, k=self.top_k, dim=-1)         # (B, K)
        top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)

        return top_probs, idx, probs




# =========================
# Baseline Conformer Encoder Layer (unchanged forward; streaming disabled)
# =========================

class ConformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        d_ffn,
        nhead,
        kernel_size=31,
        kdim=None,
        vdim=None,
        activation=Swish,
        bias=True,
        dropout=0.0,
        causal=False,
        attention_type="RelPosMHAXL",
    ):
        super().__init__()

        if attention_type == "regularMHA":
            self.mha_layer = MultiheadAttention(nhead=nhead, d_model=d_model, dropout=dropout, kdim=kdim, vdim=vdim)
        elif attention_type == "RelPosMHAXL":
            self.mha_layer = RelPosMHAXL(num_heads=nhead, embed_dim=d_model, dropout=dropout, mask_pos_future=causal)
        elif attention_type == "hypermixing":
            self.mha_layer = HyperMixing(
                input_output_dim=d_model,
                hypernet_size=d_ffn,
                tied=False,
                num_heads=nhead,
                fix_tm_hidden_size=False,
            )
        elif attention_type == "RoPEMHA":
            self.mha_layer = RoPEMHA(num_heads=nhead, embed_dim=d_model, dropout=dropout)
        else:
            raise ValueError(f"Unknown attention_type={attention_type}")

        self.convolution_module = ConvolutionModule(d_model, kernel_size, bias, activation, dropout, causal=causal)

        self.ffn_module1 = nn.Sequential(
            nn.LayerNorm(d_model),
            PositionalwiseFeedForward(d_ffn=d_ffn, input_size=d_model, dropout=dropout, activation=activation),
            nn.Dropout(dropout),
        )

        self.ffn_module2 = nn.Sequential(
            nn.LayerNorm(d_model),
            PositionalwiseFeedForward(d_ffn=d_ffn, input_size=d_model, dropout=dropout, activation=activation),
            nn.Dropout(dropout),
        )

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

    def forward(
        self,
        x,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos_embs: Optional[torch.Tensor] = None,
        dynchunktrain_config: Optional[DynChunkTrainConfig] = None,
    ):
        conv_mask = src_key_padding_mask.unsqueeze(-1) if src_key_padding_mask is not None else None

        x = x + 0.5 * self.ffn_module1(x)

        skip = x
        x = self.norm1(x)
        x, self_attn = self.mha_layer(
            x, x, x,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            pos_embs=pos_embs,
        )
        x = x + skip

        x = x + self.convolution_module(x, conv_mask, dynchunktrain_config=dynchunktrain_config)

        x = self.norm2(x + 0.5 * self.ffn_module2(x))
        return x, self_attn

    def forward_streaming(self, *args, **kwargs):
        raise NotImplementedError("Streaming is disabled (offline-only encoder).")

    def make_streaming_context(self, *args, **kwargs):
        raise NotImplementedError("Streaming is disabled (offline-only encoder).")


# =========================
# MoE Conformer Encoder Layer (utterance-level routing; streaming disabled)
# =========================

class ConformerEncoderLayer_MoE(nn.Module):
    def __init__(
        self,
        d_model,
        d_ffn,
        nhead,
        n_experts: int = 3,
        top_k: int = 1,
        kernel_size=31,
        kdim=None,
        vdim=None,
        activation=Swish,
        bias=True,
        dropout=0.0,
        causal=False,
        attention_type="RelPosMHAXL",
    ):
        super().__init__()

        if attention_type == "regularMHA":
            self.mha_layer = MultiheadAttention(nhead=nhead, d_model=d_model, dropout=dropout, kdim=kdim, vdim=vdim)
        elif attention_type == "RelPosMHAXL":
            self.mha_layer = RelPosMHAXL(num_heads=nhead, embed_dim=d_model, dropout=dropout, mask_pos_future=causal)
        elif attention_type == "hypermixing":
            self.mha_layer = HyperMixing(
                input_output_dim=d_model,
                hypernet_size=d_ffn,
                tied=False,
                num_heads=nhead,
                fix_tm_hidden_size=False,
            )
        elif attention_type == "RoPEMHA":
            self.mha_layer = RoPEMHA(num_heads=nhead, embed_dim=d_model, dropout=dropout)
        else:
            raise ValueError(f"Unknown attention_type={attention_type}")

        self.convolution_module = ConvolutionModule(d_model, kernel_size, bias, activation, dropout, causal=causal)

        # utterance-level router
        self.router = GlobalRouter_statpooling(
            input_dim=d_model,
            experts_nb=n_experts,
            top_k=top_k,
        )

        # two FFN expert banks (FFN1 and FFN2)
        self.ffn1_experts = nn.ModuleList(
            [FFNExpert(d_model, d_ffn, dropout=dropout, activation=activation) for _ in range(n_experts)]
        )
        self.ffn2_experts = nn.ModuleList(
            [FFNExpert(d_model, d_ffn, dropout=dropout, activation=activation) for _ in range(n_experts)]
        )

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

    def forward(
        self,
        x,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos_embs: Optional[torch.Tensor] = None,
        dynchunktrain_config: Optional[DynChunkTrainConfig] = None,
    ):
        conv_mask = src_key_padding_mask.unsqueeze(-1) if src_key_padding_mask is not None else None

        # route once per utterance
        top_probs, top_idx, probs = self.router(x, padding_mask=src_key_padding_mask)    
        aux_loss = moe_aux_loss_topk(probs, top_idx, top_probs)



        # FFN1 (selected expert)
        x = x + 0.5 * moe_apply_topk_utterance(x, self.ffn1_experts, top_idx, top_probs)

        # MHA
        skip = x
        x = self.norm1(x)
        x, self_attn = self.mha_layer(
            x, x, x,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            pos_embs=pos_embs,
        )
        x = x + skip

        # Conv
        x = x + self.convolution_module(x, conv_mask, dynchunktrain_config=dynchunktrain_config)

        # FFN2 (same selected expert id)
        x = self.norm2(x + 0.5 * moe_apply_topk_utterance(x, self.ffn2_experts, top_idx, top_probs))

        return x, self_attn, aux_loss

    def forward_streaming(self, *args, **kwargs):
        raise NotImplementedError("Streaming is disabled (offline-only MoE layer).")

    def make_streaming_context(self, *args, **kwargs):
        raise NotImplementedError("Streaming is disabled (offline-only MoE layer).")


# =========================
# Conformer Encoder (supports MoE layer selection; streaming disabled)
# =========================

class ConformerEncoder_MoE(nn.Module):
    def __init__(
        self,
        num_layers,
        d_model,
        d_ffn,
        nhead,
        n_experts: int = 3,
        top_k: int = 1,
        moe_idx_layer: Optional[List[int]] = None,
        kernel_size=31,
        kdim=None,
        vdim=None,
        activation=Swish,
        bias=True,
        dropout=0.0,
        causal=False,
        attention_type="RelPosMHAXL",
        output_hidden_states=False,
        layerdrop_prob=0.0,
    ):
        super().__init__()

        moe_idx_layer = set(moe_idx_layer or [])

        self.layers = nn.ModuleList()
        for layer_idx in range(num_layers):
            if layer_idx in moe_idx_layer:
                self.layers.append(
                    ConformerEncoderLayer_MoE(
                        d_model=d_model,
                        d_ffn=d_ffn,
                        nhead=nhead,
                        n_experts=n_experts,
                        top_k=top_k,
                        kernel_size=kernel_size,
                        kdim=kdim,
                        vdim=vdim,
                        activation=activation,
                        bias=bias,
                        dropout=dropout,
                        causal=causal,
                        attention_type=attention_type,
                    )
                )
            else:
                self.layers.append(
                    ConformerEncoderLayer(
                        d_model=d_model,
                        d_ffn=d_ffn,
                        nhead=nhead,
                        kernel_size=kernel_size,
                        kdim=kdim,
                        vdim=vdim,
                        activation=activation,
                        bias=bias,
                        dropout=dropout,
                        causal=causal,
                        attention_type=attention_type,
                    )
                )

        self.norm = LayerNorm(d_model, eps=1e-6)
        self.layerdrop_prob = layerdrop_prob
        self.attention_type = attention_type
        self.output_hidden_states = output_hidden_states

    def forward(
        self,
        src,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        pos_embs: Optional[torch.Tensor] = None,
        dynchunktrain_config: Optional[DynChunkTrainConfig] = None,
    ):
        if self.attention_type == "RelPosMHAXL" and pos_embs is None:
            raise ValueError(
                f"attention_type={self.attention_type} requires pos_embs (shape: (1, 2*T-1, d_model))."
            )

        output = src
        attention_lst = []

        if self.output_hidden_states:
            hidden_state_lst = [output]

        if self.layerdrop_prob > 0.0:
            keep_probs = torch.rand(len(self.layers), device=output.device)
        else:
            keep_probs = None

        aux_loss_list = []
        for i, enc_layer in enumerate(self.layers):
            if (not self.training) or (self.layerdrop_prob == 0.0) or (keep_probs[i] > self.layerdrop_prob):
                if isinstance(enc_layer, ConformerEncoderLayer_MoE):
                    output, attn, aux_loss = enc_layer(
                        output,
                        src_mask=src_mask,
                        src_key_padding_mask=src_key_padding_mask,
                        pos_embs=pos_embs,
                        dynchunktrain_config=dynchunktrain_config,
                    )
                    aux_loss_list.append(aux_loss)
                    # aux_loss can be used for logging if desired
                else:
                    output, attn = enc_layer(
                        output,
                        src_mask=src_mask,
                        src_key_padding_mask=src_key_padding_mask,
                        pos_embs=pos_embs,
                        dynchunktrain_config=dynchunktrain_config,
                    )
                attention_lst.append(attn)
                if self.output_hidden_states:
                    hidden_state_lst.append(output)

        output = self.norm(output)

        if self.output_hidden_states:
            return output, attention_lst, hidden_state_lst, aux_loss_list
        
        return output, attention_lst, aux_loss_list

    def forward_streaming(self, *args, **kwargs):
        raise NotImplementedError(
            "Streaming is not supported for this ConformerEncoder (offline / full-utterance only, MoE enabled)."
        )

    def make_streaming_context(self, *args, **kwargs):
        raise NotImplementedError("Streaming context is not available: this encoder is offline-only.")


# =========================
# Decoder (unchanged from your pasted version; kept here for completeness)
# =========================

class ConformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        d_ffn,
        nhead,
        kernel_size,
        kdim=None,
        vdim=None,
        activation=Swish,
        bias=True,
        dropout=0.0,
        causal=True,
        attention_type="RelPosMHAXL",
    ):
        super().__init__()

        if not causal:
            warnings.warn(
                "Decoder is not causal; in most applications it should be causal."
            )

        if attention_type == "regularMHA":
            self.mha_layer = MultiheadAttention(nhead=nhead, d_model=d_model, dropout=dropout, kdim=kdim, vdim=vdim)
        elif attention_type == "RelPosMHAXL":
            self.mha_layer = RelPosMHAXL(num_heads=nhead, embed_dim=d_model, dropout=dropout, mask_pos_future=causal)
        else:
            raise ValueError(f"Unknown attention_type={attention_type}")

        self.convolution_module = ConvolutionModule(d_model, kernel_size, bias, activation, dropout, causal=causal)

        self.ffn_module1 = nn.Sequential(
            nn.LayerNorm(d_model),
            PositionalwiseFeedForward(d_ffn=d_ffn, input_size=d_model, dropout=dropout, activation=activation),
            nn.Dropout(dropout),
        )

        self.ffn_module2 = nn.Sequential(
            nn.LayerNorm(d_model),
            PositionalwiseFeedForward(d_ffn=d_ffn, input_size=d_model, dropout=dropout, activation=activation),
            nn.Dropout(dropout),
        )

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        pos_embs_tgt=None,
        pos_embs_src=None,
    ):
        tgt = tgt + 0.5 * self.ffn_module1(tgt)

        skip = tgt
        x = self.norm1(tgt)
        x, self_attn = self.mha_layer(
            x,
            memory,
            memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            pos_embs=pos_embs_src,
        )
        x = x + skip

        x = x + self.convolution_module(x)
        x = self.norm2(x + 0.5 * self.ffn_module2(x))
        return x, self_attn, self_attn


class ConformerDecoder(nn.Module):
    def __init__(
        self,
        num_layers,
        nhead,
        d_ffn,
        d_model,
        kdim=None,
        vdim=None,
        dropout=0.0,
        activation=Swish,
        kernel_size=3,
        bias=True,
        causal=True,
        attention_type="RelPosMHAXL",
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                ConformerDecoderLayer(
                    d_model=d_model,
                    d_ffn=d_ffn,
                    nhead=nhead,
                    kdim=kdim,
                    vdim=vdim,
                    dropout=dropout,
                    activation=activation,
                    kernel_size=kernel_size,
                    bias=bias,
                    causal=causal,
                    attention_type=attention_type,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = sb.nnet.normalization.LayerNorm(d_model, eps=1e-6)

    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
        pos_embs_tgt=None,
        pos_embs_src=None,
    ):
        output = tgt
        self_attns, multihead_attns = [], []
        for dec_layer in self.layers:
            output, self_attn, multihead_attn = dec_layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos_embs_tgt=pos_embs_tgt,
                pos_embs_src=pos_embs_src,
            )
            self_attns.append(self_attn)
            multihead_attns.append(multihead_attn)
        output = self.norm(output)
        return output, self_attns, multihead_attns
