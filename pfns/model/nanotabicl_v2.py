# Adapted from https://github.com/soda-inria/nanotabicl
#
# BSD 3-Clause License
#
# Copyright (c) 2025, Soda team @ Inria
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""NanoTabICLv2 architecture adapted for the PFN training framework.

Modified from the original to accept pre-encoded embeddings from PFN's
encoder pipeline (x_encoder / y_encoder) instead of raw features.
Internal preprocessing (x standardization, cyclic-shift grouping, raw
feature embedding) has been removed so that the only difference with the
standard PFN model is the attention architecture itself.
"""

from __future__ import annotations

import math
import typing

import torch
import torch.nn as nn


class NanoTabICLv2(nn.Module):
    """TabICL v2 attention blocks operating on PFN-encoded embeddings.

    Expects pre-encoded inputs:
      - emb_x: ``(B, S, num_groups, emsize)`` from PFN's x_encoder
      - emb_y: ``(B, S, emsize)`` from PFN's y_encoder

    The architecture has three stages:
      1. **TF_col** — induced self-attention across sequence positions
         within each feature group (column).
      2. **TF_row** — self-attention across feature groups (+ CLS tokens)
         at each sequence position.  CLS tokens are extracted and flattened
         into ``icl_dim = embed_dim * n_cls_cols``.
      3. **TF_icl** — self-attention across sequence positions in the merged
         representation.  The final block uses cross-attention (test queries,
         train keys/values).

    Returns embeddings of shape ``(B, test_len, icl_dim)`` — no output head.
    The decoder is applied externally by :class:`NanoTabICLWrapper`.
    """

    def __init__(
        self,
        emsize: int,
        embed_dim: int = 128,
        col_num_blocks: int = 3,
        row_num_blocks: int = 3,
        icl_num_blocks: int = 12,
        col_nhead: int = 8,
        row_nhead: int = 8,
        icl_nhead: int = 8,
        n_cls_cols: int = 4,
        n_cls_rows: int = 128,
    ):
        super().__init__()

        icl_dim = embed_dim * n_cls_cols

        # Project PFN embeddings (emsize) to TabICL's working dimension
        self.x_proj = nn.Linear(emsize, embed_dim)
        self.y_proj_in = nn.Linear(emsize, embed_dim)
        self.y_proj_icl = nn.Linear(emsize, icl_dim)

        # Stage 1: column attention (across positions within each feature)
        self.col_blocks = nn.ModuleList([
            InducedTransformerBlock(
                embed_dim=embed_dim, num_heads=col_nhead,
                n_inducing=n_cls_rows, ssmax=True,
            )
            for _ in range(col_num_blocks)
        ])

        # Stage 2: row attention (across features at each position)
        self.row_blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim, num_heads=row_nhead, use_rope=True,
            )
            for _ in range(row_num_blocks)
        ])

        # Stage 3: ICL attention (across positions in merged representation)
        self.icl_blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=icl_dim, num_heads=icl_nhead, ssmax=True,
            )
            for _ in range(icl_num_blocks)
        ])

        self.row_cls_tokens = nn.Parameter(
            0.02 * torch.randn(1, 1, n_cls_cols, embed_dim)
        )
        self.row_ln = nn.LayerNorm(embed_dim)
        self.out_ln = nn.LayerNorm(icl_dim)

        self.n_cls_cols = n_cls_cols
        self.icl_dim = icl_dim

    def forward(
        self,
        emb_x: torch.Tensor,
        emb_y: torch.Tensor,
        n_train: int,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        emb_x : (B, S, num_groups, emsize)
            PFN-encoded x features (one embedding per feature group).
        emb_y : (B, S, emsize)
            PFN-encoded y values (NaN-handled, test positions encoded with
            NaN indicator by the upstream y_encoder).
        n_train : int
            Number of training (context) positions.

        Returns
        -------
        (B, test_len, icl_dim)
            Embeddings for test positions, ready for the decoder.
        """
        n_batch, n_rows, n_cols, _ = emb_x.shape

        # --- Project to TabICL embed_dim ---
        emb = self.x_proj(emb_x)  # (B, S, n_cols, embed_dim)
        # Add y information at training positions only
        emb[:, :n_train] += self.y_proj_in(emb_y[:, :n_train]).unsqueeze(2)

        # --- TF_col: induced attention within each column ---
        for block in self.col_blocks:
            emb = block.col_attn(emb, kv_max_idx=n_train)

        # --- TF_row: CLS tokens + row attention + extract CLS ---
        emb = torch.cat(
            [self.row_cls_tokens.expand(n_batch, n_rows, -1, -1), emb], dim=2,
        )
        for block in self.row_blocks[:-1]:
            emb = block.row_attn(emb)
        # Last block: only need CLS token outputs
        emb = self.row_blocks[-1].row_attn(emb, q_max_idx=self.n_cls_cols)
        emb = self.row_ln(emb).flatten(-2, -1)  # (B, S, icl_dim)

        # --- TF_icl: add y embedding again + sequence attention ---
        emb[:, :n_train] += self.y_proj_icl(emb_y[:, :n_train])
        for block in self.icl_blocks[:-1]:
            emb = block(emb, kv_max_idx=n_train)
        # Final block: cross-attention — test queries, train keys/values
        emb = self.icl_blocks[-1](emb[:, n_train:], emb[:, :n_train])

        return self.out_ln(emb)  # (B, test_len, icl_dim)


# ---------------------------------------------------------------------------
# Helper modules (unchanged from original NanoTabICLv2)
# ---------------------------------------------------------------------------


def get_mlp(n_in: int, n_hidden: int, n_out: int):
    return nn.Sequential(nn.Linear(n_in, n_hidden), nn.GELU(), nn.Linear(n_hidden, n_out))


class TableAttnBase(nn.Module):
    """Base with helpers to apply attention on 2-D tables (rows / columns)."""

    def row_attn(self, q, kv=None, **kwargs):
        n_batch, n_rows, n_cols, embed_dim = q.shape
        q = q.flatten(0, 1)
        kv = None if kv is None else kv.flatten(0, 1)
        x = self(q, kv, **kwargs)
        return x.reshape(n_batch, n_rows, -1, embed_dim)

    def col_attn(self, q, kv=None, **kwargs):
        return self.row_attn(
            q.transpose(1, 2),
            None if kv is None else kv.transpose(1, 2),
            **kwargs,
        ).transpose(1, 2)


class InducedTransformerBlock(TableAttnBase):
    def __init__(self, embed_dim: int, num_heads: int, n_inducing: int, ssmax: bool = False):
        super().__init__()
        self.tfm1 = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, ssmax=ssmax)
        self.tfm2 = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads)
        self.inducing_vectors = nn.Parameter(0.02 * torch.randn(1, n_inducing, embed_dim))

    def forward(self, q, kv=None, q_max_idx: typing.Optional[int] = None, kv_max_idx: typing.Optional[int] = None):
        kv = self.tfm1(self.inducing_vectors.expand(q.shape[0], -1, -1), q if kv is None else kv, kv_max_idx=kv_max_idx)
        return self.tfm2(q, kv, q_max_idx=q_max_idx)


class TransformerBlock(nn.MultiheadAttention, TableAttnBase):
    def __init__(self, embed_dim: int, num_heads: int, use_rope: bool = False, ssmax: bool = False):
        super().__init__(embed_dim=embed_dim, num_heads=num_heads)
        self.use_rope = use_rope
        self.ssmax_layer = QASSMax(num_heads=num_heads, head_dim=embed_dim // num_heads) if ssmax else None
        self.mlp = get_mlp(embed_dim, embed_dim * 2, embed_dim)
        self.ln_attn = nn.LayerNorm(embed_dim)
        self.ln_mlp = nn.LayerNorm(embed_dim)

    def forward(self, q, kv=None, q_max_idx: typing.Optional[int] = None, kv_max_idx: typing.Optional[int] = None):
        x, q = q, self.ln_attn(q)
        kv = q if kv is None else self.ln_attn(kv)
        if kv_max_idx is not None:
            kv = kv[..., :kv_max_idx, :]
        if q_max_idx is not None:
            x, q = x[..., :q_max_idx, :], q[..., :q_max_idx, :]
        x = x + self.attn(q, kv)
        del q, kv
        return x + self.mlp(self.ln_mlp(x))

    def attn(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        q, k, v = nn.functional._in_projection_packed(q, k, k, self.in_proj_weight, self.in_proj_bias)
        q, k, v = (t.unflatten(-1, (self.num_heads, self.head_dim)).transpose(-3, -2) for t in [q, k, v])
        if self.ssmax_layer is not None:
            q = self.ssmax_layer(q=q, n=k.size(-2))
        if self.use_rope:
            q, k = apply_rope(q), apply_rope(k)
        attn_output = nn.functional.scaled_dot_product_attention(*[t.flatten(0, 1) for t in (q, k, v)]).view(q.shape)
        del q, k, v
        return self.out_proj(attn_output.transpose(-3, -2).flatten(-2, -1))


@torch.autocast("cuda", enabled=False)
def apply_rope(x: torch.Tensor, theta: float = 100_000.0) -> torch.Tensor:
    batch_size, num_heads, seq_len, head_dim = x.shape
    pos = torch.arange(seq_len, device=x.device).float()
    inv_freq = theta ** (torch.linspace(0.0, -1.0, head_dim // 2 + 1, device=x.device)[:-1])
    angles = torch.outer(pos, inv_freq)
    sin, cos = angles.sin(), angles.cos()
    mat = torch.stack([cos, -sin, sin, cos], dim=-1).unflatten(-1, (2, 2))
    return (mat[None, None, :seq_len] @ x.unflatten(-1, (-1, 2, 1))).flatten(-3, -1)


class QASSMax(nn.Module):
    """Query-aware scalable softmax for better context length scaling."""

    def __init__(self, num_heads: int, head_dim: int, n_hidden: int = 64):
        super().__init__()
        self.base_mlp = get_mlp(1, n_hidden, num_heads * head_dim)
        self.query_mlp = get_mlp(head_dim, n_hidden, head_dim)
        nn.init.zeros_(self.query_mlp[-1].weight)
        nn.init.zeros_(self.query_mlp[-1].bias)

    def forward(self, q: torch.Tensor, n: int) -> torch.Tensor:
        batch_size, num_heads, seq_len, head_dim = q.shape
        logn = q.new_tensor(math.log(max(1, n))).view(1, 1)
        return self.base_mlp(logn).view(1, num_heads, 1, head_dim) * (1 + torch.tanh(self.query_mlp(q))) * q
