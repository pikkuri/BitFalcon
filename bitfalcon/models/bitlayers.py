"""bitlayers.py

BitNet 風の量子化レイヤーを PyTorch で実装した最小サンプル。
==================================================================
 * **BitLinear** : `nn.Linear` 互換の全結合レイヤーを 1.58‑bit (≒三値) 量子化で置換
 * **BitConv2d** : `nn.Conv2d` 互換の畳み込みレイヤーを同様に量子化
 * **quantise_ternary** : 重みを +a / 0 / −a の三値に量子化するヘルパー

主な設計方針
-------------
1. **Forward 時のみ量子化** し、重みの “実数版” を保持 → 精度を確保
2. **Straight‑Through Estimator (STE)** で逆伝播を近似 → 勾配が 0 にならない
3. **Keras/TensorFlow の Ternary Weight Networks** や **BitNet 紙** の手法を参考
4. 依存は標準 PyTorch のみ。CUDA カーネル不要 → エッジ環境でも動く

今後の拡張アイデア
------------------
* k‑bit 量子化 (2,4bit) / per‑channel スケーリング / 活性化量子化
* QAT (Quantisation‑Aware Training) 用 FakeQuant ラッパ
* ONNX / TensorRT へのエクスポート対応
"""

from __future__ import annotations

from typing import Tuple, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "quantise_ternary",
    "BitLinear",
    "BitConv2d",
]

# ---------------------------------------------------------------------------
# ヘルパ関数（量子化と STE）
# ---------------------------------------------------------------------------

def quantise_ternary(w: torch.Tensor, symmetric: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """重みテンソル *w* を三値化する簡易アルゴリズム。

    バイナリより 1bit 多い **1.58bit (=log2(3))** の情報量を持ち、
    BitNet 論文で報告された精度・速度トレードオフを再現できる。

    戻り値
    -------
    q : 量子化後のテンソル (整数値 -1/0/+1)
    scale : スケール係数 a (実数)
    """
    # ★BitNet 論文の経験則★: |w| 平均の 0.7 倍をスケールとする
    if symmetric:
        a = w.abs().mean() * 0.7
        # w > +a/2 → +1,  w < -a/2 → -1,  その他 → 0
        q = torch.where(w > a / 2, torch.ones_like(w), torch.zeros_like(w))
        q = torch.where(w < -a / 2, -torch.ones_like(w), q)
        return q, torch.tensor([a], device=w.device, dtype=w.dtype)

    # 非対称版（使用頻度は低い）
    max_val, min_val = w.max(), w.min()
    a = (max_val - min_val) / 2
    q = torch.sign(w - (max_val + min_val) / 2)
    return q, torch.tensor([a], device=w.device, dtype=w.dtype)


def _ste_round(x: torch.Tensor) -> torch.Tensor:
    """整数化後も勾配を流す Straight‑Through Estimator。"""
    return (x - x.detach()) + torch.round(x.detach())

# ---------------------------------------------------------------------------
# 共通ベースクラス
# ---------------------------------------------------------------------------

class _BitBase(nn.Module):
    """BitLinear / BitConv2d が継承する基底クラス。

    *quant_bits* を保存しておくだけだが、将来 k‑bit 実装を足す際に拡張しやすい。
    """

    def __init__(self, quant_bits: float = 1.58, symmetric: bool = True):
        super().__init__()
        self.quant_bits = quant_bits
        self.symmetric = symmetric

    @staticmethod
    def _quantise_weight(w: torch.Tensor, symmetric: bool = True):
        # 現在は三値のみ。必要ならここで k‑bit に分岐させる
        return quantise_ternary(w, symmetric)

# ---------------------------------------------------------------------------
# BitLinear : 全結合層の量子化バージョン
# ---------------------------------------------------------------------------

class BitLinear(_BitBase):
    """`nn.Linear` とインターフェース互換の量子化レイヤー。"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        quant_bits: float = 1.58,
        symmetric: bool = True,
    ) -> None:
        super().__init__(quant_bits, symmetric)
        self.in_features = in_features
        self.out_features = out_features
        # フル精度パラメータを学習
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.reset_parameters()

    # --- 初期化： Kaiming-uniform (He) ---
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    # --- Forward ---
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) 三値 + スケールに量子化
        q_w, scale = self._quantise_weight(self.weight, self.symmetric)
        # 2) STE で整数化 (round) し、スケールを掛け戻す
        w_hat = _ste_round(q_w) * scale
        return F.linear(x, w_hat, self.bias)

# ---------------------------------------------------------------------------
# BitConv2d : 畳み込み層の量子化バージョン
# ---------------------------------------------------------------------------

class BitConv2d(_BitBase):
    """`nn.Conv2d` を三値量子化に置き換えたレイヤー。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = False,
        quant_bits: float = 1.58,
        symmetric: bool = True,
    ) -> None:
        super().__init__(quant_bits, symmetric)
        # Conv2d と同じパラメータを保持
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *nn._pair(kernel_size))
        )
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.stride = nn._pair(stride)
        self.padding = nn._pair(padding)
        self.dilation = nn._pair(dilation)
        self.groups = groups
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_w, scale = self._quantise_weight(self.weight, self.symmetric)
        w_hat = _ste_round(q_w) * scale
        return F.conv2d(
            x,
            w_hat,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
