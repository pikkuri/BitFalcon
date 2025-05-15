# BitFalcon – BitNet Layer Reference

> **Purpose**  
> This document summarizes the core 1.58‑bit *BitNet* layers that will power **BitFalcon**, and shows how they can replace common layers in YOLO‑like CNN/Transformer hybrid detectors.

---

## 1. Quick Facts about BitNet

| Property | Value |
|----------|-------|
| Weight precision | **1.58 bits** (ternary \{-1, 0, 1\}) |
| Activation precision | 8‑bit (default) |
| Training method | Quantization‑aware from scratch or post‑training re‑quantization |
| Key advantage | ×20 ‑ ×25 smaller weight size & 2‑3× faster dense math on CPU/GPU |
| Key trade‑off | Slight accuracy drop, compensated by larger width/depth |

---

## 2. Core Layer Types

### 2.1 `BitLinear`
* **What it is**   Ternary fully‑connected layer: `y = α·(B ⊙ ŵ)x + β`  
* **PyTorch alias**   `bitlinear_pytorch.BitLinear(in_features, out_features)`  
* **Common uses**   • MLP blocks   • Attention Q / K / V projections   • YOLO head FC layers  

---

### 2.2 `BitFFN`
Feed‑forward block that stacks  
`BitLinear → GELU/SiLU → BitLinear`  
and optionally a residual connection.  
Acts as a plug‑in replacement for Transformer MLPs or CSP bottlenecks.

---

### 2.3 `BitAttention`
Multi‑head attention where **all** linear projections (Q, K, V, O) are `BitLinear`.  
Scaling factors are folded into softmax to avoid de‑quant overhead.

---

### 2.4 `BitConv2d` *(vision add‑on)*
Depth‑wise or group‑wise 1.58‑bit convolution. Implemented by:  
1. Binarising filters per‑group,  
2. Packing to int32,  
3. Using XNOR‑popcount kernels.  
**Drop‑in for:** `nn.Conv2d` (3×3 / 1×1) used in YOLO backbones & PAN.

---

### 2.5 Auxiliary Layers
| Original          | BitNet‑friendly variant |
|-------------------|-------------------------|
| `nn.LayerNorm`    | **BitLayerNorm** (folded scale/shift into int8) |
| `nn.BatchNorm2d`  | **BitBatchNorm** (int8 affine) |
| `SiLU / Swish`    | Int8 LUT‑based SiLU |
| Positional Embed  | 8‑bit sinusoidal or rotary |

---

## 3. Replacement Map for a YOLO‑like Detector

| Standard YOLO Layer | BitFalcon Substitute | Notes |
|---------------------|----------------------|-------|
| `Conv2d 3×3`        | `BitConv2d 3×3`      | keep stride/padding |
| `C3/C2f` bottleneck | `BitFFN`             | remove concat if using MLP |
| PAN neck conv       | `BitConv2d 1×1`      | or small `BitLinear` if flattened |
| Detection head FC   | `BitLinear` cascade  | 2–3 cascaded for precision |
| SPPF pooling        | *unchanged* or int8  | pooling has no weights |

---

## 4. Practical Usage Patterns

### 4.1 Swapping a Linear Layer
```python
from bitlinear_pytorch import BitLinear
old = nn.Linear(1024, 256)
new = BitLinear(1024, 256)   # plug & play
```

### 4.2 Building a BitFFN Block
```python
class BitFFN(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = BitLinear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = BitLinear(hidden, dim)
    def forward(self, x):
        return x + self.fc2(self.act(self.fc1(x)))
```

### 4.3 Quantization‑Aware Training Flags
```python
for m in model.modules():
    if isinstance(m, BitLinear):
        m.set_qat(True)      # enables STE and ternary clamp
```

---

## 5. Tips for Maintaining Accuracy

1. **Scale width & depth** – follow YOLO‑style multipliers (e.g. n/s/m/l).  
2. **8‑bit activations** – keep activations in int8 to prevent information loss.  
3. **Knowledge distillation** – train BitFalcon from a full‑precision teacher.  
4. **Warm‑up in FP16** – first 10 % epochs in FP16, then switch to ternary.  

---

## 6. Reference Implementations

* [`bitlinear-pytorch`](https://github.com/ingur/bitlinear-pytorch) – Minimal BitLinear layer  
* [`microsoft/BitNet`](https://github.com/microsoft/BitNet) – Official 1‑bit LLM kernels  
* [`kyegomez/BitNet`](https://github.com/kyegomez/BitNet) – Clean Transformer demo  
* [`AlarioAI/bitnet`](https://github.com/AlarioAI/bitnet) – Training scripts  

---

### Version
`BitFalcon docs v0.1 – 2025‑05‑16`

