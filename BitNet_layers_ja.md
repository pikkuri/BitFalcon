# BitFalcon – BitNetレイヤーリファレンス

> **目的**  
> このドキュメントは、**BitFalcon**の基盤となる1.58ビットの*BitNet*レイヤーについて要約し、YOLO系のCNN/Transformerハイブリッド検出器における一般的なレイヤーをどのように置き換えられるかを示します。

---

## 1. BitNetの基本情報

| 特性 | 値 |
|----------|-------|
| 重み精度 | **1.58ビット**（3値 \{-1, 0, 1\}） |
| 活性化精度 | 8ビット（デフォルト） |
| 学習手法 | 量子化を考慮した初期学習または学習後の再量子化 |
| 主な利点 | CPU/GPUで重みサイズが20〜25倍小さく、密行列演算が2〜3倍高速 |
| トレードオフ | わずかな精度低下（より大きな幅/深さで補償可能） |

---

## 2. 基本レイヤータイプ

### 2.1 `BitLinear`
* **概要**   3値の全結合層：`y = α·(B ⊙ ŵ)x + β`  
* **PyTorchエイリアス**   `bitlinear_pytorch.BitLinear(in_features, out_features)`  
* **一般的な用途**   • MLPブロック   • アテンションのQ/K/V射影   • YOLOヘッドの全結合層  

---

### 2.2 `BitFFN`
以下の構造を積み重ねるフィードフォワードブロック：  
`BitLinear → GELU/SiLU → BitLinear`  
オプションで残差接続も可能。  
TransformerのMLPやCSPボトルネックの代替として使用可能。

---

### 2.3 `BitAttention`
**すべての**線形射影（Q, K, V, O）が`BitLinear`となるマルチヘッドアテンション。  
スケーリング係数はソフトマックスに組み込まれ、逆量子化のオーバーヘッドを回避します。

---

### 2.4 `BitConv2d` *（ビジョン用拡張）*
深さ方向またはグループ単位の1.58ビット畳み込み。実装方法：  
1. フィルターをグループごとに二値化  
2. int32にパッキング  
3. XNORポップカウントカーネルを使用  
**代替対象：** YOLOバックボーンやPANで使用される`nn.Conv2d`（3×3 / 1×1）

---

### 2.5 補助レイヤー
| オリジナル | BitNet対応バリアント |
|-------------------|-------------------------|
| `nn.LayerNorm`    | **BitLayerNorm**（int8にスケール/シフトを統合） |
| `nn.BatchNorm2d`  | **BitBatchNorm**（int8アフィン） |
| `SiLU / Swish`    | ルックアップテーブルベースのint8 SiLU |
| 位置エンベディング  | 8ビット正弦波または回転型 |

---

## 3. YOLO系検出器の置換マップ

| 標準YOLOレイヤー | BitFalcon代替 | 備考 |
|---------------------|----------------------|-------|
| `Conv2d 3×3`        | `BitConv2d 3×3`      | ストライド/パディングは維持 |
| `C3/C2f`ボトルネック | `BitFFN`             | MLPを使用する場合は連結を削除 |
| PANネックの畳み込み  | `BitConv2d 1×1`      | または平坦化する場合は小さな`BitLinear` |
| 検出ヘッドFC        | `BitLinear`カスケード | 精度向上のため2〜3段カスケード接続 |
| SPPFプーリング      | *変更なし*またはint8  | プーリングには重みがない |

---

## 4. 実用的な使用パターン

### 4.1 線形レイヤーの置き換え
```python
from bitlinear_pytorch import BitLinear
old = nn.Linear(1024, 256)
new = BitLinear(1024, 256)   # そのまま置き換え可能
```

### 4.2 BitFFNブロックの構築
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

### 4.3 量子化を考慮した学習フラグ
```python
for m in model.modules():
    if isinstance(m, BitLinear):
        m.set_qat(True)      # STEと3値クランプを有効化
```

---

## 5. 精度維持のためのヒント

1. **幅と深さの拡大** – YOLO形式の倍率（n/s/m/lなど）に従う。  
2. **8ビット活性化** – 情報損失を防ぐため、活性化をint8に保つ。  
3. **知識蒸留** – 全精度の教師モデルからBitFalconを学習させる。  
4. **FP16でのウォームアップ** – 最初の10%のエポックをFP16で実行し、その後3値に切り替える。  

---

## 6. 参考実装

* [`bitlinear-pytorch`](https://github.com/ingur/bitlinear-pytorch) – 最小限のBitLinearレイヤー  
* [`microsoft/BitNet`](https://github.com/microsoft/BitNet) – 公式1ビットLLMカーネル  
* [`kyegomez/BitNet`](https://github.com/kyegomez/BitNet) – クリーンなTransformerデモ  
* [`AlarioAI/bitnet`](https://github.com/AlarioAI/bitnet) – トレーニングスクリプト  

---

### バージョン
`BitFalcon ドキュメント v0.1 – 2025-05-16`
