import torch
import torch.nn as nn
import torch.nn.functional as F

# ========= 共通ユーティリティ ========= #
def autopad(k, p=None):            # Darknet 系の同義関数
    return k // 2 if p is None else p

# ---------- 7×7 Separable Conv ---------- #
class SepConv7x7(nn.Module):
    """7×7 depthwise separable convolution.
    - 深い受容野を低コストで確保し、位置情報を維持する。
    """
    def __init__(self, c1, c2):
        super().__init__()
        self.dw = nn.Conv2d(c1, c1, 7, 1, autopad(7), groups=c1, bias=False)
        self.pw = nn.Conv2d(c1, c2, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))

# ---------- R-ELAN Block ---------- #
class RELANBlock(nn.Module):
    """
    Residual Efficient Layer Aggregation Network block.
    * ELAN 由来の並列/密結合パスに residual を追加。
    * scaling=0.01 で数値安定性を確保（論文推奨）。
    """
    def __init__(self, c, scaling=0.01, depth=3, width=1):
        super().__init__()
        self.scaling = scaling
        self.convs = nn.ModuleList([
            nn.Conv2d(c, c * width, 3, 1, 1, bias=False) for _ in range(depth)
        ])
        self.bn = nn.BatchNorm2d(c * width)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        out = []
        y = x
        for conv in self.convs:
            y = self.act(conv(y))
            out.append(y)
        y = torch.cat(out, 1)                 # 特徴集約
        y = self.bn(y) * self.scaling         # residual scaling
        return self.act(x + y)                # 残差加算

# ---------- Area Attention (簡易版) ---------- #
class AreaAttention(nn.Module):
    """
    Query/Key/Value を特徴マップを格子状に分割（area）して計算。
    FlashAttention 実装に置き換えればメモリ効率が向上。
    """
    def __init__(self, dim, num_heads=8, area=4):
        super().__init__()
        self.area = area
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, x):
        b, c, h, w = x.shape
        # エリア分割（畳み込み的にリシェイプ）
        x = x.unfold(2, h // self.area, h // self.area)\
             .unfold(3, w // self.area, w // self.area)  # (b,c,a,a,ah,aw)
        x = x.contiguous().view(b, c, -1)                # token 化
        x = x.permute(0, 2, 1)                           # (b, token, c)
        y, _ = self.attn(x, x, x)
        y = y.permute(0, 2, 1).view(b, c, h, w)
        return y

# ---------- Backbone ---------- #
class Backbone(nn.Module):
    """
    入力: (B,3,640,640) 想定
    - Conv stem → SepConv7x7 → RELANBlock を段階的に配置。
    """
    def __init__(self, base_ch=32, depth_ratio=1.0, width_ratio=1.0):
        super().__init__()
        ch = int(base_ch * width_ratio)
        self.stem = nn.Conv2d(3, ch, 3, 2, 1, bias=False)
        self.stage1 = RELANBlock(ch, depth=int(3*depth_ratio))
        self.stage2 = RELANBlock(ch*2, depth=int(6*depth_ratio), width=2)
        self.stage3 = RELANBlock(ch*4, depth=int(6*depth_ratio), width=2)
        self.sepconv = SepConv7x7(ch*4, ch*4)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = F.max_pool2d(x, 2)
        x = self.stage2(x)
        x = F.max_pool2d(x, 2)
        x = self.stage3(x)
        x = self.sepconv(x)
        return x  # P5 相当

# ---------- Neck ---------- #
class Neck(nn.Module):
    """
    - FPN/PAN 風の upsample + concat
    - AreaAttention を要所で挿入し情報強調
    """
    def __init__(self, ch, width_ratio=1.0):
        super().__init__()
        self.reduce = nn.Conv2d(ch, int(ch*width_ratio), 1, 1, 0, bias=False)
        self.attn = AreaAttention(int(ch*width_ratio))
        self.fuse = nn.Conv2d(int(ch*width_ratio), int(ch*width_ratio), 3, 1, 1, bias=False)

    def forward(self, x):
        x = self.reduce(x)
        x = self.attn(x)
        x = self.fuse(x)
        return x  # 統合特徴

# ---------- Head ---------- #
class DetectionHead(nn.Module):
    """
    - 3 スケール (P3,P4,P5) を前提とした YOLO 風ヘッド。
    - 量子化を想定し BitLinear でも差し替え可。
    """
    def __init__(self, ch, num_classes=80, anchors=3):
        super().__init__()
        self.cls_conv = nn.Conv2d(ch, anchors * num_classes, 1)
        self.box_conv = nn.Conv2d(ch, anchors * 5, 1)  # x,y,w,h,obj

    def forward(self, x):
        cls = self.cls_conv(x)
        box = self.box_conv(x)
        return torch.cat([box, cls], 1)

# ---------- 完成モデル ---------- #
class YOLOv12_Lite(nn.Module):
    """
    depth_ratio, width_ratio を変えるだけで n/s/m/l/x を生成可。
    """
    def __init__(self, num_classes=80, depth_ratio=0.33, width_ratio=0.50):
        super().__init__()
        self.backbone = Backbone(32, depth_ratio, width_ratio)
        self.neck = Neck(32*4, width_ratio)
        self.head = DetectionHead(int(32*4*width_ratio), num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        feats = self.neck(feats)
        preds = self.head(feats)
        return preds
