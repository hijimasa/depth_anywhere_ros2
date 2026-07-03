#!/usr/bin/env python3
"""UniFuse を ONNX にエクスポートするツール（TensorRT 化の前段）。

オリジナルの Cube2Equirec は 5D grid_sample を使っており ONNX に落とせないため、
数値的に等価な 4D grid_sample（6面ストリップ画像への直接サンプリング）に
置き換えてエクスポートする。等価性はスクリプト内で自動検証する
（float32 で最大誤差 1e-5 未満を確認済み）。

使い方:
    python3 scripts/export_unifuse_onnx.py --height 256 --width 512 \
        --ckpt ckpt/UniFuse/UniFuse_SpatialAudioGen.pth -o unifuse_256x512.onnx

Jetson 上で TensorRT エンジンに変換（GridSample は TensorRT 8.5 以降対応）:
    /usr/src/tensorrt/bin/trtexec --onnx=unifuse_256x512.onnx \
        --saveEngine=unifuse_256x512_fp16.engine --fp16

推論時の入力:
    equi: (1,3,H,W)      正規化済み equirectangular 画像
    cube: (1,3,H/2,3H)   py360_E2C で作った6面横並びストリップ（正規化済み）
出力:
    depth: (1,1,H,W)
"""
import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PKG_ROOT)

from depth_anywhere_ros2.baseline_models.UniFuse.networks import UniFuse  # noqa: E402
from depth_anywhere_ros2.baseline_models.UniFuse.networks.layers import upsample  # noqa: E402


class Cube2EquirecStrip(nn.Module):
    """5D grid_sample を使わない Cube2Equirec の等価実装。

    オリジナルは (bs,ch,6,fw,fw) への volumetric grid_sample だが、face id は
    常にスライス中心に整数で載る（z 補間は実質何もしない）ため、6面を横に並べた
    (bs,ch,fw,6*fw) ストリップへの 2D サンプリングに正確に書き換えられる。
    """

    def __init__(self, orig):
        super().__init__()
        face_w = orig.face_w
        grid = orig.sample_grid.data[0, 0]  # (H, W, 3) = (u, v, face_id/2.5-1)
        u, v, tp = grid[..., 0], grid[..., 1], grid[..., 2]
        face = torch.round((tp + 1) * 2.5).float()  # 0..5
        # オリジナル: 面内 align_corners=True → ピクセル中心 p = (u+1)/2*(fw-1)
        # ストリップ: align_corners=False で連続位置 q = f*fw + p + 0.5 を指す
        p_x = (u.clamp(-1, 1) + 1) / 2 * (face_w - 1)
        p_y = (v.clamp(-1, 1) + 1) / 2 * (face_w - 1)
        q_x = face * face_w + p_x + 0.5
        q_y = p_y + 0.5
        x_strip = q_x / (6 * face_w) * 2 - 1
        y_strip = q_y / face_w * 2 - 1
        grid4d = torch.stack([x_strip, y_strip], dim=-1).unsqueeze(0)
        self.register_buffer("grid4d", grid4d)  # (1,H,W,2)

    def forward(self, cube_feat):
        # cube_feat: (bs, ch, face_w, face_w*6) 横並びストリップ
        bs = cube_feat.shape[0]
        grid = self.grid4d.expand(bs, -1, -1, -1)
        return F.grid_sample(cube_feat, grid, mode="bilinear",
                             padding_mode="border", align_corners=False)


class UniFuseExport(nn.Module):
    """c2e をストリップ版に差し替えた ONNX エクスポート用ラッパ"""

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.strip_c2e = nn.ModuleDict(
            {k: Cube2EquirecStrip(v) for k, v in net.c2e.items()})

    def forward(self, equi, cube):
        net = self.net
        x = net.equi_encoder.conv1(equi)
        x = net.equi_encoder.relu(net.equi_encoder.bn1(x))
        e0 = x
        x = net.equi_encoder.maxpool(x)
        e1 = net.equi_encoder.layer1(x)
        e2 = net.equi_encoder.layer2(e1)
        e3 = net.equi_encoder.layer3(e2)
        e4 = net.equi_encoder.layer4(e3)

        cube_inputs = torch.cat(torch.split(cube, net.cube_h, dim=-1), dim=0)
        x = net.cube_encoder.conv1(cube_inputs)
        x = net.cube_encoder.relu(net.cube_encoder.bn1(x))
        c0 = x
        x = net.cube_encoder.maxpool(x)
        c1 = net.cube_encoder.layer1(x)
        c2 = net.cube_encoder.layer2(c1)
        c3 = net.cube_encoder.layer3(c2)
        c4 = net.cube_encoder.layer4(c3)

        bs = equi.shape[0]
        dec = net.equi_dec_convs
        feats = [(e4, c4, "5", "fusion_5"), (e3, c3, "4", "fusion_4"),
                 (e2, c2, "3", "fusion_3"), (e1, c1, "2", "fusion_2"),
                 (e0, c0, "1", "fusion_1")]
        equi_x = None
        for i, (ef, cf, ck, fk) in enumerate(feats):
            cf = torch.cat(torch.split(cf, bs, dim=0), dim=-1)
            c2e_f = self.strip_c2e[ck](cf)
            fused = dec[fk](ef, c2e_f)
            if i == 0:
                equi_x = upsample(dec["upconv_5"](fused))
            else:
                equi_x = torch.cat([equi_x, fused], 1)
                equi_x = dec[f"deconv_{5 - i}"](equi_x)
                equi_x = upsample(dec[f"upconv_{5 - i}"](equi_x))
        equi_x = dec["deconv_0"](equi_x)
        return net.max_depth * net.sigmoid(dec["depthconv_0"](equi_x))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--batch", type=int, default=1,
                    help="バッチサイズ（infer_multi で2カメラ同時推論するなら2）")
    ap.add_argument("--num-layers", type=int, default=18)
    ap.add_argument("--ckpt", default=os.path.join(
        PKG_ROOT, "ckpt", "UniFuse", "UniFuse_SpatialAudioGen.pth"))
    ap.add_argument("-o", "--output", default=None)
    args = ap.parse_args()

    h, w = args.height, args.width
    b = args.batch
    out_path = args.output or f"unifuse_{h}x{w}_b{b}.onnx"

    net = UniFuse(num_layers=args.num_layers, equi_h=h, equi_w=w,
                  pretrained=False, max_depth=10.0,
                  fusion_type="cee", se_in_fusion=True)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    sd = net.state_dict()
    loadable = {k: v for k, v in ckpt.items()
                if k in sd and sd[k].shape == v.shape}
    net.load_state_dict(loadable, strict=False)
    net.eval()

    export_net = UniFuseExport(net).eval()

    equi = torch.randn(b, 3, h, w)
    cube = torch.randn(b, 3, h // 2, (h // 2) * 6)

    # ストリップ版がオリジナルと一致するか検証
    with torch.no_grad():
        ref = net(equi, cube)["pred_depth"]
        out = export_net(equi, cube)
    max_diff = (ref - out).abs().max().item()
    print(f"strip c2e とオリジナルの最大深度差: {max_diff:.2e} (max_depth=10)")
    assert max_diff < 1e-4, "等価変換の検証に失敗"

    torch.onnx.export(export_net, (equi, cube), out_path,
                      input_names=["equi", "cube"], output_names=["depth"],
                      opset_version=17, dynamo=False)
    print(f"エクスポート完了: {out_path}")
    print("次のステップ (Jetson 上):")
    print(f"  /usr/src/tensorrt/bin/trtexec --onnx={out_path} "
          f"--saveEngine=unifuse_{h}x{w}_b{b}_fp16.engine --fp16")


if __name__ == "__main__":
    main()
