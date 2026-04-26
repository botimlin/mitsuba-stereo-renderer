"""
EXR 檔案檢視工具
將 EXR 轉換為 PNG 以便檢視

Usage:
    # 單一檔案
    python exr_viewer.py path/to/file.exr

    # 整個場景資料夾
    python exr_viewer.py path/to/scene_folder --all

    # 指定輸出目錄
    python exr_viewer.py path/to/file.exr -o output_dir

Copyright (c) 2025-2026 Po-Ting Lin
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np

try:
    import OpenEXR
    import Imath
except ImportError:
    print("Error: OpenEXR not installed. Run: pip install OpenEXR Imath")
    sys.exit(1)

try:
    import cv2
except ImportError:
    print("Error: OpenCV not installed. Run: pip install opencv-python")
    sys.exit(1)


def read_exr(filepath: str) -> tuple:
    """
    讀取 EXR 檔案

    Returns:
        (data, channels, header_info)
    """
    exr_file = OpenEXR.InputFile(filepath)
    header = exr_file.header()

    # 取得解析度
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    # 取得通道資訊
    channels = list(header['channels'].keys())

    # 讀取資料
    pt = Imath.PixelType(Imath.PixelType.FLOAT)

    data = {}
    for ch in channels:
        raw = exr_file.channel(ch, pt)
        arr = np.frombuffer(raw, dtype=np.float32).reshape(height, width)
        data[ch] = arr

    # Header 資訊
    header_info = {
        'width': width,
        'height': height,
        'channels': channels,
    }

    return data, channels, header_info


def exr_to_image(data: dict, channels: list) -> np.ndarray:
    """
    將 EXR 資料轉換為可視化的圖像
    """
    # 判斷類型
    if 'R' in channels and 'G' in channels and 'B' in channels:
        # RGB 圖像
        r = data['R']
        g = data['G']
        b = data['B']
        img = np.stack([b, g, r], axis=-1)  # BGR for OpenCV

    elif 'Y' in channels:
        # 灰度圖（深度/視差）
        img = data['Y']

    elif len(channels) == 1:
        # 單通道
        img = data[channels[0]]

    else:
        # 取第一個通道
        img = data[channels[0]]

    return img


def normalize_for_display(img: np.ndarray, percentile: float = 99) -> np.ndarray:
    """
    正規化圖像以便顯示
    使用 percentile 避免極端值影響
    """
    if img.ndim == 3:
        # RGB
        img_clipped = img.copy()
        for i in range(3):
            vmin = np.percentile(img[:, :, i], 100 - percentile)
            vmax = np.percentile(img[:, :, i], percentile)
            if vmax > vmin:
                img_clipped[:, :, i] = np.clip((img[:, :, i] - vmin) / (vmax - vmin), 0, 1)
            else:
                img_clipped[:, :, i] = 0
        return (img_clipped * 255).astype(np.uint8)
    else:
        # Grayscale
        valid = img[np.isfinite(img)]
        if len(valid) == 0:
            return np.zeros_like(img, dtype=np.uint8)

        vmin = np.percentile(valid, 100 - percentile)
        vmax = np.percentile(valid, percentile)

        if vmax > vmin:
            img_norm = np.clip((img - vmin) / (vmax - vmin), 0, 1)
        else:
            img_norm = np.zeros_like(img)

        # 標記無效值為紅色（如果輸出為彩色）
        return (img_norm * 255).astype(np.uint8)


def apply_colormap(gray: np.ndarray, colormap: int = cv2.COLORMAP_VIRIDIS) -> np.ndarray:
    """
    對灰度圖應用 colormap
    """
    return cv2.applyColorMap(gray, colormap)


def print_stats(data: dict, channels: list, filepath: str):
    """
    打印 EXR 統計資訊
    """
    print(f"\n{'='*60}")
    print(f"File: {filepath}")
    print(f"{'='*60}")

    for ch in channels:
        arr = data[ch]
        valid = arr[np.isfinite(arr)]

        print(f"\nChannel: {ch}")
        print(f"  Shape: {arr.shape}")
        print(f"  Dtype: {arr.dtype}")

        if len(valid) > 0:
            print(f"  Min:   {valid.min():.6f}")
            print(f"  Max:   {valid.max():.6f}")
            print(f"  Mean:  {valid.mean():.6f}")
            print(f"  Std:   {valid.std():.6f}")

            # 檢查無效值
            nan_count = np.isnan(arr).sum()
            inf_count = np.isinf(arr).sum()
            if nan_count > 0 or inf_count > 0:
                print(f"  NaN:   {nan_count} pixels")
                print(f"  Inf:   {inf_count} pixels")
        else:
            print(f"  (All values are invalid)")


def convert_exr(input_path: str, output_path: str = None,
                show_stats: bool = True, use_colormap: bool = True):
    """
    轉換單一 EXR 檔案
    """
    input_path = Path(input_path)

    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        return

    # 讀取 EXR
    data, channels, header_info = read_exr(str(input_path))

    # 打印統計
    if show_stats:
        print_stats(data, channels, str(input_path))

    # 轉換為圖像
    img = exr_to_image(data, channels)

    # 正規化
    img_norm = normalize_for_display(img)

    # 如果是灰度且使用 colormap
    if img.ndim == 2 and use_colormap:
        img_out = apply_colormap(img_norm)
    else:
        img_out = img_norm

    # 輸出路徑
    if output_path is None:
        output_path = input_path.with_suffix('.png')
    else:
        output_path = Path(output_path)
        if output_path.is_dir():
            output_path = output_path / (input_path.stem + '.png')

    # 儲存
    cv2.imwrite(str(output_path), img_out)
    print(f"\nSaved: {output_path}")

    return img_out


def convert_scene_folder(scene_dir: str, output_dir: str = None):
    """
    轉換整個場景資料夾的所有 EXR
    """
    scene_dir = Path(scene_dir)

    if output_dir is None:
        output_dir = scene_dir / 'preview'
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 找所有 EXR
    exr_files = list(scene_dir.glob('*.exr'))

    if not exr_files:
        print(f"No EXR files found in {scene_dir}")
        return

    print(f"Found {len(exr_files)} EXR files")

    for exr_path in exr_files:
        output_path = output_dir / (exr_path.stem + '.png')
        convert_exr(str(exr_path), str(output_path), show_stats=True)

    print(f"\nAll files saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='EXR to PNG converter')
    parser.add_argument('input', help='Input EXR file or scene folder')
    parser.add_argument('-o', '--output', help='Output path (file or directory)')
    parser.add_argument('--all', action='store_true',
                        help='Convert all EXR files in the folder')
    parser.add_argument('--no-colormap', action='store_true',
                        help='Disable colormap for depth/disparity')
    parser.add_argument('--no-stats', action='store_true',
                        help='Disable statistics output')

    args = parser.parse_args()

    input_path = Path(args.input)

    if args.all or input_path.is_dir():
        convert_scene_folder(str(input_path), args.output)
    else:
        convert_exr(
            str(input_path),
            args.output,
            show_stats=not args.no_stats,
            use_colormap=not args.no_colormap
        )


if __name__ == '__main__':
    main()
