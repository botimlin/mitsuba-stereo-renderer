#!/usr/bin/env python3
"""
EXR 轉 PNG 簡易工具
==================

用法：
    python exr_to_png.py *.exr
    python exr_to_png.py --unified left.exr right.exr
    python exr_to_png.py --dir /workspace/output_stage1
"""

import numpy as np
import os
import sys
import argparse
from pathlib import Path

try:
    import OpenEXR
    import Imath
    HAS_OPENEXR = True
except ImportError:
    HAS_OPENEXR = False

try:
    import mitsuba as mi
    HAS_MITSUBA = True
except ImportError:
    HAS_MITSUBA = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def load_exr(path):
    """載入 EXR"""
    if not os.path.exists(path):
        print(f"[ERROR] 文件不存在: {path}")
        return None

    # 優先使用 OpenEXR 庫
    if HAS_OPENEXR:
        try:
            exr_file = OpenEXR.InputFile(path)
            header = exr_file.header()
            dw = header['dataWindow']
            width = dw.max.x - dw.min.x + 1
            height = dw.max.y - dw.min.y + 1

            pt = Imath.PixelType(Imath.PixelType.FLOAT)
            channels = list(header['channels'].keys())

            if len(channels) == 1:
                # 單通道 (深度/視差)
                data = exr_file.channel(channels[0], pt)
                image = np.frombuffer(data, dtype=np.float32).reshape(height, width)
            else:
                # 多通道 (RGB)
                arrays = []
                for ch in ['R', 'G', 'B']:
                    if ch in channels:
                        data = exr_file.channel(ch, pt)
                        arr = np.frombuffer(data, dtype=np.float32).reshape(height, width)
                        arrays.append(arr)
                if arrays:
                    image = np.stack(arrays, axis=-1)
                else:
                    # 使用前三個通道
                    for ch in channels[:3]:
                        data = exr_file.channel(ch, pt)
                        arr = np.frombuffer(data, dtype=np.float32).reshape(height, width)
                        arrays.append(arr)
                    image = np.stack(arrays, axis=-1) if len(arrays) > 1 else arrays[0]

            return image
        except Exception as e:
            print(f"[WARN] OpenEXR 載入失敗: {e}")

    if HAS_MITSUBA:
        try:
            bitmap = mi.Bitmap(path)
            image = np.array(bitmap).astype(np.float32)
            return image
        except:
            pass

    if HAS_CV2:
        try:
            os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
            image = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if image is not None:
                return image.astype(np.float32)
        except:
            pass

    print(f"[ERROR] 無法載入: {path}")
    return None


def save_png(image, output_path, vmin=None, vmax=None, gamma=2.2):
    """保存為 PNG"""
    # 處理多通道
    if image.ndim == 3:
        if image.shape[2] > 3:
            image = image[:, :, :3]
    
    # 確定範圍
    if vmin is None:
        vmin = np.min(image[np.isfinite(image)])
    if vmax is None:
        vmax = np.max(image[np.isfinite(image)])
    
    if vmax - vmin < 1e-10:
        vmax = vmin + 1
    
    # 正規化 + gamma
    result = (image - vmin) / (vmax - vmin)
    result = np.clip(result, 0, 1)
    result = np.power(result, 1.0 / gamma)
    result = (result * 255).astype(np.uint8)
    
    # 保存
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    if HAS_CV2:
        if result.ndim == 3 and result.shape[2] >= 3:
            result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, result)
    else:
        from PIL import Image
        if result.ndim == 2:
            img = Image.fromarray(result, mode='L')
        else:
            img = Image.fromarray(result, mode='RGB')
        img.save(output_path)
    
    return True


def convert_single(exr_path, output_dir=None):
    """轉換單個文件"""
    image = load_exr(exr_path)
    if image is None:
        return False
    
    vmin = float(np.min(image))
    vmax = float(np.max(image))
    
    if output_dir:
        png_path = os.path.join(output_dir, Path(exr_path).stem + '.png')
    else:
        png_path = str(Path(exr_path).with_suffix('.png'))
    
    save_png(image, png_path)
    print(f"[OK] {exr_path}")
    print(f"     範圍: [{vmin:.4f}, {vmax:.4f}] -> {png_path}")
    return True


def convert_unified(exr_paths, output_dir=None):
    """統一範圍轉換（用於比較）"""
    # 載入所有圖像
    images = []
    global_min = float('inf')
    global_max = float('-inf')
    
    for path in exr_paths:
        image = load_exr(path)
        if image is not None:
            images.append((path, image))
            global_min = min(global_min, float(np.min(image)))
            global_max = max(global_max, float(np.max(image)))
    
    if not images:
        print("[ERROR] 沒有載入任何圖像")
        return
    
    print(f"\n統一範圍: [{global_min:.4f}, {global_max:.4f}]")
    print(f"比值: {global_max / (global_min + 1e-10):.2f}x\n")
    
    # 轉換
    out_dir = output_dir or './png_unified'
    os.makedirs(out_dir, exist_ok=True)
    
    for path, image in images:
        png_name = Path(path).stem + '_unified.png'
        png_path = os.path.join(out_dir, png_name)
        save_png(image, png_path, vmin=global_min, vmax=global_max)
        print(f"[OK] {path} -> {png_path}")


def convert_directory(input_dir, output_dir=None, unified=False):
    """轉換目錄中所有 EXR"""
    exr_files = sorted(Path(input_dir).glob('*.exr'))
    
    if not exr_files:
        print(f"[ERROR] 目錄中沒有 EXR 文件: {input_dir}")
        return
    
    print(f"找到 {len(exr_files)} 個 EXR 文件\n")
    
    if unified:
        convert_unified([str(f) for f in exr_files], output_dir)
    else:
        out_dir = output_dir or os.path.join(input_dir, 'png')
        os.makedirs(out_dir, exist_ok=True)
        for exr_path in exr_files:
            convert_single(str(exr_path), out_dir)


def main():
    parser = argparse.ArgumentParser(description='EXR 轉 PNG')
    parser.add_argument('files', nargs='*', help='EXR 文件')
    parser.add_argument('--dir', '-d', type=str, help='轉換目錄中所有 EXR')
    parser.add_argument('--output', '-o', type=str, help='輸出目錄')
    parser.add_argument('--unified', '-u', action='store_true', 
                        help='統一範圍轉換（用於比較 I∥ 和 I⊥）')
    
    args = parser.parse_args()
    
    if not HAS_OPENEXR and not HAS_MITSUBA and not HAS_CV2:
        print("[ERROR] 需要 OpenEXR, mitsuba 或 opencv")
        print("       pip install OpenEXR")
        return 1
    
    if args.dir:
        convert_directory(args.dir, args.output, args.unified)
    elif args.files:
        if args.unified:
            convert_unified(args.files, args.output)
        else:
            for f in args.files:
                convert_single(f, args.output)
    else:
        parser.print_help()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
