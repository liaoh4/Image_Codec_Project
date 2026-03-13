"""
visualize.py
依赖：benchmark.py 已跑完，results/kodak/ 目录下有解码图片和 CSV
用法：python visualize.py
"""
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim_fn

# ─────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────
RESULTS_DIR = 'results/kodak'
OUTPUT_DIR  = 'results/visualization'
KODAK_DIR   = 'data/raw'
CSV_PATH    = f'{RESULTS_DIR}/kodak_metrics_all.csv'

FOCAL_IMAGES = {
    'kodim01': 'Building',
    'kodim14': 'Boating',
    'kodim15': 'Portrait',
    'kodim21': 'Lighthouse',
    'kodim22': 'View',
    'kodim23': 'Parrot',

}

TARGET_BPPS = {'low': 0.5, 'high': 2.5}

CODECS = ['JPEG', 'BPG', 'JPEG-XL']

COLORS = {
    'JPEG':    'tomato',
    'BPG':     'steelblue',
    'JPEG-XL': 'mediumpurple',
}
MARKERS = {
    'JPEG':    'o',
    'BPG':     'D',
    'JPEG-XL': '*',
}

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────
def load_img(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.dtype == np.uint16:
        img = (img / 256).astype(np.uint8)
    if len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def find_decoded_path(img_name, codec, param, results_dir):
    label_map = {
        'JPEG':    'q',
        'BPG':     'crf',
        'JPEG-XL': 'd',
    }
    label = label_map.get(codec, 'q')
    if codec == 'JPEG-XL':
        
        param_str = str(float(param))  # 1.0→'1.0', 0.4→'0.4', 4.5→'4.5'
        
    else:
        param_str = str(int(float(param)))
    path = f"{results_dir}/{img_name}_dec_{codec.lower()}_{label}{param_str}.png"
    return path if os.path.exists(path) else None


def find_closest_param(df, img_name, codec, target_bpp):
    sub = df[(df['Image'] == img_name) & (df['Codec'] == codec)]
    if sub.empty:
        return None, None
    idx = (sub['BPP'] - target_bpp).abs().idxmin()
    row = sub.loc[idx]
    return row['Param'], row['BPP']


def auto_crop_region_from_diff(diff, patch_size=96, stride=48):
    h, w = diff.shape
    best_score = -1
    best_xy = (0, 0)
    for y in range(0, h - patch_size, stride):
        for x in range(0, w - patch_size, stride):
            score = diff[y:y+patch_size, x:x+patch_size].mean()
            if score > best_score:
                best_score = score
                best_xy = (x, y)
    return (*best_xy, patch_size, patch_size)


def interpolate_aligned(df, codec, metric, bpp_grid):
    """
    对每张图单独插值到 bpp_grid，再求均值和标准差。
    超出该图实际 BPP 范围的点设为 nan，避免外推。
    要求至少一半图片有数据才保留该网格点。
    """
    per_image = []
    for img in df['Image'].unique():
        sub = df[(df['Image'] == img) & (df['Codec'] == codec)].sort_values('BPP')
        if len(sub) < 2:
            continue
        interp = np.interp(bpp_grid, sub['BPP'], sub[metric],
                           left=np.nan, right=np.nan)
        interp[bpp_grid < sub['BPP'].min()] = np.nan
        interp[bpp_grid > sub['BPP'].max()] = np.nan
        per_image.append(interp)
    arr = np.array(per_image)
    mean = np.nanmean(arr, axis=0)
    std  = np.nanstd(arr, axis=0)
    valid_count = np.sum(~np.isnan(arr), axis=0)
    mean[valid_count < len(per_image) // 2] = np.nan
    std [valid_count < len(per_image) // 2] = np.nan
    return mean, std

# ─────────────────────────────────────────────
# 图2：三码率点视觉对比 + 自动局部放大
# ─────────────────────────────────────────────
def plot_visual_comparison(df):
    print("Plotting visual comparisons...")

    for img_name, desc in FOCAL_IMAGES.items():
        orig = load_img(f'{KODAK_DIR}/{img_name}.png')
        if orig is None:
            print(f"  SKIP {img_name}: original not found")
            continue

        for bpp_label, target_bpp in TARGET_BPPS.items():
            decoded_imgs, meta = {}, {}

            for codec in CODECS:
                param, actual_bpp = find_closest_param(df, img_name, codec, target_bpp)
                if param is None:
                    continue
                dec_path = find_decoded_path(img_name, codec, param, RESULTS_DIR)
                if dec_path is None:
                    continue
                dec = load_img(dec_path)
                if dec is None:
                    continue
                h = min(orig.shape[0], dec.shape[0])
                w = min(orig.shape[1], dec.shape[1])
                decoded_imgs[codec] = dec[:h, :w]
                p = psnr(orig[:h, :w], dec[:h, :w])
                s = ssim_fn(orig[:h, :w], dec[:h, :w], channel_axis=2)
                meta[codec] = {'bpp': actual_bpp, 'psnr': p, 'ssim': s}

            if not decoded_imgs:
                continue

            n_cols = len(decoded_imgs) + 1
            fig, axes = plt.subplots(2, n_cols, figsize=(n_cols * 3.2, 7))

            h = min(orig.shape[0], min(d.shape[0] for d in decoded_imgs.values()))
            w = min(orig.shape[1], min(d.shape[1] for d in decoded_imgs.values()))
            orig_crop = orig[:h, :w]

            combined_diff = np.zeros((h, w), dtype=float)
            for dec in decoded_imgs.values():
                diff = np.abs(orig_crop.astype(float) - dec[:h, :w].astype(float)).mean(axis=2)
                combined_diff = np.maximum(combined_diff, diff)

            cx, cy, cw, ch = auto_crop_region_from_diff(combined_diff,patch_size=96)

            axes[0][0].imshow(bgr2rgb(orig))
            rect = patches.Rectangle((cx, cy), cw, ch,
                                      linewidth=2, edgecolor='red', facecolor='none')
            axes[0][0].add_patch(rect)
            axes[0][0].set_title('Original', fontsize=10, fontweight='bold')
            axes[0][0].axis('off')

            axes[1][0].imshow(bgr2rgb(orig[cy:cy+ch, cx:cx+cw]))
            axes[1][0].set_title('Original (crop)', fontsize=9)
            axes[1][0].axis('off')

            for col, (codec, dec) in enumerate(decoded_imgs.items(), 1):
                axes[0][col].imshow(bgr2rgb(dec))
                rect2 = patches.Rectangle((cx, cy), cw, ch,
                                           linewidth=2, edgecolor='red', facecolor='none')
                axes[0][col].add_patch(rect2)
                axes[0][col].set_title(
                    f'{codec}\nBPP={meta[codec]["bpp"]:.3f}\n'
                    f'PSNR={meta[codec]["psnr"]:.1f}dB  SSIM={meta[codec]["ssim"]:.3f}',
                    fontsize=9, color=COLORS.get(codec, 'black')
                )
                axes[0][col].axis('off')

                crop_h = min(ch, dec.shape[0] - cy)
                crop_w = min(cw, dec.shape[1] - cx)
                axes[1][col].imshow(bgr2rgb(dec[cy:cy+crop_h, cx:cx+crop_w]))
                axes[1][col].set_title('(crop)', fontsize=9)
                axes[1][col].axis('off')

            plt.suptitle(
                f'{img_name} — {desc}\nTarget BPP ≈ {target_bpp} ({bpp_label})',
                fontsize=12
            )
            plt.tight_layout()
            path = f'{OUTPUT_DIR}/02_{img_name}_{bpp_label}_visual.png'
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved: {path}")

# ─────────────────────────────────────────────
# 图3：像素误差热力图（中码率）
# ─────────────────────────────────────────────
def plot_diff_heatmap(df):
    print("Plotting difference heatmaps...")

    for img_name in FOCAL_IMAGES:
        orig = load_img(f'{KODAK_DIR}/{img_name}.png')
        if orig is None:
            continue

        # 分别加载低码率和高码率的解码图
        rows_data = {}
        for bpp_label, target_bpp in [('low', TARGET_BPPS['low']), ('high', TARGET_BPPS['high'])]:
            decoded_imgs, meta = {}, {}
            for codec in CODECS:
                param, actual_bpp = find_closest_param(df, img_name, codec, target_bpp)
                if param is None:
                    continue
                dec_path = find_decoded_path(img_name, codec, param, RESULTS_DIR)
                if dec_path is None:
                    continue
                dec = load_img(dec_path)
                if dec is None:
                    continue
                h = min(orig.shape[0], dec.shape[0])
                w = min(orig.shape[1], dec.shape[1])
                decoded_imgs[codec] = dec[:h, :w]
                meta[codec] = {'bpp': actual_bpp}
            rows_data[bpp_label] = (decoded_imgs, meta, target_bpp)

        if not any(rows_data[k][0] for k in rows_data):
            continue

        n = len(CODECS)
        fig, axes = plt.subplots(2, n, figsize=(n * 4, 8))

        # 计算全局 vmax，使两行色阶一致，便于比较
        global_vmax = 0
        for bpp_label, (decoded_imgs, meta, _) in rows_data.items():
            for codec, dec in decoded_imgs.items():
                h = min(orig.shape[0], dec.shape[0])
                w = min(orig.shape[1], dec.shape[1])
                diff = np.abs(orig[:h, :w].astype(float) - dec.astype(float)).mean(axis=2)
                global_vmax = max(global_vmax, diff.max())

        for row_idx, (bpp_label, (decoded_imgs, meta, target_bpp)) in enumerate(rows_data.items()):
            for col_idx, codec in enumerate(CODECS):
                ax = axes[row_idx][col_idx]
                if codec not in decoded_imgs:
                    ax.axis('off')
                    continue
                dec = decoded_imgs[codec]
                h = min(orig.shape[0], dec.shape[0])
                w = min(orig.shape[1], dec.shape[1])
                diff = np.abs(orig[:h, :w].astype(float) - dec.astype(float)).mean(axis=2)
                im = ax.imshow(diff, cmap='hot', vmin=0, vmax=global_vmax)
                ax.set_title(
                    f'{codec}\nBPP={meta[codec]["bpp"]:.3f}  MaxErr={diff.max():.1f}',
                    fontsize=9, color=COLORS.get(codec, 'black')
                )
                ax.axis('off')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                # 左侧加行标签
                if col_idx == 0:
                    ax.set_ylabel(f'{bpp_label.upper()} BPP\n≈{target_bpp}',
                                  fontsize=10, labelpad=10)
                    ax.axis('on')
                    ax.set_yticks([])
                    ax.set_xticks([])

        plt.suptitle(
            f'{img_name} — Pixel Error Heatmap\n'
            f'Top: Low BPP ≈ {TARGET_BPPS["low"]}  |  Bottom: High BPP ≈ {TARGET_BPPS["high"]}\n'
            f'Brighter = larger distortion  (same color scale)',
            fontsize=12
        )
        plt.tight_layout()
        path = f'{OUTPUT_DIR}/03_{img_name}_diff_heatmap.png'
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")



# # ─────────────────────────────────────────────
# # 图4：三张典型图各自的 RD 曲线（2×3布局）
# # ─────────────────────────────────────────────
# def plot_focal_rd(df):
#     print("Plotting focal image RD curves...")
#     fig, axes = plt.subplots(2, 3, figsize=(15, 8))

#     for col, (img_name, desc) in enumerate(FOCAL_IMAGES.items()):
#         for row, metric in enumerate(['PSNR', 'SSIM']):
#             ax = axes[row][col]
#             sub = df[df['Image'] == img_name]
#             for codec in CODECS:
#                 c_sub = sub[sub['Codec'] == codec].sort_values('BPP')
#                 if c_sub.empty:
#                     continue
#                 ax.plot(c_sub['BPP'], c_sub[metric],
#                         marker=MARKERS[codec], color=COLORS[codec],
#                         label=codec, linewidth=1.8, markersize=5)
#             if row == 0:
#                 ax.set_title(f'{img_name}\n{desc}', fontsize=9)
#             ax.set_xlabel('BPP', fontsize=8)
#             ax.set_ylabel(metric, fontsize=8)
#             ax.grid(True, alpha=0.3)
#             ax.tick_params(labelsize=7)
#             if col == 0:
#                 ax.legend(fontsize=7)

#     plt.suptitle('RD Curves — Three Focal Images (Kodak Dataset)', fontsize=13)
#     plt.tight_layout()
#     path = f'{OUTPUT_DIR}/04_focal_rd_curves.png'
#     plt.savefig(path, dpi=150)
#     plt.close()
#     print(f"  Saved: {path}")





# ─────────────────────────────────────────────
# 图5：编解码时间对比
# ─────────────────────────────────────────────
def plot_timing(df):
    print("Plotting encode/decode timing...")
    if 'EncTime' not in df.columns:
        print("  No EncTime column — re-run benchmark.py to collect timing data.")
        return

    agg = df.groupby(['Codec', 'Param']).agg(
        EncTime=('EncTime', 'mean'),
        DecTime=('DecTime', 'mean'),
        BPP=('BPP', 'mean'),
    ).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, col_name, ylabel in zip(
        axes[:2],
        ['EncTime', 'DecTime'],
        ['Encode Time (s)', 'Decode Time (s)']
    ):
        for codec in CODECS:
            sub = agg[agg['Codec'] == codec].sort_values('BPP')
            ax.plot(sub['BPP'], sub[col_name],
                    marker=MARKERS[codec], color=COLORS[codec],
                    label=codec, linewidth=2, markersize=7)
        ax.set_xlabel('Bits Per Pixel (BPP)')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{ylabel.split("(")[0].strip()} vs BPP', fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.4)

    ax = axes[2]
    mean_times = df.groupby('Codec')[['EncTime', 'DecTime']].mean().reindex(CODECS)
    x = np.arange(len(CODECS))
    width = 0.35
    bars_enc = ax.bar(x - width/2, mean_times['EncTime'], width,
                      label='Encode', color=[COLORS[c] for c in CODECS], alpha=0.85)
    bars_dec = ax.bar(x + width/2, mean_times['DecTime'], width,
                      label='Decode', color=[COLORS[c] for c in CODECS], alpha=0.45,
                      edgecolor=[COLORS[c] for c in CODECS], linewidth=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels(CODECS, fontsize=12)
    ax.set_ylabel('Time (s)')
    ax.set_title('Average Encode / Decode Time', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4, axis='y')
    for bar in list(bars_enc) + list(bars_dec):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.002,
                f'{h:.3f}s', ha='center', va='bottom', fontsize=9)

    plt.suptitle(
        'Encode / Decode Time — JPEG / BPG / JPEG-XL\n'
        '(Kodak Dataset, 24-image average per quality point)',
        fontsize=13
    )
    plt.tight_layout()
    path = f'{OUTPUT_DIR}/05_timing.png'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────
def main():
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: {CSV_PATH} not found.")
        print("Please run benchmark.py first.")
        return

    df = pd.read_csv(CSV_PATH)
    df = df[df['Codec'] != 'JPEG2000']

    print(f"Loaded {len(df)} records")
    print(f"Images: {df['Image'].nunique()}, Codecs: {list(df['Codec'].unique())}")

    plot_visual_comparison(df)   # 图2：三码率视觉对比 + 局部放大
    plot_diff_heatmap(df)        # 图3：像素误差热力图
    

    plot_timing(df)              # 图5：编解码时间对比

    print(f"\nAll visualizations saved to: {OUTPUT_DIR}/")
    print("  01_average_rd_curves.png     — 24-image average RD, BPP-aligned, ±1 std")
    print("  02_*_visual.png              — Visual comparison at 3 bitrates + auto crop")
    print("  03_*_diff_heatmap.png        — Pixel error heatmap at mid BPP")
    print("  04_focal_rd_curves.png       — RD curves for 3 focal images")
    print("  05_psnr_gain_over_jpeg.png   — PSNR gain over JPEG baseline")
    print("  06_timing.png                — Encode / decode time comparison")


if __name__ == "__main__":
    main()