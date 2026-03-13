import os
import glob
import subprocess
import time
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def compute_metrics(orig, decoded):
    h = min(orig.shape[0], decoded.shape[0])
    w = min(orig.shape[1], decoded.shape[1])
    o = orig[:h, :w]
    d = decoded[:h, :w]
    if len(d.shape) == 3 and d.shape[2] == 4:
        d = cv2.cvtColor(d, cv2.COLOR_BGRA2BGR)
    if len(d.shape) == 2:
        d = cv2.cvtColor(d, cv2.COLOR_GRAY2BGR)
    p = psnr(o, d)
    ch = 2 if len(o.shape) == 3 else None
    s = ssim(o, d, channel_axis=ch)
    return p, s


def load_and_normalize(path):
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


def ffmpeg_encode_decode(input_path, out_compressed, out_png, encode_args):
    t0 = time.perf_counter()
    enc = subprocess.run(
        ['ffmpeg', '-y', '-i', input_path] + encode_args + [out_compressed],
        capture_output=True, text=True
    )
    enc_time = time.perf_counter() - t0
    if enc.returncode != 0:
        return False, enc.stderr, enc_time, 0.0

    t0 = time.perf_counter()
    dec = subprocess.run([
        'ffmpeg', '-y', '-i', out_compressed,
        '-frames:v', '1', '-update', '1', out_png
    ], capture_output=True, text=True)
    dec_time = time.perf_counter() - t0

    if dec.returncode != 0:
        return False, dec.stderr, enc_time, dec_time
    return True, "", enc_time, dec_time


def test_single_image(input_img_path, output_dir, codecs):
    img_name = os.path.splitext(os.path.basename(input_img_path))[0]
    img = cv2.imread(input_img_path)
    if img is None:
        print(f"  ERROR: Cannot read {input_img_path}")
        return []

    h, w = img.shape[:2]
    pixel_count = h * w
    results = []

    for codec in codecs:
        name = codec['name']
        ext  = codec['ext']

        for q in codec['qualities']:
            out_file = f"{output_dir}/{img_name}_{name.lower()}_{codec['label']}{q}.{ext}"
            out_png  = f"{output_dir}/{img_name}_dec_{name.lower()}_{codec['label']}{q}.png"
            enc_time = 0.0
            dec_time = 0.0

            if name == 'JPEG':
                t0 = time.perf_counter()
                cv2.imencode('.jpg', img,
                             [int(cv2.IMWRITE_JPEG_QUALITY), q])[1].tofile(out_file)
                enc_time = time.perf_counter() - t0

                t0 = time.perf_counter()
                decoded = cv2.imdecode(np.fromfile(out_file, dtype=np.uint8), 1)
                dec_time = time.perf_counter() - t0
                cv2.imwrite(out_png, decoded)

            elif name == 'BPG':
                ok, err, enc_time, dec_time = ffmpeg_encode_decode(
                    input_img_path, out_file, out_png,
                    ['-c:v', 'libx265',
                     '-crf', str(q),
                     '-pix_fmt', 'yuv444p',
                     '-x265-params', 'keyint=1:deblock=-1,-1']
                )
                decoded = cv2.imread(out_png) if ok else None

            elif name == 'JPEG-XL':
                t0 = time.perf_counter()
                enc = subprocess.run([
                    CJXL_BIN, input_img_path, out_file,
                    '-d', str(q), '--lossless_jpeg=0'
                ], capture_output=True, text=True)
                enc_time = time.perf_counter() - t0
                if enc.returncode != 0:
                    decoded = None
                else:
                    t0 = time.perf_counter()
                    dec = subprocess.run([
                        DJXL_BIN, out_file, out_png
                    ], capture_output=True, text=True)
                    dec_time = time.perf_counter() - t0
                    decoded = load_and_normalize(out_png) if dec.returncode == 0 else None

            else:
                continue

            if decoded is not None:
                bpp = (os.path.getsize(out_file) * 8) / pixel_count
                p, s = compute_metrics(img, decoded)
                results.append({
                    'Image':    img_name,
                    'Codec':    name,
                    'Param':    q,
                    'BPP':      bpp,
                    'PSNR':     p,
                    'SSIM':     s,
                    'EncTime':  enc_time,
                    'DecTime':  dec_time,
                })

    return results


CJXL_BIN = '/WAVE/users2/unix/hliao/libjxl/build/tools/cjxl'
DJXL_BIN = '/WAVE/users2/unix/hliao/libjxl/build/tools/djxl'
COLORS  = {'JPEG': 'tomato', 'BPG': 'steelblue', 'JPEG-XL': 'mediumpurple'}
MARKERS = {'JPEG': 'o',      'BPG': 'D',          'JPEG-XL': '*'}
CODECS  = ['JPEG', 'BPG', 'JPEG-XL']


def plot_average_rd(df, output_dir, n_images):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, metric in zip(axes, ['PSNR', 'SSIM']):
        for c in CODECS:
            sub = df[df['Codec'] == c]
            if sub.empty:
                continue
            grouped = sub.groupby('Param').agg(
                BPP=('BPP', 'mean'),
                Score=(metric, 'mean'),
            ).reset_index().sort_values('BPP')
            ax.plot(grouped['BPP'], grouped['Score'],
                    marker=MARKERS[c], color=COLORS[c],
                    label=c, linewidth=2, markersize=7)
        ax.set_title(f'Average {metric} vs BPP', fontsize=13)
        ax.set_xlabel('Bits Per Pixel (BPP)')
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid(True, alpha=0.4)
    plt.suptitle(
        f'Average RD Curve — JPEG / BPG / JPEG-XL\n'
        f'(Kodak Dataset, {n_images} images)', fontsize=13
    )
    plt.tight_layout()
    path = f'{output_dir}/kodak_average_rd_curves.png'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Average RD curve saved: {path}")


def run_benchmark():
    kodak_dir  = 'data/raw'
    output_dir = 'results/kodak'
    os.makedirs(output_dir, exist_ok=True)

    image_paths = sorted(glob.glob(f'{kodak_dir}/kodim*.png'))
    if not image_paths:
        print(f"ERROR: No images found in {kodak_dir}")
        return

    print(f"Found {len(image_paths)} images in Kodak dataset")

    codecs = [
        {'name': 'JPEG',    'ext': 'jpg', 'qualities': [10, 30, 50, 70, 80, 90, 95],                   'label': 'q'},
        {'name': 'BPG',     'ext': 'mkv', 'qualities': [40, 35, 31, 27, 23, 19, 15],                  'label': 'crf'},
        {'name': 'JPEG-XL', 'ext': 'jxl', 'qualities': [5, 4.5, 3.5, 2.8, 1.7, 1.0, 0.4],    'label': 'd'},
    ]

    all_results = []
    n = len(image_paths)

    for i, img_path in enumerate(image_paths, 1):
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        print(f"\n{'='*50}")
        print(f"[{i}/{n}] Processing {img_name} ...")
        print(f"{'='*50}")
        results = test_single_image(img_path, output_dir, codecs)
        all_results.extend(results)
        pd.DataFrame(all_results).to_csv(
            f'{output_dir}/kodak_metrics_partial.csv', index=False)

    df = pd.DataFrame(all_results)
    df.to_csv(f'{output_dir}/kodak_metrics_all.csv', index=False)
    print(f"\nAll metrics saved: {output_dir}/kodak_metrics_all.csv")

    for img_name in df['Image'].unique():
        sub = df[df['Image'] == img_name]
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, metric in zip(axes, ['PSNR', 'SSIM']):
            for c in CODECS:
                s = sub[sub['Codec'] == c].sort_values('BPP')
                if s.empty:
                    continue
                ax.plot(s['BPP'], s[metric],
                        marker=MARKERS[c], color=COLORS[c],
                        label=c, linewidth=2, markersize=6)
            ax.set_title(f'{metric} vs BPP')
            ax.set_xlabel('BPP')
            ax.set_ylabel(metric)
            ax.legend()
            ax.grid(True, alpha=0.4)
        plt.suptitle(f'RD Curve — JPEG / BPG / JPEG-XL ({img_name})', fontsize=13)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{img_name}_rd_curves.png', dpi=120)
        plt.close()

    plot_average_rd(df, output_dir, n_images=len(image_paths))
    print("\nDONE!")


if __name__ == "__main__":
    run_benchmark()
