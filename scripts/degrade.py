import os
import sys
import cv2
import numpy as np

IMG_DIR = "data/images_eval"
OUT_DIRS = {
    "motion_blur": "data/images_eval_motion_blur",
    "low_light": "data/images_eval_low_light",
    "jpeg30": "data/images_eval_jpeg30",
    "downup": "data/images_eval_downup",
}

# --- Motion blur: 15x15 horizontal linear motion-blur kernel ---
MOTION_KSIZE = 15
def make_motion_kernel(ksize=MOTION_KSIZE):
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    kernel[(ksize - 1) // 2, :] = 1.0
    kernel /= ksize
    return kernel
MOTION_KERNEL = make_motion_kernel()

def motion_blur(img):
    return cv2.filter2D(img, -1, MOTION_KERNEL, borderType=cv2.BORDER_REFLECT101)

# --- Low light: gamma correction, darkens image ---
GAMMA = 2.5
_gamma_lut = np.array([((i / 255.0) ** GAMMA) * 255.0 for i in range(256)], dtype=np.uint8)
def low_light(img):
    return cv2.LUT(img, _gamma_lut)

# --- JPEG compression at quality 30 ---
JPEG_QUALITY = 30
def jpeg_compress(img):
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    assert ok
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)

# --- Downscale to 50% (INTER_AREA) then upscale back to original size (INTER_LINEAR) ---
def down_up(img):
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_AREA)
    back = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return back


TRANSFORMS = {
    "motion_blur": motion_blur,
    "low_light": low_light,
    "jpeg30": jpeg_compress,
    "downup": down_up,
}


def main():
    with open("data/eval_image_ids.txt") as f:
        fnames = [line.strip().split("\t")[1] for line in f]

    for name, out_dir in OUT_DIRS.items():
        os.makedirs(out_dir, exist_ok=True)
        fn = TRANSFORMS[name]
        n_ok = 0
        for fname in fnames:
            src = os.path.join(IMG_DIR, fname)
            dst = os.path.join(out_dir, fname)
            img = cv2.imread(src)
            if img is None:
                print("WARN could not read", src)
                continue
            out = fn(img)
            cv2.imwrite(dst, out)
            n_ok += 1
        print(f"{name}: wrote {n_ok}/{len(fnames)} images to {out_dir}")


if __name__ == "__main__":
    main()
