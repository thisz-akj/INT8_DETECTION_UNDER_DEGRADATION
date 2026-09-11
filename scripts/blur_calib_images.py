"""Task 3: build a motion-blurred copy of the 120 calibration images, using the
same 15x15 kernel from degrade.py, for the blur-recalibrated INT8 intervention."""
import os
import sys
import cv2

sys.path.insert(0, os.path.dirname(__file__))
from degrade import motion_blur

IN_DIR = "data/images_calib"
OUT_DIR = "data/images_calib_motion_blur"


def main():
    with open("data/calib_image_ids.txt") as f:
        fnames = [line.strip().split("\t")[1] for line in f]

    os.makedirs(OUT_DIR, exist_ok=True)
    n_ok = 0
    for fname in fnames:
        img = cv2.imread(os.path.join(IN_DIR, fname))
        if img is None:
            print("WARN could not read", fname)
            continue
        cv2.imwrite(os.path.join(OUT_DIR, fname), motion_blur(img))
        n_ok += 1
    print(f"wrote {n_ok}/{len(fnames)} blurred calibration images to {OUT_DIR}")


if __name__ == "__main__":
    main()
