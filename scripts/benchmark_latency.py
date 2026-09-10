
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import time
import json
import platform

import cv2
cv2.setNumThreads(1)
import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(__file__))
from yolo_utils import preprocess, postprocess

IMG_DIR = "data/images_eval"


def get_cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor()


def bench(onnx_path, image_paths, n_warmup=15):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess = ort.InferenceSession(onnx_path, sess_options=so, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    imgs = [cv2.imread(p) for p in image_paths]

    # warmup
    for img in imgs[:n_warmup]:
        tensor, ratio, pad = preprocess(img)
        out = sess.run(None, {input_name: tensor})[0]
        postprocess(out, ratio, pad, img.shape[:2], conf_thres=0.25, iou_thres=0.45)

    full_times = []
    infer_times = []
    for img in imgs:
        t0 = time.perf_counter()
        tensor, ratio, pad = preprocess(img)
        t1 = time.perf_counter()
        out = sess.run(None, {input_name: tensor})[0]
        t2 = time.perf_counter()
        postprocess(out, ratio, pad, img.shape[:2], conf_thres=0.25, iou_thres=0.45)
        t3 = time.perf_counter()
        infer_times.append(t2 - t1)
        full_times.append(t3 - t0)

    def stats(times):
        arr = np.array(times) * 1000.0  # ms
        return {
            "mean_ms": float(arr.mean()),
            "p95_ms": float(np.percentile(arr, 95)),
            "min_ms": float(arr.min()),
            "max_ms": float(arr.max()),
        }

    return {"full_pipeline": stats(full_times), "inference_only": stats(infer_times)}


if __name__ == "__main__":
    onnx_path = sys.argv[1]
    tag = sys.argv[2]
    with open("data/eval_image_ids.txt") as f:
        fnames = [line.strip().split("\t")[1] for line in f]
    image_paths = [os.path.join(IMG_DIR, fn) for fn in fnames]
    result = bench(onnx_path, image_paths)
    result["cpu_model"] = get_cpu_model()
    result["threads"] = 1
    result["n_images"] = len(image_paths)
    with open(f"results/latency_{tag}.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
