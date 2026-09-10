import json
import os
import sys
import io
import contextlib

import cv2
import numpy as np
import onnxruntime as ort
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.insert(0, os.path.dirname(__file__))
from yolo_utils import preprocess, postprocess, COCO80_NAMES

SUBSET_JSON = "data/annotations/instances_val2017_subset500.json"
IMG_DIR = "data/images_eval"


def run_model(onnx_path, results_path, conf_thres=0.001, iou_thres=0.7, threads=1, img_dir=IMG_DIR):
    coco = COCO(SUBSET_JSON)
    cat_ids = coco.getCatIds()
    cats = coco.loadCats(cat_ids)
    name_to_id = {c["name"]: c["id"] for c in cats}
    target_names = set(name_to_id.keys())
    img_infos = coco.loadImgs(coco.getImgIds())

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = threads
    sess = ort.InferenceSession(onnx_path, sess_options=so, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    results = []
    for info in img_infos:
        img_path = os.path.join(img_dir, info["file_name"])
        img = cv2.imread(img_path)
        if img is None:
            print("WARNING: could not read", img_path)
            continue
        h0, w0 = img.shape[:2]
        tensor, ratio, pad = preprocess(img)
        out = sess.run(None, {input_name: tensor})[0]
        dets = postprocess(out, ratio, pad, (h0, w0), conf_thres=conf_thres, iou_thres=iou_thres)
        for d in dets:
            name = COCO80_NAMES[d["cls_idx"]]
            if name not in target_names:
                continue
            x1, y1, x2, y2 = d["bbox"]
            results.append({
                "image_id": info["id"],
                "category_id": name_to_id[name],
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": d["score"],
            })

    with open(results_path, "w") as f:
        json.dump(results, f)
    print(f"Wrote {len(results)} detections to {results_path}")
    return coco, cat_ids, results_path


def summarize_coco(coco, cat_ids, results_path):
    coco_dt = coco.loadRes(results_path)
    ev = COCOeval(coco, coco_dt, iouType="bbox")
    ev.params.catIds = cat_ids
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    overall = {
        "AP@[.5:.95]": ev.stats[0], "AP@.5": ev.stats[1], "AP@.75": ev.stats[2],
        "AP_small": ev.stats[3], "AP_medium": ev.stats[4], "AP_large": ev.stats[5],
        "AR@1": ev.stats[6], "AR@10": ev.stats[7], "AR@100": ev.stats[8],
        "AR_small": ev.stats[9], "AR_medium": ev.stats[10], "AR_large": ev.stats[11],
    }

    per_class = {}
    cats = coco.loadCats(cat_ids)
    for c in cats:
        ev_c = COCOeval(coco, coco_dt, iouType="bbox")
        ev_c.params.catIds = [c["id"]]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ev_c.evaluate()
            ev_c.accumulate()
            ev_c.summarize()
        per_class[c["name"]] = {
            "AP@[.5:.95]": ev_c.stats[0], "AP@.5": ev_c.stats[1],
            "AP_small": ev_c.stats[3], "AP_medium": ev_c.stats[4], "AP_large": ev_c.stats[5],
        }
    return overall, per_class


if __name__ == "__main__":
    onnx_path = sys.argv[1]
    tag = sys.argv[2]  # e.g. fp32 / int8, or fp32_motion_blur etc.
    img_dir = sys.argv[3] if len(sys.argv) > 3 else IMG_DIR
    results_path = f"results/dets_{tag}.json"
    coco, cat_ids, rp = run_model(onnx_path, results_path, img_dir=img_dir)
    overall, per_class = summarize_coco(coco, cat_ids, rp)
    out = {"model": onnx_path, "img_dir": img_dir, "overall": overall, "per_class": per_class}
    with open(f"results/metrics_{tag}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
