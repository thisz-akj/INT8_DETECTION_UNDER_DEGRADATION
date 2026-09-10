import numpy as np
import cv2

# COCO-80 class names in the order YOLOv8 (ultralytics) outputs them (index 0..79)
COCO80_NAMES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
    "toothbrush",
]

IMG_SIZE = 640


def letterbox(img, new_shape=IMG_SIZE, color=(114, 114, 114)):
    h0, w0 = img.shape[:2]
    r = min(new_shape / h0, new_shape / w0)
    new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
    dw, dh = new_shape - new_unpad[0], new_shape - new_unpad[1]
    dw /= 2
    dh /= 2
    if (w0, h0) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (left, top)


def preprocess(img_bgr):
    img, ratio, pad = letterbox(img_bgr, IMG_SIZE)
    img = img[:, :, ::-1].astype(np.float32) / 255.0  # BGR->RGB, normalize
    img = img.transpose(2, 0, 1)[None]  # NCHW
    img = np.ascontiguousarray(img)
    return img, ratio, pad


def xywh2xyxy(x):
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def postprocess(output, ratio, pad, orig_shape, conf_thres=0.001, iou_thres=0.7, max_det=300):
    """output: raw model output, shape (1, 84, 8400) -> boxes in original image coords.
    Returns list of dicts: {bbox:[x1,y1,x2,y2], score, cls_idx}
    """
    pred = output[0]  # (84, 8400)
    pred = pred.transpose(1, 0)  # (8400, 84)
    boxes_xywh = pred[:, :4]
    cls_scores = pred[:, 4:]
    cls_idx = cls_scores.argmax(axis=1)
    scores = cls_scores[np.arange(cls_scores.shape[0]), cls_idx]

    keep = scores > conf_thres
    if not np.any(keep):
        return []
    boxes_xywh = boxes_xywh[keep]
    scores = scores[keep]
    cls_idx = cls_idx[keep]

    boxes_xyxy = xywh2xyxy(boxes_xywh)

    # undo letterbox padding/scale -> original image coords
    left, top = pad
    boxes_xyxy[:, [0, 2]] -= left
    boxes_xyxy[:, [1, 3]] -= top
    boxes_xyxy /= ratio
    h0, w0 = orig_shape
    boxes_xyxy[:, [0, 2]] = boxes_xyxy[:, [0, 2]].clip(0, w0)
    boxes_xyxy[:, [1, 3]] = boxes_xyxy[:, [1, 3]].clip(0, h0)

    # class-wise NMS via cv2 (offset trick to avoid cross-class suppression)
    boxes_for_nms = boxes_xyxy.copy()
    max_coord = boxes_for_nms.max() if boxes_for_nms.size else 0
    offsets = cls_idx.astype(np.float32)[:, None] * (max_coord + 1)
    boxes_nms = boxes_for_nms + offsets
    boxes_wh = boxes_nms.copy()
    boxes_wh[:, 2] -= boxes_wh[:, 0]
    boxes_wh[:, 3] -= boxes_wh[:, 1]
    idxs = cv2.dnn.NMSBoxes(boxes_wh.tolist(), scores.tolist(), conf_thres, iou_thres)
    if len(idxs) == 0:
        return []
    idxs = np.array(idxs).reshape(-1)
    idxs = idxs[np.argsort(-scores[idxs])][:max_det]

    results = []
    for i in idxs:
        results.append({
            "bbox": boxes_xyxy[i].tolist(),
            "score": float(scores[i]),
            "cls_idx": int(cls_idx[i]),
        })
    return results
