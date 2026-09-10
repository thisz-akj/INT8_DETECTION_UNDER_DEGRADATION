
import glob
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import cv2
import numpy as np
from onnxruntime.quantization import CalibrationDataReader, QuantType, QuantFormat, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process

from yolo_utils import preprocess

CALIB_DIR = "data/images_calib"
FP32_MODEL = "models/yolov8n.onnx"
PREPROCESSED_MODEL = "models/yolov8n_preproc.onnx"
INT8_MODEL = "models/yolov8n_int8.onnx"


class YoloCalibrationReader(CalibrationDataReader):
    def __init__(self, image_dir):
        self.paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
        print(f"Calibration images found: {len(self.paths)}")
        self._iter = iter(self.paths)

    def get_next(self):
        p = next(self._iter, None)
        if p is None:
            return None
        img = cv2.imread(p)
        tensor, _, _ = preprocess(img)
        return {"images": tensor.astype(np.float32)}


if __name__ == "__main__":
    calib_dir = sys.argv[1] if len(sys.argv) > 1 else CALIB_DIR
    int8_out = sys.argv[2] if len(sys.argv) > 2 else INT8_MODEL

    if not os.path.exists(PREPROCESSED_MODEL):
        print("Running shape-inference pre-processing...")
        quant_pre_process(FP32_MODEL, PREPROCESSED_MODEL, skip_symbolic_shape=False)

    reader = YoloCalibrationReader(calib_dir)
    quantize_static(
        model_input=PREPROCESSED_MODEL,
        model_output=int8_out,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        # U8S8 (uint8 activations, int8 weights) is ORT's recommended combo for
        # x86-64 CPU: it maps onto the fast oneDNN/MLAS integer GEMM kernels.
        # QInt8/QInt8 (S8S8) fell back to a slower path and was measured slower
        # than FP32 on this CPU.
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
        # Restrict to Conv: quantizing the detection-head Concat (which merges
        # box coords, range ~0-640, with class-score sigmoids, range 0-1) under
        # one shared per-tensor scale crushes the class scores to zero. Conv
        # layers hold nearly all the FLOPs anyway, so this keeps the speed
        # benefit while leaving the numerically sensitive head in float.
        op_types_to_quantize=["Conv"],
    )
    print("Wrote", int8_out, "(calibrated on", calib_dir, ")")
