#!/usr/bin/env bash
# Runs the entire pipeline in order: setup -> Task 1 -> Task 2 -> Task 3 -> charts.
# Safe to re-run: every step is idempotent (downloads skip existing files, everything
# else just overwrites its own output deterministically).
#
# Does NOT regenerate results/summary_report.docx -- that file has been hand-edited;
# run scripts/make_docx_report.py yourself only if you want to discard those edits
# and rebuild it from scratch.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [ ! -d venv ]; then
  echo "==> Creating venv and installing requirements..."
  python3 -m venv venv
  source venv/bin/activate
  pip install -q --upgrade pip
  pip install -q -r requirements.txt
else
  source venv/bin/activate
fi

echo "==> Task 1: FP32 baseline, then quantize to INT8"
python3 scripts/select_subset.py
python3 scripts/download_images.py data/eval_image_ids.txt data/images_eval
python3 scripts/download_images.py data/calib_image_ids.txt data/images_calib
python3 scripts/export_onnx.py
python3 scripts/evaluate.py models/yolov8n.onnx fp32
python3 scripts/benchmark_latency.py models/yolov8n.onnx fp32
python3 scripts/quantize.py data/images_calib models/yolov8n_int8.onnx
python3 scripts/evaluate.py models/yolov8n_int8.onnx int8
python3 scripts/benchmark_latency.py models/yolov8n_int8.onnx int8

echo "==> Task 2: Degrade"
python3 scripts/degrade.py
for cond in motion_blur low_light jpeg30 downup; do
  python3 scripts/evaluate.py models/yolov8n.onnx fp32_$cond data/images_eval_$cond
  python3 scripts/evaluate.py models/yolov8n_int8.onnx int8_$cond data/images_eval_$cond
done

echo "==> Task 3: one targeted intervention"
python3 scripts/blur_calib_images.py
python3 scripts/quantize.py data/images_calib_motion_blur models/yolov8n_int8_blurcalib.onnx
python3 scripts/evaluate.py models/yolov8n_int8_blurcalib.onnx int8blurcalib_motion_blur data/images_eval_motion_blur
python3 scripts/evaluate.py models/yolov8n_int8_blurcalib.onnx int8blurcalib_clean data/images_eval
python3 scripts/benchmark_latency.py models/yolov8n_int8_blurcalib.onnx int8blurcalib

echo "==> Charts"
python3 scripts/make_figures.py

echo
echo "==> Done. Metrics in results/*.json, latency in results/latency_*.json, charts in results/figures/."
echo "    results/summary_report.docx was NOT touched (it has manual edits) -- rebuild it yourself with"
echo "    scripts/make_docx_report.py only if you want to discard those edits."
