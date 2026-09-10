# YOLOv8n INT8 Quantization — FP32 vs INT8 on COCO val2017 (5-class, 500-image subset)

Three tasks: **Task 1** — FP32 baseline, quantize to INT8, compare accuracy/latency/size. **Task 2** — Degrade: four degraded copies of the eval set, both models across all four, the 5×2 mAP@0.5 table. **Task 3** — one targeted intervention on the worst degradation.

## Task 1: FP32 baseline, then quantize to INT8

## Setup

- **Model**: YOLOv8n (Ultralytics), pretrained COCO weights, exported to ONNX (opset 13, fixed 640x640 input, no NMS baked into the graph).
- **Quantization**: Post-training static quantization via ONNX Runtime (`onnxruntime.quantization.quantize_static`), QDQ format, U8S8 (activations `QUInt8`, weights `QInt8` per-channel) — ORT's recommended combo for x86-64 CPU integer kernels.
  - Quantization restricted to `Conv` ops only. Quantizing the detection head's final `Concat` (which merges box coordinates, range ~0–640, with class-score sigmoids, range 0–1, under one shared tensor scale) destroyed all class scores; Conv holds effectively all the FLOPs anyway, so the head is left in float.
  - Calibration set: 120 COCO val2017 images, disjoint from the 500-image eval set, same 5-class pool.
- **Eval set**: 500 images sampled (seed=42) from COCO val2017 images containing ≥1 instance of `person`, `car`, `bicycle`, `traffic light`, `stop sign`. Ground truth restricted to these 5 categories (2,353 annotations). Both models evaluated with identical pre/post-processing (letterbox to 640, conf=0.001, IoU=0.7 NMS) via `pycocotools.COCOeval`.
- **CPU**: 12th Gen Intel(R) Core(TM) i5-1245U. Latency measured single-threaded (`intra_op_num_threads=1`, process pinned to one core via `taskset -c 0`).

## Accuracy: mAP@0.5 and mAP@0.5:0.95, FP32 vs INT8

| Metric | FP32 | INT8 | Δ | Δ% |
|---|---|---|---|---|
| mAP@0.5:0.95 | 0.4147 | 0.3944 | −0.0203 | −4.9% |
| mAP@0.5 | 0.5862 | 0.5688 | −0.0174 | −3.0% |

## Per-class AP delta

| Class | AP@.5:.95 FP32 | AP@.5:.95 INT8 | Δ | AP@.5 FP32 | AP@.5 INT8 | Δ |
|---|---|---|---|---|---|---|
| person | 0.5291 | 0.5111 | −0.0180 | 0.7751 | 0.7630 | −0.0121 |
| bicycle | 0.2446 | 0.2356 | −0.0091 | 0.4081 | 0.3885 | −0.0196 |
| car | 0.3797 | 0.3704 | −0.0093 | 0.5937 | 0.5879 | −0.0058 |
| traffic light | 0.2326 | 0.2138 | −0.0189 | 0.4361 | 0.4284 | −0.0078 |
| stop sign | 0.6872 | 0.6411 | −0.0461 | 0.7178 | 0.6761 | −0.0417 |

`stop sign` has only 15 ground-truth instances in this subset, so its AP is high-variance — treat that delta as noisy rather than a robust class-level signal. `person`, `bicycle`, `car`, `traffic light` are all more sample-rich and show a consistent, modest AP@.5:.95 drop of 1–2 points.

## AP by object size (small / medium / large)

| Size | FP32 | INT8 | Δ | Δ% |
|---|---|---|---|---|
| small (area < 32²) | 0.2045 | 0.1658 | −0.0387 | **−18.9%** |
| medium (32²–96²) | 0.5978 | 0.5730 | −0.0248 | −4.1% |
| large (area > 96²) | 0.7186 | 0.7518 | +0.0331 | +4.6% |

Quantization error hits small objects hardest by far (relative terms) — expected, since small-object detection lives on fine-grained activation magnitudes that are most sensitive to the int8 dynamic range. The large-object AP increase is within normal COCOeval noise at this sample size (few hundred large instances) and shouldn't be read as INT8 "improving" large-object detection.

## CPU latency, single-threaded (CPU: 12th Gen Intel Core i5-1245U)

| | FP32 mean | FP32 p95 | INT8 mean | INT8 p95 | Speedup (mean) |
|---|---|---|---|---|---|
| Full pipeline (preprocess+infer+postprocess) | 134.92 ms | 138.56 ms | 69.72 ms | 71.97 ms | 1.94x |
| Inference only | 128.99 ms | 132.43 ms | 64.41 ms | 66.62 ms | 2.00x |

(500 images, 15-image warmup excluded, single core pinned via `taskset -c 0`, `intra_op_num_threads=1`.)

## Model size on disk

| Model | Size |
|---|---|
| FP32 ONNX | 12.85 MB (12,850,959 bytes) |
| INT8 ONNX | 3.60 MB (3,598,268 bytes) |
| Ratio | 3.57x smaller |

## Bottom line

INT8 static quantization gives **~2x CPU inference speedup** and **~3.6x smaller model** for a **~2–5% relative mAP loss** (0.5–2 points absolute on mAP@0.5/mAP@0.5:0.95), heavily concentrated in small-object AP (−19% relative). For deployment on CPU where small-object recall isn't critical, this is a strong trade. If small-object accuracy matters, consider: per-channel quantization is already on; next steps would be QAT, mixed-precision (keep the first/last few Conv layers in FP32), or a higher-resolution input to recover small-object AP.

## Task 2: Degrade

Four degraded copies of the same 500 images were built (dimensions preserved, so the same ground-truth boxes apply unchanged to every condition):

| Degradation | Definition |
|---|---|
| Motion blur | 15×15 horizontal linear motion-blur kernel (`cv2.filter2D`, kernel is a normalized single-row-of-ones 15×15 matrix — averages 15 pixels along a horizontal line) |
| Low light | Gamma correction, γ=2.5: `out = 255 * (in/255)^γ` (darkens midtones/shadows nonlinearly) |
| JPEG compression | Re-encoded at quality 30 (`cv2.IMWRITE_JPEG_QUALITY=30`) |
| Downscale/upscale | Resize to 50% (`INTER_AREA`) then back to original size (`INTER_LINEAR`) — simulates low sensor resolution / upsampled low-res source |

Sanity-checked before running: motion blur cuts Laplacian sharpness variance by ~87%, low-light drops mean pixel brightness by ~60%, JPEG30 shrinks file size ~3.3x, downscale/upscale cuts sharpness variance ~94% — all in the expected direction.

### 5×2 table — mAP@0.5, FP32 vs INT8, across conditions

| Condition | FP32 mAP@0.5 | INT8 mAP@0.5 | Δ (INT8−FP32) |
|---|---|---|---|
| Clean (no degradation) | 0.5862 | 0.5688 | −0.0174 |
| Motion blur | 0.2807 | 0.2502 | −0.0305 |
| Low light (γ=2.5) | 0.5155 | 0.5124 | −0.0031 |
| JPEG quality 30 | 0.5337 | 0.5145 | −0.0192 |
| Downscale 50% → upscale | 0.5426 | 0.5167 | −0.0259 |

(mAP@0.5:0.95 for the same 10 runs is in `results/metrics_{fp32,int8}_{condition}.json` — motion blur: 0.1876/0.1695, low light: 0.3740/0.3590, jpeg30: 0.3672/0.3544, downup: 0.3887/0.3599.)

### Relative drop from each model's own clean baseline

| Condition | FP32 drop | INT8 drop | INT8 vs FP32 gap |
|---|---|---|---|
| Motion blur | −52.1% | −56.0% | INT8 loses 3.9pp more |
| Low light | −12.1% | −9.9% | INT8 loses 2.1pp *less* |
| JPEG30 | −8.9% | −9.5% | INT8 loses 0.6pp more |
| Downscale/upscale | −7.4% | −9.2% | INT8 loses 1.7pp more |

**Key findings:**
- **Motion blur is by far the dominant failure mode for both models** — over 50% relative mAP loss, an order of magnitude larger than any other degradation. This is a property of the input signal (high-frequency edge information the detector relies on is destroyed by the blur), not of quantization.
- **INT8's *additional* sensitivity to degradation (beyond its own clean-image gap) is degradation-dependent, not uniform** — see the deep-dive below: it widens sharply under motion blur, mildly under downscale/upscale, stays flat under JPEG, and actually narrows under low light.
- Practically: if you're deploying this detector where motion blur is likely (fast-moving camera or subject), that dwarfs anything quantization costs you — worth addressing at the source (shutter speed, deblur preprocessing) rather than expecting INT8/FP32 choice to matter much here.

### Deep dive: does INT8 lose *more* than FP32 under degradation than it does on clean images?

The right way to ask this is: does the INT8-vs-FP32 gap, **as a fraction of FP32's own score in that condition**, grow relative to the clean-image gap? (Comparing absolute-point drops is misleading once FP32 itself collapses under a degradation like motion blur.)

| Condition | FP32 mAP@0.5 | INT8 mAP@0.5 | Relative INT8 gap (= (INT8−FP32)/FP32) |
|---|---|---|---|
| Clean | 0.5862 | 0.5688 | −3.0% |
| Low light | 0.5155 | 0.5124 | **−0.6%** (narrower than clean) |
| JPEG q30 | 0.5337 | 0.5145 | −3.6% (~same as clean) |
| Downscale/upscale | 0.5426 | 0.5167 | −4.8% (wider than clean) |
| Motion blur | 0.2807 | 0.2502 | **−10.9%** (much wider than clean) |

**Answer: it depends on the degradation.** Yes for motion blur (clearly), mildly yes for downscale/upscale, no change for JPEG, and no (INT8 actually degrades *less*) for low light.

**Why — traced to the object-size level.** Breaking AP@[.5:.95] out by object size for motion blur:

| Size | FP32 | INT8 | Relative gap |
|---|---|---|---|
| Small | 0.0223 | 0.0257 | +15% (both ≈0 — floor effect, not a real trend, see caveat) |
| **Medium** | **0.2586** | **0.1788** | **−30.9%** |
| Large | 0.4547 | 0.4381 | −3.7% |

The entire motion-blur INT8 penalty concentrates in **medium-sized objects**. Small objects collapse to near-zero AP for *both* precisions under blur (there's almost no usable signal left for either model), and large objects retain enough edge information that quantization barely matters. It's the medium bucket — where FP32 still has exploitable signal — where INT8 can no longer resolve it.

Per-class, medium-size AP under motion blur:

| Class (medium-size) | FP32 AP | INT8 AP | Relative gap |
|---|---|---|---|
| person | 0.297 | 0.283 | −4.6% |
| car | 0.240 | 0.192 | −20.3% |
| traffic light | 0.218 | 0.143 | −34.3% |

`person` (detected largely from holistic shape/texture/color) barely widens. `car` and `traffic light` (detected from crisp geometric edges — wheel arches, window lines, a rigid sign silhouette) widen sharply. Blur destroys exactly the fine-edge information these classes depend on; the residual signal is weaker, and INT8's coarser numeric resolution — with quantization ranges calibrated on clean, unblurred calibration images — can no longer discriminate it, while FP32's full precision still can.

**Why low light doesn't show this**: gamma correction is a monotonic, pixel-wise brightness remap. It doesn't mix neighboring pixels or destroy edges — everything just gets darker, edges stay exactly as crisp. That's a far gentler distribution shift than blur or resampling, so it doesn't clash with INT8's fixed quantization ranges the way spatially-destructive degradations do.

**General pattern**: the size of INT8's extra penalty tracks how much *fine spatial/edge information* a degradation destroys, not how much it shifts overall pixel statistics — motion blur (large 15px kernel, destroys edges directly) ≫ downscale/upscale (smaller effective blur radius) > JPEG (localized block artifacts) > low light (no spatial information loss at all).

*Caveat*: the small-object motion-blur numbers (0.022 vs 0.026 AP) are statistically unreliable — at that AP floor both models are essentially failing, so a percentage difference between two near-zero numbers isn't a meaningful trend and was excluded from the argument above.

## Task 3: one targeted intervention on the worst degradation

**Worst degradation**: motion blur — by far the largest relative INT8 gap (clean: −3.0% vs FP32; motion blur: **−10.9%**), and the diagnosis in Part 2 pointed at a specific, testable cause: the INT8 quantization scales were computed once via MinMax calibration on 120 **clean** images, then applied unchanged to blurred inputs at inference — a calibration/deployment distribution mismatch.

**Intervention** (one change only, everything else — architecture, op selection, QDQ format, U8S8, per-channel, opset — held fixed): re-run the identical `quantize_static` call, swapping only the 120 calibration images for the same 120 images with the same 15×15 motion-blur kernel applied. This directly tests the diagnosed cause without confounding it with an architectural change (e.g. mixed precision would also change latency/size, making it hard to attribute any effect to calibration data specifically).

### Result on the target condition (motion-blurred images) — it did not work

| Model | mAP@0.5 | mAP@0.5:0.95 | AP_small | AP_medium | AP_large |
|---|---|---|---|---|---|
| FP32 | 0.2807 | 0.1876 | 0.0223 | 0.2586 | 0.4547 |
| INT8, clean-calibrated (original) | 0.2502 | 0.1695 | 0.0257 | 0.1788 | 0.4381 |
| INT8, **blur-calibrated (new)** | 0.2496 | 0.1702 | 0.0201 | **0.1958** | 0.4203 |

Net change vs the original INT8 model, on the motion-blurred eval set: mAP@0.5 **−0.0006** (noise-level), mAP@0.5:0.95 **+0.0007** (noise-level). AP_medium — the exact bucket Part 2 identified as where the INT8 penalty concentrated — did improve (+0.017, +9.5% relative), but AP_large got worse (−0.018) and AP_small got worse (−0.006), netting out to essentially zero overall improvement. **The intervention failed at its stated goal**: the INT8-vs-FP32 gap under motion blur is unchanged (relative gap −11.1% vs the original −10.9%).

### An unexpected side effect: it improved clean-image accuracy instead

The same blur-calibrated model, evaluated on the **clean** (non-degraded) images it was never targeting:

| Model | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|
| FP32 | 0.5862 | 0.4147 |
| INT8, clean-calibrated (original) | 0.5688 | 0.3944 |
| INT8, blur-calibrated (new) | **0.5824** | **0.3999** |

This closes about 78% of the original clean-image INT8 gap (0.0174 → 0.0038 absolute), moving in the *opposite* direction from what was intended.

### Why — verified directly against the calibrated quantization scales, not just guessed

Pulled the actual per-tensor activation `_scale` constants out of both ONNX graphs (124 comparable tensors): blur-calibration produced a **smaller (tighter) scale in 95.2%** of them, averaging **21% tighter** (mean ratio 0.79, e.g. `/model.1/conv/Conv_output_0_scale` 1.325 → 0.949, `/model.3/conv/Conv_output_0_scale` 0.106 → 0.067).

The mechanism: MinMax calibration sets the quantization range from the observed min/max activations in the calibration set. Motion-blurred images suppress the sharp, high-contrast peak activations that clean images produce at hard edges — so calibrating on blurred images yields a **narrower dynamic range**, hence **finer per-step resolution** across the 8-bit budget. That finer resolution is a net win wherever the *typical* (non-extreme) activation magnitudes dominate the accuracy — which turned out to be the clean-image distribution — but it does nothing for the blurred target distribution, because the accuracy lost to blur is a genuine information loss from the 15×15 spatial averaging, not a quantization-resolution problem; extra numeric precision applied to a signal that's already gone doesn't recover it. It even slightly hurts blurred large objects (0.4381 → 0.4203), consistent with the now-tighter range clipping the occasional still-large activation that survives blurring in bigger objects.

**Lesson**: "calibrate on the data you expect to deploy on" is not the operative variable here — the tightness of the resulting MinMax range is. This calibration-set swap changed *where* 8-bit resolution gets spent, and that reallocation happened to favor the clean distribution, not the blurred one it was aimed at.

### Cost

- **Model size**: 3,598,268 → 3,598,274 bytes (+6 bytes — identical graph, only scale/zero-point constants differ).
- **Latency**: benchmarked back-to-back on the same machine state to avoid conflating with earlier thermal/frequency drift — original-calib INT8: 101.40 ms mean / 93.71 ms inference-only; blur-calib INT8: 101.62 ms mean / 93.93 ms inference-only. A <0.3% difference, within run-to-run noise, as expected since op selection, format, and per-channel settings are unchanged — only calibration constants differ.

**Verdict**: free in latency and size, but it does not fix the diagnosed problem. A calibration-data swap changes *how* the fixed 8-bit budget is allocated across the activation range, not how much information the degradation destroyed — and for motion blur specifically, the bottleneck is the latter. A more promising next intervention (not attempted here, to keep this to one change) would be to leave calibration alone and instead run mixed precision — keep `model.1`/`model.2` (the earliest backbone stages, which showed the largest scale shrinkage above and feed every downstream layer) in FP32 — since that targets the layers doing the initial edge/gradient extraction that blur most directly degrades, rather than reshuffling quantization resolution after the fact.

## Reproducing

```
scripts/select_subset.py        # picks 500 eval + 120 calib images from instances_val2017.json
scripts/download_images.py      # fetches images by file_name from images.cocodataset.org
scripts/export_onnx.py          # yolov8n.pt -> models/yolov8n.onnx (FP32)
scripts/quantize.py <calib_dir> <out.onnx>   # post-training static INT8 quantization
scripts/evaluate.py <model> <tag> [img_dir]  # mAP via pycocotools COCOeval, defaults to data/images_eval
scripts/benchmark_latency.py <model> <tag>   # single-threaded latency, mean/p95
scripts/degrade.py               # builds the 4 degraded copies under data/images_eval_{motion_blur,low_light,jpeg30,downup}
scripts/make_figures.py          # regenerates the README's chart images from results/*.json
```
