# Shelf Count — Project Overview

## Vision

A mobile tool for retail shelf auditing. A store associate photographs a shelf, and the system counts products, identifies gaps, and reports shelf status.

## Problem

Manual shelf counts are slow, error-prone, and tedious. Associates walk aisles with clipboards or handheld scanners, counting items one by one. This takes hours and the data is stale by the time it's compiled.

## Solution

Point a phone camera at a shelf → get instant product counts and out-of-stock detection.

## ML Model

All prototypes use the same model for consistent comparison:

- **Model:** YOLOv8m fine-tuned on SKU-110K (by albertferre/shelf-product-identifier)
- **Format:** TFLite float32 (~99MB) for on-device, ONNX for server
- **Input:** `[1, 640, 640, 3]` float32 NHWC, normalized 0–1
- **Output:** `[1, 5, 8400]` float32 — 8400 candidate boxes × 5 values (cx, cy, w, h, confidence)
- **Class:** Single class ("object" = any shelf product). Product classification is a future concern.
- **Postprocessing:** Confidence threshold (0.25) + NMS (IoU 0.5) implemented in app code
- **Source weights:** `training/models/` (not tracked in git — exported via Colab)

## Prototypes

| Prototype | Architecture | ML Runtime | Status |
|-----------|-------------|------------|--------|
| **A** — React Native | React Native + Expo | `react-native-fast-tflite` (on-device) | Working on Android |
| **B** — Flutter | Flutter | `tflite_flutter` (on-device) | Working on Android |
| **C** — Rust + Flutter | Axum server + Flutter client | ONNX Runtime (server-side) | Not started |

## Core User Flow (all prototypes)

1. User opens app → camera viewfinder
2. User points camera at shelf → taps capture button
3. System analyzes photo → detects and counts products
4. User sees results: product count, annotated image with bounding boxes, timing info
5. User can scan another shelf

## Success Criteria for Prototypes

Each prototype is a **minimal spike** to evaluate:

- [x] Camera capture works reliably
- [x] ML inference produces bounding boxes on shelf products
- [x] Count of detected items is displayed
- [ ] End-to-end latency is acceptable (< 3s for on-device, < 5s for server)
- [ ] Developer experience (build times, debugging, iteration speed)

## Decision Point

After building all three, we compare on:
1. **Accuracy** — detection count and bounding box quality
2. **Speed** — inference latency and total pipeline time
3. **Developer experience** — how fast can we iterate?
4. **Offline capability** — does the use case require it?
5. **Scalability** — path to production
