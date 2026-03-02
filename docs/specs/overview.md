# Shelf Count — Project Overview

## Vision

A mobile tool for retail shelf auditing. A store associate photographs a shelf, and the system counts products, identifies gaps, and reports shelf status.

## Problem

Manual shelf counts are slow, error-prone, and tedious. Associates walk aisles with clipboards or handheld scanners, counting items one by one. This takes hours and the data is stale by the time it's compiled.

## Solution

Point a phone camera at a shelf → get instant product counts and out-of-stock detection.

## Prototypes

We are exploring three architectural approaches before committing to one:

| Prototype | Architecture | ML Location | Connectivity |
|-----------|-------------|-------------|-------------|
| **A** — React Native | React Native + Expo | On-device (TFLite) | Offline capable |
| **B** — Flutter | Flutter + ML Kit | On-device (Google ML Kit) | Offline capable |
| **C** — Rust + Flutter | Axum server + Flutter client | Server-side (Rust) | Requires network |

## Core User Flow (all prototypes)

1. User opens app → camera viewfinder
2. User points camera at shelf → captures image
3. System analyzes image → detects and counts products
4. User sees results: product count, annotated image with bounding boxes
5. User can save/export the audit result

## Success Criteria for Prototypes

Each prototype is a **minimal spike** to evaluate:

- [ ] Camera capture works reliably
- [ ] ML inference produces bounding boxes on shelf products
- [ ] Count of detected items is displayed
- [ ] End-to-end latency is acceptable (< 3s for on-device, < 5s for server)
- [ ] Developer experience (build times, debugging, iteration speed)

## Decision Point

After building all three, we compare on:
1. **Accuracy** — which ML approach detects products best?
2. **Speed** — inference latency and UX responsiveness
3. **Developer experience** — how fast can we iterate?
4. **Offline capability** — does the use case require it?
5. **Scalability** — path to production
