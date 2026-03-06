# Non-Front-Facing Product Classification

## The Core Challenge

Front-facing products show their label → easy for CLIP/classification. But real shelves have:
- **Side-facing** products (only the cap/side visible)
- **Top-down** views (only the top of the package)
- **Partially occluded** (one product behind another)
- **Rotated/knocked over** items

Current pipeline (YOLO detects box → CLIP classifies crop) relies on the crop looking visually similar to a reference image. A side view of a Coke bottle looks nothing like the front.

## Approaches (Increasing Complexity)

### 1. More Reference Images (Low Effort — Do This First)
Instead of 1-3 front-facing reference photos per product, capture 10-20 covering:
- Front, back, side, top, 45° angles
- With and without cap visible
- Partial views (just the label corner)

The existing CLIP pipeline improves significantly. Cheapest win.

### 2. Fine-Tuned CLIP / Contrastive Learning (Medium Effort)
Train a model that maps all views of the same product to nearby embeddings:
- Take multi-angle dataset
- Fine-tune CLIP's visual encoder so front, side, and top views of the same product all cluster together
- This is **metric learning** — teaching the model "these are the same product despite looking different"
- Frameworks: OpenCLIP fine-tuning, or a simpler Siamese network with triplet loss

### 3. Multi-View / 3D-Aware Models (High Effort)
Train a model that understands products as 3D objects, not flat images:
- **NeRF-based product representations** — build a 3D model from photos, match any view
- **Synthetic data** — render 3D product models on virtual shelves, train on those

## Hardware Considerations

For the current prototype: phone camera is fine.

For production at scale, the industry uses:

| Hardware | Use Case | Why |
|---|---|---|
| Phone camera (current) | Auditor walks store, takes photos | Cheap, flexible, good enough for front-facing |
| Fixed shelf cameras | Real-time monitoring | Always-on, catches restocking events |
| Depth cameras (Intel RealSense, LiDAR) | 3D shelf mapping | Distinguishes front vs back products, measures depth |
| Robot-mounted cameras | Autonomous scanning | Walmart/Simbe use shelf-scanning robots |
| Multi-angle rigs | Product registration | Capture all angles for training data |

Depth cameras are the most relevant upgrade — they detect that a product is side-facing based on 3D orientation, then route to a different classification strategy.

## Competitive Landscape

### Established Players
- **Trax (now part of Smarkio)** — crowdsourced photos from auditors + massive proprietary product image DB (millions of SKUs, all angles). Solve it with sheer data volume.
- **Standard AI / Grabango** — ceiling-mounted cameras + depth sensors in checkout-free stores. Multiple camera angles = don't need to classify from one view.
- **Simbe Robotics (Tally)** — autonomous robot drives aisles with multiple cameras. Captures shelves from different positions, stitches together panoramic view.
- **Walmart** — partnered with multiple vendors, uses robot scanning + fixed cameras in some stores.
- **Amazon (Just Walk Out)** — ceiling cameras + shelf weight sensors. Weight sensor tells them what was taken regardless of visual angle.

### Common Patterns
1. Nobody relies on a single image — they all use multiple views, multiple sensors, or massive training datasets
2. Sensor fusion is the industry direction — cameras + weight sensors + RFID
3. The real moat is the product database — Trax has millions of SKU images

## Training Data Requirements

| Stage | Data Required | Volume |
|---|---|---|
| Current (CLIP matching) | 3-5 front-facing photos per SKU | ~50 SKUs = 250 images |
| Multi-angle CLIP | 15-20 photos per SKU (all angles) | ~50 SKUs = 1,000 images |
| Fine-tuned classifier | 50-100 photos per SKU in real shelf conditions | ~50 SKUs = 5,000 images |
| Production-grade | 200+ per SKU + synthetic augmentation | Need a data pipeline, not manual collection |

## Complexity Assessment

```
Current state:  Front-facing detection + CLIP matching
                Complexity: ██░░░░░░░░

Multi-angle:    More reference images, same pipeline
                Complexity: ███░░░░░░░  (low effort, good ROI)

Fine-tuned:     Custom metric learning model
                Complexity: ██████░░░░  (needs ML training pipeline)

3D-aware:       Depth cameras + multi-view fusion
                Complexity: ████████░░  (hardware + ML + calibration)

Production:     Sensor fusion + massive product DB
                Complexity: ██████████  (what Trax/Simbe do, $M+ investment)
```

## Recommended Path for Prototype

1. **Now**: Capture multi-angle reference images (15-20 per SKU). See how much CLIP improves. Zero code changes needed.
2. **Next**: If CLIP plateaus, explore fine-tuning with contrastive learning.
3. **Don't**: Jump to depth cameras or robots yet. Validate the classification approach first with better data.

The biggest bang for buck is **more training data, not more complex models**.
