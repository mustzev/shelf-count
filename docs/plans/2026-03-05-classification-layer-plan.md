# Classification Layer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add product identification (SKU classification) to the proto-c detection pipeline using CLIP embeddings + nearest-neighbor matching.

**Architecture:** After detection, crop each bounding box from the original image, run through CLIP ViT-B/32 visual encoder to get a 512-dim embedding, then find the closest match in a reference product database using cosine similarity. Products are registered via a REST endpoint — no retraining needed.

**Tech Stack:** Rust (Axum, ort), CLIP ViT-B/32 (ONNX), JSON file for product reference store, Flutter/Dart client.

---

### Task 1: Export CLIP Visual Encoder to ONNX

**Files:**
- Create: `training/scripts/export_clip_onnx.py`

**Context:** We only need the visual (image) encoder, not the text encoder. CLIP ViT-B/32 takes a 224×224 image and outputs a 512-dim embedding vector.

**Step 1: Create the export script**

```python
"""Export CLIP ViT-B/32 visual encoder to ONNX."""

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

def main():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    visual = model.vision_model
    projection = model.visual_projection

    class VisualEncoder(torch.nn.Module):
        def __init__(self, vision_model, visual_projection):
            super().__init__()
            self.vision_model = vision_model
            self.visual_projection = visual_projection

        def forward(self, pixel_values):
            outputs = self.vision_model(pixel_values=pixel_values)
            pooled = outputs.pooler_output
            embeds = self.visual_projection(pooled)
            # L2 normalize
            embeds = embeds / embeds.norm(dim=-1, keepdim=True)
            return embeds

    encoder = VisualEncoder(visual, projection)
    encoder.eval()

    dummy = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        encoder,
        dummy,
        "models/clip-vit-b32-visual.onnx",
        input_names=["pixel_values"],
        output_names=["embeddings"],
        dynamic_axes={"pixel_values": {0: "batch"}, "embeddings": {0: "batch"}},
        opset_version=17,
    )

    print("Exported to models/clip-vit-b32-visual.onnx")
    print("Input:  pixel_values [batch, 3, 224, 224]")
    print("Output: embeddings   [batch, 512] (L2-normalized)")

if __name__ == "__main__":
    main()
```

**Step 2: Run the export**

Run: `cd /Users/erchis/Projects/shelf-count/training && python scripts/export_clip_onnx.py`
Expected: `models/clip-vit-b32-visual.onnx` created (~350 MB)

**Step 3: Copy ONNX to server**

Run: `cp training/models/clip-vit-b32-visual.onnx proto-c-rust-flutter/server/models/`

---

### Task 2: Add Classification Types

**Files:**
- Modify: `proto-c-rust-flutter/server/src/ml/types.rs`

**Context:** Add `sku` and `sku_confidence` fields to `Detection`. Add `ProductRef` struct for the reference store. These must be `Option` so classification is backwards-compatible.

**Step 1: Update Detection struct**

Add two optional fields to `Detection` in `types.rs`:

```rust
#[derive(Debug, Clone, Serialize)]
pub struct Detection {
    pub label: String,
    pub confidence: f64,
    pub bbox: BoundingBox,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sku: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sku_confidence: Option<f64>,
}
```

**Step 2: Add ProductRef struct**

Add to `types.rs`:

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductRef {
    pub name: String,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductStore {
    pub products: Vec<ProductRef>,
}
```

**Step 3: Fix all places that construct Detection**

Update `yolo.rs` and `detr.rs` where `Detection` is constructed to add `sku: None, sku_confidence: None`.

**Step 4: Verify it compiles**

Run: `cd proto-c-rust-flutter/server && cargo build`
Expected: Compiles with no errors.

---

### Task 3: Implement CLIP Embedding Model

**Files:**
- Create: `proto-c-rust-flutter/server/src/ml/clip.rs`
- Modify: `proto-c-rust-flutter/server/src/ml/mod.rs`

**Context:** The CLIP model takes a 224×224 image crop and returns a 512-dim L2-normalized embedding. Preprocessing: resize to 224×224, normalize with CLIP mean/std, NCHW layout.

**Step 1: Create clip.rs**

```rust
use std::path::Path;
use std::sync::Mutex;

use image::DynamicImage;
use ndarray::Array4;
use ort::inputs;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::TensorRef;

use crate::error::AppError;

const CLIP_SIZE: u32 = 224;
const CLIP_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const CLIP_STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];

pub struct ClipModel {
    session: Mutex<Session>,
}

impl ClipModel {
    pub fn load(model_path: &Path) -> Result<Self, AppError> {
        let session = Session::builder()
            .map_err(|e| AppError::internal(format!("CLIP session builder: {e}")))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| AppError::internal(format!("CLIP optimization: {e}")))?
            .with_intra_threads(4)
            .map_err(|e| AppError::internal(format!("CLIP threads: {e}")))?
            .commit_from_file(model_path)
            .map_err(|e| AppError::internal(format!("CLIP load from {}: {e}", model_path.display())))?;

        tracing::info!("Loaded CLIP model from {}", model_path.display());
        Ok(Self { session: Mutex::new(session) })
    }

    /// Embed a single image crop. Returns a 512-dim L2-normalized vector.
    pub fn embed(&self, img: &DynamicImage) -> Result<Vec<f32>, AppError> {
        let resized = img.resize_exact(CLIP_SIZE, CLIP_SIZE, image::imageops::FilterType::Triangle);
        let rgb = resized.to_rgb8();

        let mut input = Array4::<f32>::zeros((1, 3, CLIP_SIZE as usize, CLIP_SIZE as usize));
        for y in 0..CLIP_SIZE as usize {
            for x in 0..CLIP_SIZE as usize {
                let pixel = rgb.get_pixel(x as u32, y as u32);
                input[[0, 0, y, x]] = (pixel[0] as f32 / 255.0 - CLIP_MEAN[0]) / CLIP_STD[0];
                input[[0, 1, y, x]] = (pixel[1] as f32 / 255.0 - CLIP_MEAN[1]) / CLIP_STD[1];
                input[[0, 2, y, x]] = (pixel[2] as f32 / 255.0 - CLIP_MEAN[2]) / CLIP_STD[2];
            }
        }

        let tensor = TensorRef::from_array_view(input.view())
            .map_err(|e| AppError::internal(format!("CLIP tensor: {e}")))?;

        let mut session = self.session.lock()
            .map_err(|e| AppError::internal(format!("CLIP lock: {e}")))?;

        let outputs = session.run(inputs![tensor])
            .map_err(|e| AppError::internal(format!("CLIP inference: {e}")))?;

        let embedding = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| AppError::internal(format!("CLIP extract: {e}")))?;

        Ok(embedding.1.to_vec())
    }
}
```

**Step 2: Register clip module in mod.rs**

Add `pub mod clip;` to `proto-c-rust-flutter/server/src/ml/mod.rs`.

**Step 3: Verify it compiles**

Run: `cargo build`
Expected: Compiles.

---

### Task 4: Implement Product Reference Store

**Files:**
- Create: `proto-c-rust-flutter/server/src/products.rs`
- Modify: `proto-c-rust-flutter/server/src/main.rs` (add `mod products;`)

**Context:** The product store loads/saves `products.json`, holds product embeddings in memory, and provides cosine similarity matching.

**Step 1: Create products.rs**

```rust
use std::path::{Path, PathBuf};
use std::sync::RwLock;

use crate::error::AppError;
use crate::ml::types::{ProductRef, ProductStore};

const SIMILARITY_THRESHOLD: f32 = 0.6;

pub struct ProductDb {
    store: RwLock<ProductStore>,
    path: PathBuf,
}

impl ProductDb {
    pub fn load(path: &Path) -> Self {
        let store = if path.exists() {
            let data = std::fs::read_to_string(path).unwrap_or_else(|_| "{}".to_string());
            serde_json::from_str(&data).unwrap_or(ProductStore { products: vec![] })
        } else {
            ProductStore { products: vec![] }
        };

        tracing::info!("Product DB: {} products loaded from {}", store.products.len(), path.display());

        Self {
            store: RwLock::new(store),
            path: path.to_path_buf(),
        }
    }

    pub fn add(&self, product: ProductRef) -> Result<(), AppError> {
        let mut store = self.store.write()
            .map_err(|e| AppError::internal(format!("Product DB write lock: {e}")))?;

        // Replace if name exists
        store.products.retain(|p| p.name != product.name);
        store.products.push(product);

        self.save(&store)
    }

    pub fn remove(&self, name: &str) -> Result<bool, AppError> {
        let mut store = self.store.write()
            .map_err(|e| AppError::internal(format!("Product DB write lock: {e}")))?;

        let before = store.products.len();
        store.products.retain(|p| p.name != name);
        let removed = store.products.len() < before;

        if removed {
            self.save(&store)?;
        }

        Ok(removed)
    }

    pub fn list(&self) -> Result<Vec<String>, AppError> {
        let store = self.store.read()
            .map_err(|e| AppError::internal(format!("Product DB read lock: {e}")))?;
        Ok(store.products.iter().map(|p| p.name.clone()).collect())
    }

    /// Find the closest product to the given embedding. Returns (name, similarity) or None.
    pub fn find_closest(&self, embedding: &[f32]) -> Result<Option<(String, f32)>, AppError> {
        let store = self.store.read()
            .map_err(|e| AppError::internal(format!("Product DB read lock: {e}")))?;

        if store.products.is_empty() {
            return Ok(None);
        }

        let mut best_name = String::new();
        let mut best_sim: f32 = -1.0;

        for product in &store.products {
            let sim = cosine_similarity(embedding, &product.embedding);
            if sim > best_sim {
                best_sim = sim;
                best_name = product.name.clone();
            }
        }

        if best_sim >= SIMILARITY_THRESHOLD {
            Ok(Some((best_name, best_sim)))
        } else {
            Ok(None)
        }
    }

    fn save(&self, store: &ProductStore) -> Result<(), AppError> {
        let json = serde_json::to_string_pretty(store)
            .map_err(|e| AppError::internal(format!("Serialize products: {e}")))?;
        std::fs::write(&self.path, json)
            .map_err(|e| AppError::internal(format!("Write products file: {e}")))?;
        Ok(())
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}
```

**Step 2: Add `mod products;` to main.rs**

**Step 3: Verify it compiles**

Run: `cargo build`

---

### Task 5: Create Shared App State

**Files:**
- Modify: `proto-c-rust-flutter/server/src/main.rs`

**Context:** Currently the Axum state is `Arc<dyn DetectionModel>`. We need to add the CLIP model and ProductDb to the state. Create an `AppState` struct.

**Step 1: Define AppState**

In `main.rs`, define:

```rust
use crate::ml::clip::ClipModel;
use crate::products::ProductDb;

pub struct AppState {
    pub detection_model: Arc<dyn DetectionModel>,
    pub clip_model: Option<Arc<ClipModel>>,
    pub product_db: Arc<ProductDb>,
}
```

**Step 2: Initialize in main**

After detection model loading, add:

```rust
let clip_model = env::var("CLIP_MODEL_PATH").ok().map(|path| {
    let m = ClipModel::load(std::path::Path::new(&path))
        .expect("Failed to load CLIP model");
    Arc::new(m)
});

let product_db_path = env::var("PRODUCT_DB_PATH")
    .unwrap_or_else(|_| "products.json".to_string());
let product_db = Arc::new(ProductDb::load(std::path::Path::new(&product_db_path)));

let state = Arc::new(AppState {
    detection_model: model,
    clip_model,
    product_db,
});
```

**Step 3: Update Router state type**

Change `.with_state(model)` to `.with_state(state)`.

**Step 4: Update analyze.rs and health.rs**

Change `State(model): State<Arc<dyn DetectionModel>>` to `State(state): State<Arc<AppState>>` in both handlers. Update references from `model.` to `state.detection_model.`.

**Step 5: Update .env**

Add to `.env`:

```
CLIP_MODEL_PATH=models/clip-vit-b32-visual.onnx
PRODUCT_DB_PATH=products.json
```

**Step 6: Verify it compiles and runs**

Run: `cargo build`

---

### Task 6: Add Classification to Analyze Endpoint

**Files:**
- Modify: `proto-c-rust-flutter/server/src/routes/analyze.rs`

**Context:** After detection, if CLIP model is loaded, crop each detection from the original image, embed it, and match against the product DB. Mutate the `Detection` structs to add `sku` and `sku_confidence`.

**Step 1: Add classification logic to analyze handler**

After `let result = state.detection_model.run(img.clone())?;`, add:

```rust
let mut result = state.detection_model.run(img.clone())?;

// Classify each detection if CLIP model is available
if let Some(clip) = &state.clip_model {
    let (img_w, img_h) = (img.width(), img.height());

    for det in &mut result.detections {
        // Crop bounding box from original image
        let x = (det.bbox.x * img_w as f64) as u32;
        let y = (det.bbox.y * img_h as f64) as u32;
        let w = (det.bbox.width * img_w as f64) as u32;
        let h = (det.bbox.height * img_h as f64) as u32;

        let x = x.min(img_w.saturating_sub(1));
        let y = y.min(img_h.saturating_sub(1));
        let w = w.min(img_w - x).max(1);
        let h = h.min(img_h - y).max(1);

        let crop = img.crop_imm(x, y, w, h);
        let embedding = clip.embed(&crop.into())?;

        if let Some((name, sim)) = state.product_db.find_closest(&embedding)? {
            det.sku = Some(name);
            det.sku_confidence = Some(sim as f64);
        }
    }
}
```

**Step 2: Verify it compiles**

Run: `cargo build`

---

### Task 7: Product Management Endpoints

**Files:**
- Create: `proto-c-rust-flutter/server/src/routes/products.rs`
- Modify: `proto-c-rust-flutter/server/src/routes/mod.rs`
- Modify: `proto-c-rust-flutter/server/src/main.rs` (add routes)

**Context:** `POST /products` to register a product (name + images), `DELETE /products/:name` to remove, `GET /products` to list.

**Step 1: Create products.rs route**

```rust
use axum::extract::{Multipart, Path, State};
use axum::Json;
use std::sync::Arc;

use crate::error::AppError;
use crate::ml::types::ProductRef;
use crate::AppState;

pub async fn add_product(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Result<Json<serde_json::Value>, AppError> {
    let clip = state.clip_model.as_ref()
        .ok_or_else(|| AppError::bad_request("CLIP model not loaded — classification not available"))?;

    let mut name: Option<String> = None;
    let mut embeddings: Vec<Vec<f32>> = Vec::new();

    while let Some(field) = multipart.next_field().await.map_err(|e| {
        AppError::bad_request(format!("Multipart error: {e}"))
    })? {
        match field.name() {
            Some("name") => {
                name = Some(field.text().await.map_err(|e| {
                    AppError::bad_request(format!("Failed to read name: {e}"))
                })?);
            }
            Some("images") => {
                let data = field.bytes().await.map_err(|e| {
                    AppError::bad_request(format!("Failed to read image: {e}"))
                })?;
                let img = image::load_from_memory(&data)
                    .map_err(|e| AppError::bad_request(format!("Failed to decode image: {e}")))?;
                let emb = clip.embed(&img)?;
                embeddings.push(emb);
            }
            _ => {}
        }
    }

    let name = name.ok_or_else(|| AppError::bad_request("Missing 'name' field"))?;
    if embeddings.is_empty() {
        return Err(AppError::bad_request("At least one image required"));
    }

    // Average embeddings and L2-normalize
    let dim = embeddings[0].len();
    let mut avg = vec![0.0f32; dim];
    for emb in &embeddings {
        for (i, v) in emb.iter().enumerate() {
            avg[i] += v;
        }
    }
    let n = embeddings.len() as f32;
    let norm: f32 = avg.iter().map(|x| (x / n) * (x / n)).sum::<f32>().sqrt();
    for v in &mut avg {
        *v = (*v / n) / norm;
    }

    let product = ProductRef { name: name.clone(), embedding: avg };
    state.product_db.add(product)?;

    tracing::info!("Registered product '{}' with {} reference images", name, embeddings.len());

    Ok(Json(serde_json::json!({
        "status": "ok",
        "name": name,
        "images_processed": embeddings.len()
    })))
}

pub async fn remove_product(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<Json<serde_json::Value>, AppError> {
    let removed = state.product_db.remove(&name)?;

    if removed {
        Ok(Json(serde_json::json!({ "status": "ok", "removed": name })))
    } else {
        Err(AppError::bad_request(format!("Product '{}' not found", name)))
    }
}

pub async fn list_products(
    State(state): State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>, AppError> {
    let products = state.product_db.list()?;
    Ok(Json(serde_json::json!({ "products": products })))
}
```

**Step 2: Register in routes/mod.rs**

Add `pub mod products;`

**Step 3: Add routes in main.rs**

```rust
.route("/products", axum::routing::get(routes::products::list_products))
.route("/products", axum::routing::post(routes::products::add_product))
.route("/products/:name", axum::routing::delete(routes::products::remove_product))
```

**Step 4: Verify it compiles**

Run: `cargo build`

---

### Task 8: Update Flutter Client Models

**Files:**
- Modify: `proto-c-rust-flutter/client/lib/models/detection.dart`

**Context:** Add nullable `sku` and `skuConfidence` fields to `Detection` model.

**Step 1: Update Detection class**

```dart
class Detection {
  final String label;
  final double confidence;
  final BoundingBox bbox;
  final String? sku;
  final double? skuConfidence;

  const Detection({
    required this.label,
    required this.confidence,
    required this.bbox,
    this.sku,
    this.skuConfidence,
  });

  factory Detection.fromJson(Map<String, dynamic> json) {
    return Detection(
      label: json['label'] as String,
      confidence: (json['confidence'] as num).toDouble(),
      bbox: BoundingBox.fromJson(json['bbox'] as Map<String, dynamic>),
      sku: json['sku'] as String?,
      skuConfidence: json['sku_confidence'] != null
          ? (json['sku_confidence'] as num).toDouble()
          : null,
    );
  }
}
```

---

### Task 9: Update Bounding Box Overlay to Show SKU Names

**Files:**
- Modify: `proto-c-rust-flutter/client/lib/widgets/bounding_box_overlay.dart`

**Context:** When a detection has an `sku`, show the product name instead of (or alongside) the confidence percentage.

**Step 1: Update _BoxPainter.paint**

Change the text label logic:

```dart
// Build label text
String labelText;
if (det.sku != null) {
  labelText = ' ${det.sku} ${(det.skuConfidence! * 100).round()}% ';
} else {
  labelText = ' ${(det.confidence * 100).round()}% ';
}

final textSpan = TextSpan(
  text: labelText,
  style: textStyle,
);
```

---

### Task 10: End-to-End Test

**Steps:**
1. Export CLIP model and copy to `server/models/`
2. Set `CLIP_MODEL_PATH=models/clip-vit-b32-visual.onnx` in `.env`
3. Start server: `cargo run`
4. Register a product via curl:
   ```bash
   curl -X POST http://localhost:9000/products \
     -F "name=Test Product" \
     -F "images=@photo1.jpg" \
     -F "images=@photo2.jpg"
   ```
5. List products: `curl http://localhost:9000/products`
6. Scan a shelf with the Flutter app — detections with matching products should show SKU names
7. Remove product: `curl -X DELETE http://localhost:9000/products/Test%20Product`
