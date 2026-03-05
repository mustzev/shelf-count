# Proto C: Rust Axum Server + Flutter Client — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Rust Axum server that runs ONNX object detection on shelf images, with a Flutter client that captures and uploads photos.

**Architecture:** Trait-based model abstraction (`DetectionModel`) with two implementations (YOLOv8m, DETR). Server loads one model at startup via CLI arg. Flutter client sends images over HTTP and displays results.

**Tech Stack:** Rust (Axum, ort, image, ndarray, clap, serde), Flutter/Dart (camera, http)

**Design doc:** `docs/plans/2026-03-05-proto-c-rust-flutter-design.md`

---

## Part 1: Rust Server

### Task 1: Scaffold Rust project with Cargo

**Files:**
- Create: `proto-c-rust-flutter/server/Cargo.toml`
- Create: `proto-c-rust-flutter/server/src/main.rs`
- Create: `proto-c-rust-flutter/server/models/.gitkeep`
- Remove: `proto-c-rust-flutter/server/.gitkeep`

**Step 1: Initialize Cargo project**

Run from repo root:
```bash
cd proto-c-rust-flutter/server && cargo init --name shelf-count-server
```

**Step 2: Add dependencies to Cargo.toml**

Replace the `[dependencies]` section:
```toml
[dependencies]
axum = { version = "0.8", features = ["multipart"] }
tokio = { version = "1", features = ["full"] }
tower-http = { version = "0.6", features = ["cors", "trace"] }
ort = { version = "2.0.0-rc.9", features = ["ndarray"] }
ndarray = "0.16"
image = "0.25"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
clap = { version = "4", features = ["derive"] }
tracing = "0.1"
tracing-subscriber = "0.3"
```

**Step 3: Write minimal main.rs that compiles**

```rust
fn main() {
    println!("shelf-count-server");
}
```

**Step 4: Verify it builds**

Run: `cd proto-c-rust-flutter/server && cargo build`
Expected: Compiles successfully (downloads deps on first run — may take a few minutes).

**Step 5: Create models directory**

```bash
mkdir -p proto-c-rust-flutter/server/models
touch proto-c-rust-flutter/server/models/.gitkeep
rm proto-c-rust-flutter/server/.gitkeep
```

**Step 6: Commit**

```bash
git add proto-c-rust-flutter/server/
git commit -m "chore(proto-c): scaffold Rust server with cargo"
```

---

### Task 2: Types and error handling

**Files:**
- Create: `proto-c-rust-flutter/server/src/ml/types.rs`
- Create: `proto-c-rust-flutter/server/src/ml/mod.rs`
- Create: `proto-c-rust-flutter/server/src/error.rs`

**Step 1: Create ml/types.rs with serde-serializable detection types**

```rust
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct BoundingBox {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Detection {
    pub label: String,
    pub confidence: f64,
    pub bbox: BoundingBox,
}

#[derive(Debug, Clone, Serialize)]
pub struct DetectionResult {
    pub model: String,
    pub count: usize,
    pub detections: Vec<Detection>,
    pub inference_time_ms: u64,
}
```

**Step 2: Create ml/mod.rs with DetectionModel trait**

```rust
pub mod types;

use image::DynamicImage;
use crate::error::AppError;
use types::DetectionResult;

pub trait DetectionModel: Send + Sync {
    fn name(&self) -> &str;
    fn run(&self, image: DynamicImage) -> Result<DetectionResult, AppError>;
}
```

**Step 3: Create error.rs**

```rust
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::json;

#[derive(Debug)]
pub struct AppError {
    pub status: StatusCode,
    pub message: String,
}

impl AppError {
    pub fn bad_request(msg: impl Into<String>) -> Self {
        Self { status: StatusCode::BAD_REQUEST, message: msg.into() }
    }

    pub fn internal(msg: impl Into<String>) -> Self {
        Self { status: StatusCode::INTERNAL_SERVER_ERROR, message: msg.into() }
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let body = json!({ "error": self.message });
        (self.status, axum::Json(body)).into_response()
    }
}

impl std::fmt::Display for AppError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.status, self.message)
    }
}

impl std::error::Error for AppError {}
```

**Step 4: Wire modules in main.rs**

```rust
mod error;
mod ml;

fn main() {
    println!("shelf-count-server");
}
```

**Step 5: Verify it compiles**

Run: `cd proto-c-rust-flutter/server && cargo build`
Expected: Compiles successfully.

**Step 6: Commit**

```bash
git add proto-c-rust-flutter/server/src/
git commit -m "feat(proto-c): add detection types, trait, and error handling"
```

---

### Task 3: Health endpoint and Axum server setup

**Files:**
- Create: `proto-c-rust-flutter/server/src/routes/mod.rs`
- Create: `proto-c-rust-flutter/server/src/routes/health.rs`
- Modify: `proto-c-rust-flutter/server/src/main.rs`

**Step 1: Create routes/health.rs**

```rust
use axum::extract::State;
use axum::Json;
use serde_json::{json, Value};
use std::sync::Arc;

use crate::ml::DetectionModel;

pub async fn health(State(model): State<Arc<dyn DetectionModel>>) -> Json<Value> {
    Json(json!({
        "status": "ok",
        "model": model.name()
    }))
}
```

**Step 2: Create routes/mod.rs**

```rust
pub mod health;
```

**Step 3: Update main.rs with Axum server, CLI parsing, and a stub model**

For now, use a `StubModel` so we can test the health endpoint without needing an ONNX file.

```rust
mod error;
mod ml;
mod routes;

use std::sync::Arc;
use axum::{routing::get, Router};
use clap::Parser;
use tower_http::cors::CorsLayer;
use tracing_subscriber;

use error::AppError;
use ml::{DetectionModel, types::DetectionResult};

#[derive(Parser)]
#[command(name = "shelf-count-server")]
struct Args {
    /// Model to use: "yolo" or "detr"
    #[arg(long, default_value = "yolo")]
    model: String,

    /// Path to the ONNX model file
    #[arg(long, default_value = "models/yolov8m-sku110k.onnx")]
    model_path: String,

    /// Port to listen on
    #[arg(long, default_value_t = 3000)]
    port: u16,
}

/// Temporary stub for testing server setup before real ONNX integration.
struct StubModel {
    model_name: String,
}

impl DetectionModel for StubModel {
    fn name(&self) -> &str {
        &self.model_name
    }

    fn run(&self, _image: image::DynamicImage) -> Result<DetectionResult, AppError> {
        Ok(DetectionResult {
            model: self.model_name.clone(),
            count: 0,
            detections: vec![],
            inference_time_ms: 0,
        })
    }
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    // Stub model for now — replaced in Task 5 with real ONNX model
    let model: Arc<dyn DetectionModel> = Arc::new(StubModel {
        model_name: format!("stub-{}", args.model),
    });

    let app = Router::new()
        .route("/health", get(routes::health::health))
        .layer(CorsLayer::permissive())
        .with_state(model);

    let addr = format!("0.0.0.0:{}", args.port);
    tracing::info!("Listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
```

**Step 4: Run the server and test health endpoint**

Terminal 1:
```bash
cd proto-c-rust-flutter/server && cargo run
```

Terminal 2:
```bash
curl http://localhost:3000/health
```
Expected: `{"model":"stub-yolo","status":"ok"}`

**Step 5: Commit**

```bash
git add proto-c-rust-flutter/server/src/
git commit -m "feat(proto-c): add health endpoint and Axum server setup"
```

---

### Task 4: Analyze endpoint (multipart image upload)

**Files:**
- Create: `proto-c-rust-flutter/server/src/routes/analyze.rs`
- Modify: `proto-c-rust-flutter/server/src/routes/mod.rs`
- Modify: `proto-c-rust-flutter/server/src/main.rs`

**Step 1: Create routes/analyze.rs**

```rust
use axum::extract::{Multipart, State};
use axum::Json;
use std::sync::Arc;

use crate::error::AppError;
use crate::ml::DetectionModel;
use crate::ml::types::DetectionResult;

pub async fn analyze(
    State(model): State<Arc<dyn DetectionModel>>,
    mut multipart: Multipart,
) -> Result<Json<DetectionResult>, AppError> {
    // Extract image bytes from multipart form
    let mut image_bytes: Option<Vec<u8>> = None;

    while let Some(field) = multipart.next_field().await.map_err(|e| {
        AppError::bad_request(format!("Failed to read multipart field: {e}"))
    })? {
        if field.name() == Some("image") {
            let data = field.bytes().await.map_err(|e| {
                AppError::bad_request(format!("Failed to read image data: {e}"))
            })?;
            image_bytes = Some(data.to_vec());
            break;
        }
    }

    let bytes = image_bytes.ok_or_else(|| AppError::bad_request("Missing 'image' field"))?;

    // Decode image
    let img = image::load_from_memory(&bytes)
        .map_err(|e| AppError::bad_request(format!("Failed to decode image: {e}")))?;

    // Run inference
    let result = model.run(img)?;

    Ok(Json(result))
}
```

**Step 2: Update routes/mod.rs**

```rust
pub mod analyze;
pub mod health;
```

**Step 3: Add the route to main.rs**

Add this import and route in main.rs. Find the `Router::new()` block and add:
```rust
.route("/analyze", axum::routing::post(routes::analyze::analyze))
```

**Step 4: Test with curl using a real JPEG**

Terminal 1 (server running):
```bash
cd proto-c-rust-flutter/server && cargo run
```

Terminal 2 (use any JPEG file — grab a test image):
```bash
curl -X POST http://localhost:3000/analyze \
  -F "image=@/path/to/test.jpg"
```
Expected: `{"model":"stub-yolo","count":0,"detections":[],"inference_time_ms":0}`

**Step 5: Test error cases**

```bash
# Missing image field
curl -X POST http://localhost:3000/analyze
# Expected: 400 with {"error":"Missing 'image' field"}

# Invalid image data
curl -X POST http://localhost:3000/analyze -F "image=@Cargo.toml"
# Expected: 400 with {"error":"Failed to decode image: ..."}
```

**Step 6: Commit**

```bash
git add proto-c-rust-flutter/server/src/
git commit -m "feat(proto-c): add POST /analyze endpoint with multipart upload"
```

---

### Task 5: YOLOv8 model implementation

**Files:**
- Create: `proto-c-rust-flutter/server/src/ml/yolo.rs`
- Modify: `proto-c-rust-flutter/server/src/ml/mod.rs`
- Modify: `proto-c-rust-flutter/server/src/main.rs`

**Prerequisite:** You need the ONNX model file. Export from your training pipeline or download. Place at `proto-c-rust-flutter/server/models/yolov8m-sku110k.onnx`.

**Step 1: Create ml/yolo.rs**

```rust
use std::path::Path;
use std::time::Instant;

use image::DynamicImage;
use ndarray::Array4;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::TensorRef;
use ort::inputs;

use crate::error::AppError;
use super::DetectionModel;
use super::types::{BoundingBox, Detection, DetectionResult};

const INPUT_SIZE: u32 = 640;
const NUM_BOXES: usize = 8400;
const CONFIDENCE_THRESHOLD: f32 = 0.25;
const IOU_THRESHOLD: f32 = 0.5;

pub struct YoloModel {
    session: Session,
    name: String,
}

impl YoloModel {
    pub fn load(model_path: &Path) -> Result<Self, AppError> {
        let session = Session::builder()
            .map_err(|e| AppError::internal(format!("Failed to create session builder: {e}")))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| AppError::internal(format!("Failed to set optimization: {e}")))?
            .with_intra_threads(4)
            .map_err(|e| AppError::internal(format!("Failed to set threads: {e}")))?
            .commit_from_file(model_path)
            .map_err(|e| AppError::internal(format!("Failed to load model from {}: {e}", model_path.display())))?;

        let name = model_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("yolov8")
            .to_string();

        tracing::info!("Loaded YOLO model: {} from {}", name, model_path.display());
        Ok(Self { session, name })
    }

    fn preprocess(&self, img: &DynamicImage) -> Array4<f32> {
        let resized = img.resize_exact(INPUT_SIZE, INPUT_SIZE, image::imageops::FilterType::Triangle);
        let rgb = resized.to_rgb8();

        // Build NCHW [1, 3, 640, 640] with pixels normalized to 0–1
        let mut input = Array4::<f32>::zeros((1, 3, INPUT_SIZE as usize, INPUT_SIZE as usize));
        for y in 0..INPUT_SIZE as usize {
            for x in 0..INPUT_SIZE as usize {
                let pixel = rgb.get_pixel(x as u32, y as u32);
                input[[0, 0, y, x]] = pixel[0] as f32 / 255.0;
                input[[0, 1, y, x]] = pixel[1] as f32 / 255.0;
                input[[0, 2, y, x]] = pixel[2] as f32 / 255.0;
            }
        }
        input
    }

    fn postprocess(&self, raw: &[f32]) -> Vec<Detection> {
        // raw is [1, 5, 8400] row-major. First dimension (batch=1) already stripped by ort.
        // Layout: row 0 = all cx, row 1 = all cy, row 2 = all w, row 3 = all h, row 4 = all confidence
        struct Candidate {
            x: f32,
            y: f32,
            w: f32,
            h: f32,
            confidence: f32,
        }

        let mut candidates: Vec<Candidate> = Vec::new();

        for b in 0..NUM_BOXES {
            let confidence = raw[4 * NUM_BOXES + b];
            if confidence < CONFIDENCE_THRESHOLD {
                continue;
            }

            let cx = raw[0 * NUM_BOXES + b];
            let cy = raw[1 * NUM_BOXES + b];
            let w = raw[2 * NUM_BOXES + b];
            let h = raw[3 * NUM_BOXES + b];

            candidates.push(Candidate {
                x: cx - w / 2.0,
                y: cy - h / 2.0,
                w,
                h,
                confidence,
            });
        }

        // Sort by confidence descending
        candidates.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

        // NMS
        let mut suppressed = vec![false; candidates.len()];
        let mut kept: Vec<usize> = Vec::new();

        for i in 0..candidates.len() {
            if suppressed[i] {
                continue;
            }
            kept.push(i);

            for j in (i + 1)..candidates.len() {
                if suppressed[j] {
                    continue;
                }
                if iou(&candidates[i], &candidates[j]) > IOU_THRESHOLD {
                    suppressed[j] = true;
                }
            }
        }

        kept.iter()
            .map(|&i| {
                let c = &candidates[i];
                Detection {
                    label: "object".to_string(),
                    confidence: c.confidence as f64,
                    bbox: BoundingBox {
                        x: c.x.clamp(0.0, 1.0) as f64,
                        y: c.y.clamp(0.0, 1.0) as f64,
                        width: c.w.clamp(0.0, 1.0) as f64,
                        height: c.h.clamp(0.0, 1.0) as f64,
                    },
                }
            })
            .collect()
    }
}

fn iou(a: &dyn HasBox, b: &dyn HasBox) -> f32 {
    let x1 = a.x().max(b.x());
    let y1 = a.y().max(b.y());
    let x2 = (a.x() + a.w()).min(b.x() + b.w());
    let y2 = (a.y() + a.h()).min(b.y() + b.h());

    if x2 <= x1 || y2 <= y1 {
        return 0.0;
    }

    let intersection = (x2 - x1) * (y2 - y1);
    let area_a = a.w() * a.h();
    let area_b = b.w() * b.h();
    let union = area_a + area_b - intersection;

    if union > 0.0 { intersection / union } else { 0.0 }
}

trait HasBox {
    fn x(&self) -> f32;
    fn y(&self) -> f32;
    fn w(&self) -> f32;
    fn h(&self) -> f32;
}

impl HasBox for Candidate {
    fn x(&self) -> f32 { self.x }
    fn y(&self) -> f32 { self.y }
    fn w(&self) -> f32 { self.w }
    fn h(&self) -> f32 { self.h }
}

// Note: Candidate is defined inside postprocess(). To use the HasBox trait,
// move Candidate to module level. The implementor should restructure this
// so Candidate is a private struct at module scope.

impl DetectionModel for YoloModel {
    fn name(&self) -> &str {
        &self.name
    }

    fn run(&self, image: DynamicImage) -> Result<DetectionResult, AppError> {
        let input = self.preprocess(&image);

        let start = Instant::now();
        let outputs = self.session.run(
            inputs![TensorRef::from_array_view(input.view())
                .map_err(|e| AppError::internal(format!("Failed to create input tensor: {e}")))?]
            .map_err(|e| AppError::internal(format!("Failed to create inputs: {e}")))?
        ).map_err(|e| AppError::internal(format!("Inference failed: {e}")))?;

        let inference_time_ms = start.elapsed().as_millis() as u64;

        let predictions = outputs[0]
            .try_extract_raw_tensor::<f32>()
            .map_err(|e| AppError::internal(format!("Failed to extract output: {e}")))?;

        let detections = self.postprocess(predictions.1);

        Ok(DetectionResult {
            model: self.name.clone(),
            count: detections.len(),
            detections,
            inference_time_ms,
        })
    }
}
```

> **Note for implementor:** The `Candidate` struct is shown inside `postprocess()` for clarity, but Rust requires it at module scope to implement traits. Move `Candidate` to a private struct at the top of the file. Also, the `iou` function can take two `&Candidate` directly instead of using a trait — that's simpler. The trait approach above is for illustration; simplify during implementation.

**Step 2: Update ml/mod.rs**

```rust
pub mod types;
pub mod yolo;

use image::DynamicImage;
use crate::error::AppError;
use types::DetectionResult;

pub trait DetectionModel: Send + Sync {
    fn name(&self) -> &str;
    fn run(&self, image: DynamicImage) -> Result<DetectionResult, AppError>;
}
```

**Step 3: Update main.rs to load real YOLO model**

Replace the `StubModel` usage. Keep `StubModel` as fallback if model file doesn't exist (for development), or exit with error — implementor's choice. Recommended: exit with clear error.

```rust
// In main(), replace the stub block:
let model: Arc<dyn DetectionModel> = match args.model.as_str() {
    "yolo" => {
        let m = ml::yolo::YoloModel::load(Path::new(&args.model_path))
            .expect("Failed to load YOLO model");
        Arc::new(m)
    }
    other => {
        eprintln!("Unknown model: {}. Supported: yolo, detr", other);
        std::process::exit(1);
    }
};
```

Add `use std::path::Path;` and `use std::sync::Arc;` to imports.

**Step 4: Test with a real image**

Place the ONNX model at `proto-c-rust-flutter/server/models/yolov8m-sku110k.onnx`, then:

```bash
cd proto-c-rust-flutter/server && cargo run -- --model-path models/yolov8m-sku110k.onnx
```

```bash
curl -X POST http://localhost:3000/analyze -F "image=@/path/to/shelf-photo.jpg"
```
Expected: JSON with `count > 0` and `detections` array with bounding boxes.

**Step 5: Commit**

```bash
git add proto-c-rust-flutter/server/src/
git commit -m "feat(proto-c): implement YOLOv8 ONNX inference"
```

---

### Task 6: DETR model implementation

**Files:**
- Create: `proto-c-rust-flutter/server/src/ml/detr.rs`
- Modify: `proto-c-rust-flutter/server/src/ml/mod.rs`
- Modify: `proto-c-rust-flutter/server/src/main.rs`

**Prerequisite:** You need a DETR model fine-tuned on SKU-110K, exported to ONNX. Place at `proto-c-rust-flutter/server/models/detr-sku110k.onnx`. The exact input/output shapes depend on the model — inspect them at load time with `session.inputs()` / `session.outputs()`.

**Step 1: Create ml/detr.rs**

This is a skeleton — the postprocessing will need adjustment based on the actual ONNX output tensor names and shapes. The implementor should:
1. Load the model
2. Print input/output tensor names and shapes with `tracing::info!`
3. Adapt preprocessing (ImageNet normalization) and postprocessing accordingly

```rust
use std::path::Path;
use std::time::Instant;

use image::DynamicImage;
use ndarray::Array4;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::TensorRef;
use ort::inputs;

use crate::error::AppError;
use super::DetectionModel;
use super::types::{BoundingBox, Detection, DetectionResult};

const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];
const TARGET_SHORT_SIDE: u32 = 800;
const MAX_LONG_SIDE: u32 = 1333;
const CONFIDENCE_THRESHOLD: f64 = 0.5;

pub struct DetrModel {
    session: Session,
    name: String,
}

impl DetrModel {
    pub fn load(model_path: &Path) -> Result<Self, AppError> {
        let session = Session::builder()
            .map_err(|e| AppError::internal(format!("Failed to create session builder: {e}")))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| AppError::internal(format!("Failed to set optimization: {e}")))?
            .with_intra_threads(4)
            .map_err(|e| AppError::internal(format!("Failed to set threads: {e}")))?
            .commit_from_file(model_path)
            .map_err(|e| AppError::internal(format!("Failed to load model from {}: {e}", model_path.display())))?;

        // Log tensor info so implementor can adapt postprocessing
        for input in session.inputs.iter() {
            tracing::info!("DETR input: {} {:?}", input.name, input.input_type);
        }
        for output in session.outputs.iter() {
            tracing::info!("DETR output: {} {:?}", output.name, output.output_type);
        }

        let name = model_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("detr")
            .to_string();

        tracing::info!("Loaded DETR model: {} from {}", name, model_path.display());
        Ok(Self { session, name })
    }

    fn preprocess(&self, img: &DynamicImage) -> (Array4<f32>, u32, u32) {
        let (orig_w, orig_h) = (img.width(), img.height());

        // Resize: shortest side = 800, longest side <= 1333
        let (new_w, new_h) = if orig_w < orig_h {
            let new_w = TARGET_SHORT_SIDE;
            let new_h = (orig_h as f32 * TARGET_SHORT_SIDE as f32 / orig_w as f32) as u32;
            if new_h > MAX_LONG_SIDE {
                ((orig_w as f32 * MAX_LONG_SIDE as f32 / orig_h as f32) as u32, MAX_LONG_SIDE)
            } else {
                (new_w, new_h)
            }
        } else {
            let new_h = TARGET_SHORT_SIDE;
            let new_w = (orig_w as f32 * TARGET_SHORT_SIDE as f32 / orig_h as f32) as u32;
            if new_w > MAX_LONG_SIDE {
                (MAX_LONG_SIDE, (orig_h as f32 * MAX_LONG_SIDE as f32 / orig_w as f32) as u32)
            } else {
                (new_w, new_h)
            }
        };

        let resized = img.resize_exact(new_w, new_h, image::imageops::FilterType::Triangle);
        let rgb = resized.to_rgb8();

        // NCHW with ImageNet normalization
        let mut input = Array4::<f32>::zeros((1, 3, new_h as usize, new_w as usize));
        for y in 0..new_h as usize {
            for x in 0..new_w as usize {
                let pixel = rgb.get_pixel(x as u32, y as u32);
                input[[0, 0, y, x]] = (pixel[0] as f32 / 255.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
                input[[0, 1, y, x]] = (pixel[1] as f32 / 255.0 - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
                input[[0, 2, y, x]] = (pixel[2] as f32 / 255.0 - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
            }
        }

        (input, new_w, new_h)
    }
}

impl DetectionModel for DetrModel {
    fn name(&self) -> &str {
        &self.name
    }

    fn run(&self, image: DynamicImage) -> Result<DetectionResult, AppError> {
        let (input, _w, _h) = self.preprocess(&image);

        let start = Instant::now();

        let outputs = self.session.run(
            inputs![TensorRef::from_array_view(input.view())
                .map_err(|e| AppError::internal(format!("Failed to create input tensor: {e}")))?]
            .map_err(|e| AppError::internal(format!("Failed to create inputs: {e}")))?
        ).map_err(|e| AppError::internal(format!("Inference failed: {e}")))?;

        let inference_time_ms = start.elapsed().as_millis() as u64;

        // TODO: Adapt postprocessing based on actual ONNX output tensor names/shapes.
        // Standard DETR outputs:
        //   "pred_logits": [1, 100, num_classes+1] — class logits (last class = "no object")
        //   "pred_boxes":  [1, 100, 4] — normalized [cx, cy, w, h]
        //
        // For SKU-110K fine-tuned: num_classes is likely 1 (object) + 1 (no-object) = 2.
        // Apply softmax to logits, threshold on "object" class probability,
        // convert cx,cy,w,h to x,y,w,h.
        //
        // The implementor must inspect the actual output names (logged at startup)
        // and write the postprocessing accordingly.

        let detections: Vec<Detection> = Vec::new(); // Placeholder

        Ok(DetectionResult {
            model: self.name.clone(),
            count: detections.len(),
            detections,
            inference_time_ms,
        })
    }
}
```

**Step 2: Update ml/mod.rs**

Add `pub mod detr;` to the module declarations.

**Step 3: Update main.rs model selection**

Add `"detr"` arm to the match:
```rust
"detr" => {
    let m = ml::detr::DetrModel::load(Path::new(&args.model_path))
        .expect("Failed to load DETR model");
    Arc::new(m)
}
```

**Step 4: Test loading the DETR model**

```bash
cd proto-c-rust-flutter/server && cargo run -- --model detr --model-path models/detr-sku110k.onnx
```
Expected: Server starts, logs show DETR input/output tensor names and shapes. Use those to finish postprocessing.

**Step 5: Commit**

```bash
git add proto-c-rust-flutter/server/src/
git commit -m "feat(proto-c): add DETR model skeleton with ImageNet preprocessing"
```

---

## Part 2: Flutter Client

### Task 7: Scaffold Flutter project

**Files:**
- Create: Flutter project in `proto-c-rust-flutter/client/`
- Remove: `proto-c-rust-flutter/client/.gitkeep`

**Step 1: Create Flutter project**

```bash
cd proto-c-rust-flutter && rm client/.gitkeep
flutter create --project-name shelf_count_client client
```

**Step 2: Add dependencies to pubspec.yaml**

Add under `dependencies:`:
```yaml
  camera: ^0.11.0+2
  http: ^1.2.0
```

**Step 3: Run `flutter pub get`**

```bash
cd proto-c-rust-flutter/client && flutter pub get
```

**Step 4: Commit**

```bash
git add proto-c-rust-flutter/client/
git commit -m "chore(proto-c): scaffold Flutter client"
```

---

### Task 8: Config, models, and API service

**Files:**
- Create: `proto-c-rust-flutter/client/lib/config.dart`
- Create: `proto-c-rust-flutter/client/lib/models/detection.dart`
- Create: `proto-c-rust-flutter/client/lib/services/api_service.dart`

**Step 1: Create config.dart**

```dart
class Config {
  // For Android emulator use 10.0.2.2; for real device use your Mac's local IP
  static const String serverUrl = 'http://10.0.2.2:3000';
}
```

**Step 2: Create models/detection.dart**

```dart
class BoundingBox {
  final double x;
  final double y;
  final double width;
  final double height;

  const BoundingBox({
    required this.x,
    required this.y,
    required this.width,
    required this.height,
  });

  factory BoundingBox.fromJson(Map<String, dynamic> json) {
    return BoundingBox(
      x: (json['x'] as num).toDouble(),
      y: (json['y'] as num).toDouble(),
      width: (json['width'] as num).toDouble(),
      height: (json['height'] as num).toDouble(),
    );
  }
}

class Detection {
  final String label;
  final double confidence;
  final BoundingBox bbox;

  const Detection({
    required this.label,
    required this.confidence,
    required this.bbox,
  });

  factory Detection.fromJson(Map<String, dynamic> json) {
    return Detection(
      label: json['label'] as String,
      confidence: (json['confidence'] as num).toDouble(),
      bbox: BoundingBox.fromJson(json['bbox'] as Map<String, dynamic>),
    );
  }
}

class DetectionResult {
  final String model;
  final int count;
  final List<Detection> detections;
  final int inferenceTimeMs;

  const DetectionResult({
    required this.model,
    required this.count,
    required this.detections,
    required this.inferenceTimeMs,
  });

  factory DetectionResult.fromJson(Map<String, dynamic> json) {
    return DetectionResult(
      model: json['model'] as String,
      count: json['count'] as int,
      detections: (json['detections'] as List)
          .map((d) => Detection.fromJson(d as Map<String, dynamic>))
          .toList(),
      inferenceTimeMs: json['inference_time_ms'] as int,
    );
  }
}
```

**Step 3: Create services/api_service.dart**

```dart
import 'dart:convert';
import 'dart:typed_data';

import 'package:http/http.dart' as http;

import '../config.dart';
import '../models/detection.dart';

class ApiService {
  Future<DetectionResult> analyze(Uint8List imageBytes) async {
    final uri = Uri.parse('${Config.serverUrl}/analyze');
    final request = http.MultipartRequest('POST', uri)
      ..files.add(http.MultipartFile.fromBytes('image', imageBytes, filename: 'photo.jpg'));

    final streamedResponse = await request.send().timeout(
      const Duration(seconds: 10),
      onTimeout: () => throw Exception('Server unreachable (timeout)'),
    );

    final response = await http.Response.fromStream(streamedResponse);

    if (response.statusCode != 200) {
      final body = jsonDecode(response.body);
      throw Exception(body['error'] ?? 'Server error ${response.statusCode}');
    }

    return DetectionResult.fromJson(jsonDecode(response.body));
  }

  Future<bool> healthCheck() async {
    try {
      final uri = Uri.parse('${Config.serverUrl}/health');
      final response = await http.get(uri).timeout(const Duration(seconds: 3));
      return response.statusCode == 200;
    } catch (_) {
      return false;
    }
  }
}
```

**Step 4: Verify it compiles**

```bash
cd proto-c-rust-flutter/client && flutter analyze lib/config.dart lib/models/detection.dart lib/services/api_service.dart
```
Expected: No errors.

**Step 5: Commit**

```bash
git add proto-c-rust-flutter/client/lib/
git commit -m "feat(proto-c): add config, detection models, and API service"
```

---

### Task 9: Camera screen

**Files:**
- Create: `proto-c-rust-flutter/client/lib/screens/camera_screen.dart`

**Step 1: Create camera_screen.dart**

```dart
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

import '../models/detection.dart';
import '../services/api_service.dart';

class CameraScreen extends StatefulWidget {
  final void Function(String photoPath, DetectionResult result) onCaptureComplete;
  final void Function(String error) onError;

  const CameraScreen({
    super.key,
    required this.onCaptureComplete,
    required this.onError,
  });

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  CameraController? _controller;
  final _apiService = ApiService();
  bool _isProcessing = false;

  @override
  void initState() {
    super.initState();
    _init();
  }

  Future<void> _init() async {
    final cameras = await availableCameras();
    if (cameras.isEmpty) return;

    final backCamera = cameras.firstWhere(
      (c) => c.lensDirection == CameraLensDirection.back,
      orElse: () => cameras.first,
    );

    _controller = CameraController(backCamera, ResolutionPreset.high);
    await _controller!.initialize();
    if (mounted) setState(() {});
  }

  Future<void> _capture() async {
    if (_controller == null || !_controller!.value.isInitialized) return;
    if (_isProcessing) return;

    setState(() => _isProcessing = true);

    try {
      final photo = await _controller!.takePicture();
      final imageBytes = await photo.readAsBytes();
      final result = await _apiService.analyze(imageBytes);

      if (mounted) {
        widget.onCaptureComplete(photo.path, result);
      }
    } catch (e) {
      if (mounted) {
        widget.onError(e.toString());
      }
    } finally {
      if (mounted) setState(() => _isProcessing = false);
    }
  }

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final ready = _controller != null && _controller!.value.isInitialized;

    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        fit: StackFit.expand,
        children: [
          if (ready)
            CameraPreview(_controller!)
          else
            const Center(
              child: CircularProgressIndicator(color: Colors.white),
            ),
          Positioned(
            bottom: 40,
            left: 0,
            right: 0,
            child: Center(
              child: GestureDetector(
                onTap: ready && !_isProcessing ? _capture : null,
                child: Container(
                  width: 72,
                  height: 72,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    color: Colors.white.withValues(alpha: ready ? 0.3 : 0.1),
                  ),
                  child: Center(
                    child: _isProcessing
                        ? const CircularProgressIndicator(
                            color: Colors.white, strokeWidth: 3)
                        : Container(
                            width: 60,
                            height: 60,
                            decoration: BoxDecoration(
                              shape: BoxShape.circle,
                              color: ready
                                  ? Colors.white
                                  : Colors.white.withValues(alpha: 0.4),
                            ),
                          ),
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
```

**Step 2: Verify it compiles**

```bash
cd proto-c-rust-flutter/client && flutter analyze lib/screens/camera_screen.dart
```

**Step 3: Commit**

```bash
git add proto-c-rust-flutter/client/lib/screens/
git commit -m "feat(proto-c): add camera screen with server upload"
```

---

### Task 10: Results screen, bounding box overlay, and main app

**Files:**
- Create: `proto-c-rust-flutter/client/lib/widgets/bounding_box_overlay.dart`
- Create: `proto-c-rust-flutter/client/lib/screens/results_screen.dart`
- Modify: `proto-c-rust-flutter/client/lib/main.dart`

**Step 1: Create widgets/bounding_box_overlay.dart**

```dart
import 'package:flutter/material.dart';

import '../models/detection.dart';

class BoundingBoxOverlay extends StatelessWidget {
  final List<Detection> detections;
  final double viewWidth;
  final double viewHeight;

  const BoundingBoxOverlay({
    super.key,
    required this.detections,
    required this.viewWidth,
    required this.viewHeight,
  });

  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      size: Size(viewWidth, viewHeight),
      painter: _BoxPainter(detections: detections),
    );
  }
}

class _BoxPainter extends CustomPainter {
  final List<Detection> detections;

  _BoxPainter({required this.detections});

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.green
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2;

    final textStyle = TextStyle(
      color: Colors.black,
      fontSize: 10,
      fontWeight: FontWeight.bold,
      background: Paint()..color = Colors.green,
    );

    for (final det in detections) {
      final rect = Rect.fromLTWH(
        det.bbox.x * size.width,
        det.bbox.y * size.height,
        det.bbox.width * size.width,
        det.bbox.height * size.height,
      );

      canvas.drawRect(rect, paint);

      final textSpan = TextSpan(
        text: ' ${(det.confidence * 100).round()}% ',
        style: textStyle,
      );
      final textPainter = TextPainter(
        text: textSpan,
        textDirection: TextDirection.ltr,
      )..layout();

      textPainter.paint(canvas, Offset(rect.left, rect.top - textPainter.height));
    }
  }

  @override
  bool shouldRepaint(covariant _BoxPainter old) => detections != old.detections;
}
```

**Step 2: Create screens/results_screen.dart**

```dart
import 'dart:io';

import 'package:flutter/material.dart';

import '../models/detection.dart';
import '../widgets/bounding_box_overlay.dart';

class ResultsScreen extends StatelessWidget {
  final String photoPath;
  final DetectionResult result;
  final VoidCallback onScanAnother;

  const ResultsScreen({
    super.key,
    required this.photoPath,
    required this.result,
    required this.onScanAnother,
  });

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: SafeArea(
        child: Column(
          children: [
            Expanded(
              child: LayoutBuilder(
                builder: (context, constraints) {
                  return Stack(
                    alignment: Alignment.center,
                    children: [
                      Image.file(
                        File(photoPath),
                        width: constraints.maxWidth,
                        height: constraints.maxHeight,
                        fit: BoxFit.contain,
                      ),
                      BoundingBoxOverlay(
                        detections: result.detections,
                        viewWidth: constraints.maxWidth,
                        viewHeight: constraints.maxHeight,
                      ),
                    ],
                  );
                },
              ),
            ),
            Padding(
              padding: const EdgeInsets.all(20),
              child: Column(
                children: [
                  Text(
                    '${result.count} items detected',
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 32,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    'Model: ${result.model}',
                    style: const TextStyle(color: Colors.white70, fontSize: 14),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    'Inference: ${result.inferenceTimeMs}ms',
                    style: const TextStyle(color: Colors.grey, fontSize: 14),
                  ),
                ],
              ),
            ),
            Padding(
              padding: const EdgeInsets.only(bottom: 40),
              child: ElevatedButton(
                style: ElevatedButton.styleFrom(
                  backgroundColor: const Color(0xFF007AFF),
                  padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                ),
                onPressed: onScanAnother,
                child: const Text(
                  'Scan Another Shelf',
                  style: TextStyle(color: Colors.white, fontSize: 16, fontWeight: FontWeight.w600),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
```

**Step 3: Update main.dart**

```dart
import 'package:flutter/material.dart';

import 'models/detection.dart';
import 'screens/camera_screen.dart';
import 'screens/results_screen.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const ShelfCountApp());
}

class ShelfCountApp extends StatelessWidget {
  const ShelfCountApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Shelf Count',
      theme: ThemeData.dark(),
      home: const HomeScreen(),
    );
  }
}

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  String? _photoPath;
  DetectionResult? _result;
  String? _error;

  void _onCaptureComplete(String photoPath, DetectionResult result) {
    setState(() {
      _photoPath = photoPath;
      _result = result;
      _error = null;
    });
  }

  void _onError(String error) {
    setState(() {
      _error = error;
    });
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(error), backgroundColor: Colors.red),
    );
  }

  void _onScanAnother() {
    setState(() {
      _photoPath = null;
      _result = null;
      _error = null;
    });
  }

  @override
  Widget build(BuildContext context) {
    if (_photoPath != null && _result != null) {
      return ResultsScreen(
        photoPath: _photoPath!,
        result: _result!,
        onScanAnother: _onScanAnother,
      );
    }

    return CameraScreen(
      onCaptureComplete: _onCaptureComplete,
      onError: _onError,
    );
  }
}
```

**Step 4: Configure Android camera permission**

Add to `android/app/src/main/AndroidManifest.xml` inside `<manifest>`:
```xml
<uses-permission android:name="android.permission.CAMERA" />
<uses-permission android:name="android.permission.INTERNET" />
```

Also add `android:usesCleartextTraffic="true"` to the `<application>` tag (needed for HTTP to local dev server).

**Step 5: Verify it compiles**

```bash
cd proto-c-rust-flutter/client && flutter analyze
```

**Step 6: Commit**

```bash
git add proto-c-rust-flutter/client/
git commit -m "feat(proto-c): add results screen, overlay, and main app"
```

---

### Task 11: End-to-end test

**No new files.** Manual integration test.

**Step 1: Export ONNX model**

If you don't have the ONNX file yet, export from the training pipeline:
```bash
cd training && python export_tflite.py --format onnx
```
Or use your Colab notebook. Place the `.onnx` file in `proto-c-rust-flutter/server/models/`.

**Step 2: Start the Rust server**

```bash
cd proto-c-rust-flutter/server && cargo run -- --model-path models/yolov8m-sku110k.onnx
```

**Step 3: Update client config for real device**

If testing on a real phone (not emulator), update `Config.serverUrl` in `config.dart` to your Mac's local IP:
```dart
static const String serverUrl = 'http://192.168.x.x:3000';
```

**Step 4: Run Flutter client on Android**

```bash
cd proto-c-rust-flutter/client && flutter run
```

**Step 5: Test the flow**

1. App opens → camera viewfinder
2. Point at shelf → tap capture button
3. Loading spinner appears while uploading + server inference
4. Results screen shows: photo with green bounding boxes, count, model name, inference time
5. Tap "Scan Another Shelf" → back to camera

**Step 6: Verify health endpoint from client**

In debug console, confirm no network errors. Optionally add health check call on app startup.

**Step 7: Final commit**

```bash
git add -A
git commit -m "feat(proto-c): complete end-to-end server + client integration"
```

---

Plan complete and saved to `docs/plans/2026-03-05-proto-c-rust-flutter-plan.md`. Two execution options:

**1. Subagent-Driven (this session)** — I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** — Open new session with executing-plans, batch execution with checkpoints

Which approach?