use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;

use image::DynamicImage;
use ndarray::Array4;
use ort::inputs;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::TensorRef;

use super::DetectionModel;
use super::types::{BoundingBox, Detection, DetectionResult};
use crate::error::AppError;

const INPUT_SIZE: u32 = 640;
const NUM_BOXES: usize = 8400;
const CONFIDENCE_THRESHOLD: f32 = 0.25;
const IOU_THRESHOLD: f32 = 0.5;

struct Candidate {
    x: f32,
    y: f32,
    w: f32,
    h: f32,
    confidence: f32,
}

/// Letterbox metadata: how the image was padded to fit INPUT_SIZE x INPUT_SIZE.
struct LetterboxInfo {
    scale: f32,
    pad_x: f32,
    pad_y: f32,
}

pub struct YoloModel {
    session: Mutex<Session>,
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
            .map_err(|e| {
                AppError::internal(format!(
                    "Failed to load model from {}: {e}",
                    model_path.display()
                ))
            })?;

        let name = model_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("yolov8")
            .to_string();

        tracing::info!("Loaded YOLO model: {} from {}", name, model_path.display());
        Ok(Self { session: Mutex::new(session), name })
    }

    fn preprocess(&self, img: &DynamicImage) -> (Array4<f32>, LetterboxInfo) {
        let (orig_w, orig_h) = (img.width() as f32, img.height() as f32);
        let s = INPUT_SIZE as f32;

        // Scale to fit the longer side into INPUT_SIZE, preserving aspect ratio
        let scale = (s / orig_w).min(s / orig_h);
        let new_w = (orig_w * scale).round() as u32;
        let new_h = (orig_h * scale).round() as u32;

        let resized = img.resize_exact(new_w, new_h, image::imageops::FilterType::Triangle);
        let rgb = resized.to_rgb8();

        // Padding offset (center the resized image in the 640x640 canvas)
        let pad_x = ((s as u32 - new_w) / 2) as f32;
        let pad_y = ((s as u32 - new_h) / 2) as f32;

        // Fill with gray (114/255 is the YOLO letterbox convention)
        let mut input = Array4::<f32>::from_elem(
            (1, 3, INPUT_SIZE as usize, INPUT_SIZE as usize),
            114.0 / 255.0,
        );

        let px = pad_x as usize;
        let py = pad_y as usize;
        for y in 0..new_h as usize {
            for x in 0..new_w as usize {
                let pixel = rgb.get_pixel(x as u32, y as u32);
                input[[0, 0, py + y, px + x]] = pixel[0] as f32 / 255.0;
                input[[0, 1, py + y, px + x]] = pixel[1] as f32 / 255.0;
                input[[0, 2, py + y, px + x]] = pixel[2] as f32 / 255.0;
            }
        }

        let info = LetterboxInfo { scale, pad_x, pad_y };
        (input, info)
    }

    fn postprocess(&self, raw: &[f32], lb: &LetterboxInfo) -> Vec<Detection> {
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

        candidates.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

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
                // Undo letterbox: subtract padding, divide by scale → original pixel coords
                // Then normalize to 0–1 by dividing by original image dimensions
                let orig_x = (c.x - lb.pad_x) / lb.scale;
                let orig_y = (c.y - lb.pad_y) / lb.scale;
                let orig_w = c.w / lb.scale;
                let orig_h = c.h / lb.scale;
                let real_w = (INPUT_SIZE as f32 - 2.0 * lb.pad_x) / lb.scale;
                let real_h = (INPUT_SIZE as f32 - 2.0 * lb.pad_y) / lb.scale;

                Detection {
                    label: "object".to_string(),
                    confidence: c.confidence as f64,
                    bbox: BoundingBox {
                        x: (orig_x / real_w).clamp(0.0, 1.0) as f64,
                        y: (orig_y / real_h).clamp(0.0, 1.0) as f64,
                        width: (orig_w / real_w).clamp(0.0, 1.0) as f64,
                        height: (orig_h / real_h).clamp(0.0, 1.0) as f64,
                    },
                    sku: None,
                    sku_confidence: None,
                }
            })
            .collect()
    }
}

fn iou(a: &Candidate, b: &Candidate) -> f32 {
    let x1 = a.x.max(b.x);
    let y1 = a.y.max(b.y);
    let x2 = (a.x + a.w).min(b.x + b.w);
    let y2 = (a.y + a.h).min(b.y + b.h);

    if x2 <= x1 || y2 <= y1 {
        return 0.0;
    }

    let intersection = (x2 - x1) * (y2 - y1);
    let area_a = a.w * a.h;
    let area_b = b.w * b.h;
    let union = area_a + area_b - intersection;

    if union > 0.0 {
        intersection / union
    } else {
        0.0
    }
}

impl DetectionModel for YoloModel {
    fn name(&self) -> &str {
        &self.name
    }

    fn run(&self, image: DynamicImage) -> Result<DetectionResult, AppError> {
        let (input, letterbox) = self.preprocess(&image);

        let tensor = TensorRef::from_array_view(&input)
            .map_err(|e| AppError::internal(format!("Failed to create input tensor: {e}")))?;

        let start = Instant::now();
        let mut session = self
            .session
            .lock()
            .map_err(|e| AppError::internal(format!("Session lock poisoned: {e}")))?;
        let outputs = session
            .run(inputs![tensor])
            .map_err(|e| AppError::internal(format!("Inference failed: {e}")))?;

        let inference_time_ms = start.elapsed().as_millis() as u64;

        let predictions = outputs[0]
            .try_extract_array::<f32>()
            .map_err(|e| AppError::internal(format!("Failed to extract output: {e}")))?;

        let raw = predictions.as_slice().unwrap();
        let detections = self.postprocess(raw, &letterbox);

        Ok(DetectionResult {
            model: self.name.clone(),
            count: detections.len(),
            detections,
            inference_time_ms,
        })
    }
}
