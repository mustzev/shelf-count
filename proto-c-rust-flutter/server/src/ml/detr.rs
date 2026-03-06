use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;

use image::DynamicImage;
use ndarray::{Array3, Array4};
use ort::inputs;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::TensorRef;

use super::DetectionModel;
use super::types::{BoundingBox, Detection, DetectionResult};
use crate::error::AppError;

const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];
const TARGET_SHORT_SIDE: u32 = 800;
const MAX_LONG_SIDE: u32 = 1333;
const CONFIDENCE_THRESHOLD: f64 = 0.5;

pub struct DetrModel {
    session: Mutex<Session>,
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
            .map_err(|e| {
                AppError::internal(format!(
                    "Failed to load model from {}: {e}",
                    model_path.display()
                ))
            })?;

        // Log tensor info so the user can adapt postprocessing
        for input in session.inputs().iter() {
            tracing::info!("DETR input: {}", input.name());
        }
        for output in session.outputs().iter() {
            tracing::info!("DETR output: {}", output.name());
        }

        let name = model_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("detr")
            .to_string();

        tracing::info!("Loaded DETR model: {} from {}", name, model_path.display());
        Ok(Self {
            session: Mutex::new(session),
            name,
        })
    }

    fn preprocess(&self, img: &DynamicImage) -> (Array4<f32>, Array3<i64>) {
        let (orig_w, orig_h) = (img.width(), img.height());

        // Resize: shortest side = 800, longest side <= 1333
        let (new_w, new_h) = if orig_w < orig_h {
            let new_w = TARGET_SHORT_SIDE;
            let new_h = (orig_h as f32 * TARGET_SHORT_SIDE as f32 / orig_w as f32) as u32;
            if new_h > MAX_LONG_SIDE {
                (
                    (orig_w as f32 * MAX_LONG_SIDE as f32 / orig_h as f32) as u32,
                    MAX_LONG_SIDE,
                )
            } else {
                (new_w, new_h)
            }
        } else {
            let new_h = TARGET_SHORT_SIDE;
            let new_w = (orig_w as f32 * TARGET_SHORT_SIDE as f32 / orig_h as f32) as u32;
            if new_w > MAX_LONG_SIDE {
                (
                    MAX_LONG_SIDE,
                    (orig_h as f32 * MAX_LONG_SIDE as f32 / orig_w as f32) as u32,
                )
            } else {
                (new_w, new_h)
            }
        };

        let resized =
            img.resize_exact(new_w, new_h, image::imageops::FilterType::Triangle);
        let rgb = resized.to_rgb8();

        // NCHW with ImageNet normalization
        let mut pixel_values = Array4::<f32>::zeros((1, 3, new_h as usize, new_w as usize));
        for y in 0..new_h as usize {
            for x in 0..new_w as usize {
                let pixel = rgb.get_pixel(x as u32, y as u32);
                pixel_values[[0, 0, y, x]] =
                    (pixel[0] as f32 / 255.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
                pixel_values[[0, 1, y, x]] =
                    (pixel[1] as f32 / 255.0 - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
                pixel_values[[0, 2, y, x]] =
                    (pixel[2] as f32 / 255.0 - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
            }
        }

        // pixel_mask: all 1s (no padding)
        let pixel_mask = Array3::<i64>::ones((1, new_h as usize, new_w as usize));

        (pixel_values, pixel_mask)
    }

    fn postprocess(&self, logits: &[f32], boxes: &[f32], num_queries: usize, num_classes: usize) -> Vec<Detection> {
        let mut detections = Vec::new();

        for q in 0..num_queries {
            // Softmax over class logits for this query
            let logit_offset = q * num_classes;
            let logit_slice = &logits[logit_offset..logit_offset + num_classes];

            let max_logit = logit_slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = logit_slice.iter().map(|&l| (l - max_logit).exp()).sum();
            let probs: Vec<f32> = logit_slice.iter().map(|&l| (l - max_logit).exp() / exp_sum).collect();

            // Class 0 = background (LABEL_0), Class 1 = SKU Item, Class 2 = no-object
            // The "no-object" class is the last one. We want the max non-background, non-no-object score.
            // For this model: id2label has {0: "LABEL_0", 1: "SKU Item"}, so num_labels=2, num_classes=3
            // Index 0 = LABEL_0 (background), Index 1 = SKU Item, Index 2 = no-object
            // We care about index 1 (SKU Item)
            let sku_score = if num_classes >= 3 { probs[1] } else { probs[0] };

            if (sku_score as f64) < CONFIDENCE_THRESHOLD {
                continue;
            }

            // Boxes are [cx, cy, w, h] normalized 0–1
            let box_offset = q * 4;
            let cx = boxes[box_offset];
            let cy = boxes[box_offset + 1];
            let w = boxes[box_offset + 2];
            let h = boxes[box_offset + 3];

            detections.push(Detection {
                label: "object".to_string(),
                confidence: sku_score as f64,
                bbox: BoundingBox {
                    x: (cx - w / 2.0).clamp(0.0, 1.0) as f64,
                    y: (cy - h / 2.0).clamp(0.0, 1.0) as f64,
                    width: w.clamp(0.0, 1.0) as f64,
                    height: h.clamp(0.0, 1.0) as f64,
                },
                sku: None,
                sku_confidence: None,
            });
        }

        detections
    }
}

impl DetectionModel for DetrModel {
    fn name(&self) -> &str {
        &self.name
    }

    fn run(&self, image: DynamicImage) -> Result<DetectionResult, AppError> {
        let (pixel_values, pixel_mask) = self.preprocess(&image);

        let pv_tensor = TensorRef::from_array_view(&pixel_values)
            .map_err(|e| AppError::internal(format!("Failed to create pixel_values tensor: {e}")))?;
        let pm_tensor = TensorRef::from_array_view(&pixel_mask)
            .map_err(|e| AppError::internal(format!("Failed to create pixel_mask tensor: {e}")))?;

        let start = Instant::now();

        let mut session = self.session.lock().map_err(|e| {
            AppError::internal(format!("Failed to lock session: {e}"))
        })?;

        let outputs = session
            .run(inputs![pv_tensor, pm_tensor])
            .map_err(|e| AppError::internal(format!("Inference failed: {e}")))?;

        let inference_time_ms = start.elapsed().as_millis() as u64;

        // logits: [1, num_queries, num_classes]  pred_boxes: [1, num_queries, 4]
        let logits_data = outputs[0]
            .try_extract_array::<f32>()
            .map_err(|e| AppError::internal(format!("Failed to extract logits: {e}")))?;
        let boxes_data = outputs[1]
            .try_extract_array::<f32>()
            .map_err(|e| AppError::internal(format!("Failed to extract boxes: {e}")))?;

        let logits_shape = logits_data.shape();
        let num_queries = logits_shape[1];
        let num_classes = logits_shape[2];

        tracing::debug!("DETR output: {} queries, {} classes", num_queries, num_classes);

        let detections = self.postprocess(logits_data.as_slice().unwrap(), boxes_data.as_slice().unwrap(), num_queries, num_classes);

        Ok(DetectionResult {
            model: self.name.clone(),
            count: detections.len(),
            detections,
            inference_time_ms,
        })
    }
}
