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

        tracing::info!("Loaded CLIP model from {}", model_path.display());
        Ok(Self { session: Mutex::new(session) })
    }

    /// Embed a single image crop. Returns a 512-dim L2-normalized vector.
    pub fn embed(&self, img: &DynamicImage) -> Result<Vec<f32>, AppError> {
        // 1. Resize to 224x224
        let resized = img.resize_exact(CLIP_SIZE, CLIP_SIZE, image::imageops::FilterType::Triangle);
        let rgb = resized.to_rgb8();

        // 2. Normalize with CLIP mean/std and lay out in NCHW format
        let mut input = Array4::<f32>::zeros((1, 3, CLIP_SIZE as usize, CLIP_SIZE as usize));

        for y in 0..CLIP_SIZE as usize {
            for x in 0..CLIP_SIZE as usize {
                let pixel = rgb.get_pixel(x as u32, y as u32);
                for c in 0..3 {
                    let normalized = (pixel[c] as f32 / 255.0 - CLIP_MEAN[c]) / CLIP_STD[c];
                    input[[0, c, y, x]] = normalized;
                }
            }
        }

        // 3. Run through ONNX session
        let tensor = TensorRef::from_array_view(input.view())
            .map_err(|e| AppError::internal(format!("Failed to create input tensor: {e}")))?;

        let mut session = self
            .session
            .lock()
            .map_err(|e| AppError::internal(format!("Session lock poisoned: {e}")))?;

        let outputs = session
            .run(inputs![tensor])
            .map_err(|e| AppError::internal(format!("CLIP inference failed: {e}")))?;

        // 4. Extract output — model outputs a 512-dim L2-normalized embedding
        let predictions = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| AppError::internal(format!("Failed to extract CLIP output: {e}")))?;

        let embedding: Vec<f32> = predictions.1.to_vec();
        Ok(embedding)
    }
}
