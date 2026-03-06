pub mod clip;
pub mod detr;
pub mod types;
pub mod yolo;

use image::DynamicImage;
use crate::error::AppError;
use types::DetectionResult;

pub trait DetectionModel: Send + Sync {
    fn name(&self) -> &str;
    fn run(&self, image: DynamicImage) -> Result<DetectionResult, AppError>;
}
