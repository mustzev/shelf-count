use serde::{Deserialize, Serialize};

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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sku: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sku_confidence: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DetectionResult {
    pub model: String,
    pub count: usize,
    pub detections: Vec<Detection>,
    pub inference_time_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductRef {
    pub sku: String,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductStore {
    pub products: Vec<ProductRef>,
}
