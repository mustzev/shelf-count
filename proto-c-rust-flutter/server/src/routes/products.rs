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
        .ok_or_else(|| AppError::bad_request("CLIP model not loaded"))?;

    let mut sku: Option<String> = None;
    let mut embeddings: Vec<Vec<f32>> = Vec::new();

    while let Some(field) = multipart.next_field().await.map_err(|e| {
        AppError::bad_request(format!("Multipart error: {e}"))
    })? {
        match field.name() {
            Some("sku") => {
                sku = Some(field.text().await.map_err(|e| {
                    AppError::bad_request(format!("Failed to read sku: {e}"))
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

    let sku = sku.ok_or_else(|| AppError::bad_request("Missing 'sku' field"))?;
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
    for v in &mut avg {
        *v /= n;
    }
    let norm: f32 = avg.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in &mut avg {
            *v /= norm;
        }
    }

    let product = ProductRef { sku: sku.clone(), embedding: avg };
    state.product_db.add(product)?;

    tracing::info!("Registered product '{}' with {} reference images", sku, embeddings.len());

    Ok(Json(serde_json::json!({
        "status": "ok",
        "sku": sku,
        "images_processed": embeddings.len()
    })))
}

pub async fn remove_product(
    State(state): State<Arc<AppState>>,
    Path(sku): Path<String>,
) -> Result<Json<serde_json::Value>, AppError> {
    let removed = state.product_db.remove(&sku)?;
    if removed {
        Ok(Json(serde_json::json!({ "status": "ok", "removed": sku })))
    } else {
        Err(AppError::bad_request(format!("Product '{}' not found", sku)))
    }
}

pub async fn list_products(
    State(state): State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>, AppError> {
    let products = state.product_db.list()?;
    Ok(Json(serde_json::json!({ "products": products })))
}
