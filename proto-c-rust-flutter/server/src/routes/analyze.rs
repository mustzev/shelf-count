use axum::extract::{Multipart, State};
use axum::Json;
use std::sync::Arc;

use image::DynamicImage;

use crate::error::AppError;
use crate::ml::types::DetectionResult;

/// Read EXIF orientation from JPEG bytes and apply rotation/flip to the decoded image.
fn apply_exif_orientation(bytes: &[u8], img: DynamicImage) -> DynamicImage {
    let orientation = (|| -> Option<u32> {
        let exif = exif::Reader::new()
            .read_from_container(&mut std::io::Cursor::new(bytes))
            .ok()?;
        let orient = exif.get_field(exif::Tag::Orientation, exif::In::PRIMARY)?;
        orient.value.get_uint(0)
    })();

    match orientation {
        Some(2) => img.fliph(),
        Some(3) => img.rotate180(),
        Some(4) => img.flipv(),
        Some(5) => img.rotate90().fliph(),
        Some(6) => img.rotate90(),
        Some(7) => img.rotate270().fliph(),
        Some(8) => img.rotate270(),
        _ => img, // 1 or unknown = no rotation
    }
}

pub async fn analyze(
    State(state): State<Arc<crate::AppState>>,
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

    // Decode image and apply EXIF orientation
    let img = image::load_from_memory(&bytes)
        .map_err(|e| AppError::bad_request(format!("Failed to decode image: {e}")))?;
    let img = apply_exif_orientation(&bytes, img);

    tracing::info!("Image after EXIF correction: {}x{}", img.width(), img.height());

    // Run inference
    let mut result = state.detection_model.run(img.clone())?;

    // Classify each detection if CLIP model is available
    if let Some(clip) = &state.clip_model {
        let (img_w, img_h) = (img.width(), img.height());

        for det in &mut result.detections {
            // Crop bounding box from original image (coords are normalized 0-1)
            let x = (det.bbox.x * img_w as f64) as u32;
            let y = (det.bbox.y * img_h as f64) as u32;
            let w = (det.bbox.width * img_w as f64) as u32;
            let h = (det.bbox.height * img_h as f64) as u32;

            // Clamp to image bounds
            let x = x.min(img_w.saturating_sub(1));
            let y = y.min(img_h.saturating_sub(1));
            let w = w.min(img_w - x).max(1);
            let h = h.min(img_h - y).max(1);

            let crop = img.crop_imm(x, y, w, h);
            let embedding = clip.embed(&crop)?;

            if let Some((name, sim)) = state.product_db.find_closest(&embedding)? {
                det.sku = Some(name);
                det.sku_confidence = Some(sim as f64);
            }
        }
    }

    Ok(Json(result))
}
