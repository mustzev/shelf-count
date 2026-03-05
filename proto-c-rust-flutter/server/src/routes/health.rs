use axum::extract::State;
use axum::Json;
use serde_json::{json, Value};
use std::sync::Arc;

pub async fn health(State(state): State<Arc<crate::AppState>>) -> Json<Value> {
    Json(json!({
        "status": "ok",
        "model": state.detection_model.name()
    }))
}
