mod error;
mod ml;
mod products;
mod routes;

use std::env;
use std::sync::Arc;
use axum::{routing::get, Router};
use tower_http::cors::CorsLayer;

use ml::DetectionModel;

pub struct AppState {
    pub detection_model: Arc<dyn DetectionModel>,
    pub clip_model: Option<Arc<ml::clip::ClipModel>>,
    pub product_db: Arc<products::ProductDb>,
}

#[tokio::main]
async fn main() {
    dotenvy::dotenv().ok();
    tracing_subscriber::fmt::init();

    let model_type = env::var("MODEL").unwrap_or_else(|_| "yolo".to_string());
    let model_path = env::var("MODEL_PATH").unwrap_or_else(|_| "models/yolov8m-sku110k.onnx".to_string());
    let port: u16 = env::var("PORT").unwrap_or_else(|_| "3000".to_string()).parse().expect("PORT must be a number");

    let model: Arc<dyn DetectionModel> = match model_type.as_str() {
        "yolo" => {
            let m = ml::yolo::YoloModel::load(std::path::Path::new(&model_path))
                .expect("Failed to load YOLO model");
            Arc::new(m)
        }
        "detr" => {
            let m = ml::detr::DetrModel::load(std::path::Path::new(&model_path))
                .expect("Failed to load DETR model");
            Arc::new(m)
        }
        other => {
            eprintln!("Unknown model: {}. Supported: yolo, detr", other);
            std::process::exit(1);
        }
    };

    let clip_model = env::var("CLIP_MODEL_PATH").ok().map(|path| {
        let m = ml::clip::ClipModel::load(std::path::Path::new(&path))
            .expect("Failed to load CLIP model");
        Arc::new(m)
    });

    let product_db_path = env::var("PRODUCT_DB_PATH")
        .unwrap_or_else(|_| "data/products.json".to_string());
    let product_db = Arc::new(products::ProductDb::load(std::path::Path::new(&product_db_path)));

    let state = Arc::new(AppState {
        detection_model: model,
        clip_model,
        product_db,
    });

    let app = Router::new()
        .route("/health", get(routes::health::health))
        .route("/analyze", axum::routing::post(routes::analyze::analyze))
        .route("/products", axum::routing::get(routes::products::list_products))
        .route("/products", axum::routing::post(routes::products::add_product))
        .route("/products/{name}", axum::routing::delete(routes::products::remove_product))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr = format!("0.0.0.0:{}", port);
    tracing::info!("Listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
