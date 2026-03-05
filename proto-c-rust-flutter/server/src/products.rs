use std::path::{Path, PathBuf};
use std::sync::RwLock;

use crate::error::AppError;
use crate::ml::types::{ProductRef, ProductStore};

const SIMILARITY_THRESHOLD: f32 = 0.6;

pub struct ProductDb {
    store: RwLock<ProductStore>,
    path: PathBuf,
}

impl ProductDb {
    pub fn load(path: &Path) -> Self {
        let store = if path.exists() {
            match std::fs::read_to_string(path) {
                Ok(contents) => match serde_json::from_str::<ProductStore>(&contents) {
                    Ok(s) => {
                        tracing::info!("Loaded {} product(s) from {}", s.products.len(), path.display());
                        s
                    }
                    Err(e) => {
                        tracing::warn!("Failed to parse product DB at {}: {}. Starting empty.", path.display(), e);
                        ProductStore { products: Vec::new() }
                    }
                },
                Err(e) => {
                    tracing::warn!("Failed to read product DB at {}: {}. Starting empty.", path.display(), e);
                    ProductStore { products: Vec::new() }
                }
            }
        } else {
            tracing::info!("No product DB found at {}. Starting empty.", path.display());
            ProductStore { products: Vec::new() }
        };

        Self {
            store: RwLock::new(store),
            path: path.to_path_buf(),
        }
    }

    pub fn add(&self, product: ProductRef) -> Result<(), AppError> {
        let mut store = self
            .store
            .write()
            .map_err(|e| AppError::internal(format!("Product DB lock poisoned: {}", e)))?;

        // Replace existing entry with the same name, or append.
        if let Some(existing) = store.products.iter_mut().find(|p| p.sku == product.sku) {
            *existing = product;
        } else {
            store.products.push(product);
        }

        self.save(&store)
    }

    pub fn remove(&self, sku: &str) -> Result<bool, AppError> {
        let mut store = self
            .store
            .write()
            .map_err(|e| AppError::internal(format!("Product DB lock poisoned: {}", e)))?;

        let before = store.products.len();
        store.products.retain(|p| p.sku != sku);
        let removed = store.products.len() < before;

        if removed {
            self.save(&store)?;
        }

        Ok(removed)
    }

    pub fn list(&self) -> Result<Vec<String>, AppError> {
        let store = self
            .store
            .read()
            .map_err(|e| AppError::internal(format!("Product DB lock poisoned: {}", e)))?;

        Ok(store.products.iter().map(|p| p.sku.clone()).collect())
    }

    pub fn find_closest(&self, embedding: &[f32]) -> Result<Option<(String, f32)>, AppError> {
        let store = self
            .store
            .read()
            .map_err(|e| AppError::internal(format!("Product DB lock poisoned: {}", e)))?;

        if store.products.is_empty() {
            return Ok(None);
        }

        let best = store
            .products
            .iter()
            .map(|p| {
                let sim = cosine_similarity(embedding, &p.embedding);
                (p.sku.clone(), sim)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        match best {
            Some((name, sim)) if sim >= SIMILARITY_THRESHOLD => Ok(Some((name, sim))),
            _ => Ok(None),
        }
    }

    fn save(&self, store: &ProductStore) -> Result<(), AppError> {
        // Create parent directories if they don't exist.
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| AppError::internal(format!("Failed to create directories for {}: {}", self.path.display(), e)))?;
        }

        let json = serde_json::to_string_pretty(store)
            .map_err(|e| AppError::internal(format!("Failed to serialize product DB: {}", e)))?;

        std::fs::write(&self.path, json)
            .map_err(|e| AppError::internal(format!("Failed to write product DB to {}: {}", self.path.display(), e)))?;

        Ok(())
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }

    let len = a.len().min(b.len());
    let dot: f32 = a[..len].iter().zip(b[..len].iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a[..len].iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b[..len].iter().map(|x| x * x).sum::<f32>().sqrt();

    if mag_a == 0.0 || mag_b == 0.0 {
        return 0.0;
    }

    dot / (mag_a * mag_b)
}
