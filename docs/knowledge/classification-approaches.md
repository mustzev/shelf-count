# Classification Approaches for Product Identification

## Context

After detection (finding *where* products are on a shelf), classification identifies *what* each product is. The pipeline: detect → crop each bounding box → classify the crop.

## Approach A: Embedding Model + Nearest-Neighbor Lookup (Chosen)

Use a pre-trained image embedding model (e.g., CLIP, EfficientNet) to extract a feature vector from each cropped product. Store reference embeddings for known SKUs in a database. At inference time, find the closest match.

- **Pros**: No retraining when adding new products — just add reference images. Works with 1–5 images per SKU. Scales from 10 to 10,000+ SKUs naturally.
- **Cons**: Slightly less accurate than a dedicated classifier for small SKU counts. Needs a vector similarity search.
- **Data needed**: 1–5 reference images per product

## Approach B: Traditional Classifier (CNN Fine-Tuned)

Fine-tune EfficientNet or ResNet on a closed set of product classes. Each SKU is a class. Standard image classification.

- **Pros**: Very accurate when you have enough data and a fixed product set.
- **Cons**: Must retrain every time a product is added/removed. Needs 20–50+ images per SKU. Doesn't scale well — adding one product means retraining the whole model.
- **Data needed**: 20–50 images per product

## Approach C: Two-Stage — Embedding + Classifier

Use embeddings for initial matching, then a lightweight classifier to refine among similar-looking products.

- **Pros**: Best accuracy for confusable products (e.g., same brand, different size).
- **Cons**: Most complex. Overkill until you have thousands of SKUs with many visually similar items.

## Why We Chose A

- SKU count is unknown and will grow over time
- Adding new products should not require retraining
- Minimal data requirement (1–5 reference images per product)
- Can evolve to Approach C later if accuracy on similar products becomes an issue
