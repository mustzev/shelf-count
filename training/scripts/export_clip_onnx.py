"""Export CLIP ViT-B/32 visual encoder to ONNX.

Only exports the image encoder (not text encoder).
Output: 512-dim L2-normalized embedding per image.
"""

import torch
from transformers import CLIPModel


class VisualEncoder(torch.nn.Module):
    def __init__(self, vision_model, visual_projection):
        super().__init__()
        self.vision_model = vision_model
        self.visual_projection = visual_projection

    def forward(self, pixel_values):
        outputs = self.vision_model(pixel_values=pixel_values)
        pooled = outputs.pooler_output
        embeds = self.visual_projection(pooled)
        embeds = embeds / embeds.norm(dim=-1, keepdim=True)
        return embeds


def main():
    print("Loading CLIP ViT-B/32...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    encoder = VisualEncoder(model.vision_model, model.visual_projection)
    encoder.eval()

    dummy = torch.randn(1, 3, 224, 224)

    output_path = "models/clip-vit-b32-visual.onnx"
    print(f"Exporting to {output_path}...")

    torch.onnx.export(
        encoder,
        dummy,
        output_path,
        input_names=["pixel_values"],
        output_names=["embeddings"],
        dynamic_axes={
            "pixel_values": {0: "batch"},
            "embeddings": {0: "batch"},
        },
        opset_version=17,
    )

    import os
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Done! {output_path} ({size_mb:.1f} MB)")
    print("Input:  pixel_values [batch, 3, 224, 224]")
    print("Output: embeddings   [batch, 512] (L2-normalized)")


if __name__ == "__main__":
    main()
