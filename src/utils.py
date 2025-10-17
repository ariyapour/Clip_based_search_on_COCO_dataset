import os
import io
import torch
import open_clip

def merge_model_parts(model_path="models", output_file="clip-vit-b-32.pt", part_prefix="clip_model_part_"):
    parts = sorted([p for p in os.listdir('.') if p.startswith(part_prefix)])
    with open(output_file, 'wb') as outfile:
        for part in parts:
            with open(part, 'rb') as infile:
                outfile.write(infile.read())
    print(f"Merged {len(parts)} parts into {output_file}")


def load_split_model(model_path="models", prefix="clip_model_part_"):
    # Get chunk file names in correct order
    parts = sorted([p for p in os.listdir(model_path) if p.startswith(prefix)])
    print(f"Found {len(parts)} parts: {parts}")

    # Merge all parts in memory
    model_bytes = b""
    for part in parts:
        with open(os.path.join(model_path, part), 'rb') as f:
            model_bytes += f.read()

    # Load model directly from in-memory bytes
    buffer = io.BytesIO(model_bytes)
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained=None)
    model.load_state_dict(torch.load(buffer))

    print("Model loaded successfully from memory!")
    return preprocess, model