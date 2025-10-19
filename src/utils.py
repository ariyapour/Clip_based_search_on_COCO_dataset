from urllib.parse import urlparse
from io import BytesIO
from PIL import Image
import numpy as np
import open_clip
import requests
import tempfile
import zipfile
import torch
import os
import io

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

def fetch_image(url):
    response = requests.get(url, timeout=10)
    return Image.open(BytesIO(response.content))

def download_selected_images_as_zip(urls):
    """Create a temporary ZIP with original filenames"""
    if not urls:
        return None
    temp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(temp_dir, "images.zip")

    with zipfile.ZipFile(zip_path, "w") as zf:
        for url in urls:
            try:
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200:
                    # Extract original filename from URL
                    filename = os.path.basename(urlparse(url).path)
                    zf.writestr(filename, resp.content)
            except Exception as e:
                print(f"Failed to download {url}: {e}")
    return zip_path
