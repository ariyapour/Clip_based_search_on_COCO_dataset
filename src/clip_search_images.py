# src/search.py
import numpy as np
import torch
import open_clip
from utils import load_split_model

# Load embeddings and filenames
image_embeddings = np.load("embeddings/image_coco_validation_2014_embeddings.npy")
file_urls = np.load("embeddings/image_coco_validation_2014_coco_urls.npy")

device = "cuda" if torch.cuda.is_available() else "cpu"
preprocess, model = load_split_model()
tokenizer = open_clip.get_tokenizer('ViT-B-32')
model.to(device)
model.eval()

def search_images_by_url(query, top_k=5):
    text_tokens = tokenizer([query]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (text_features.cpu().numpy() @ image_embeddings.T)[0]
        top_indices = similarity.argsort()[-top_k:][::-1]
        results = [(file_urls[i], similarity[i]) for i in top_indices]
    return results
