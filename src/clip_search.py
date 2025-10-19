from utils import load_split_model, fetch_image
import torch
import open_clip
import numpy as np
import io
import os


class CLIPSearch:
    def __init__(self, model_path="models", prefix="clip_model_part_",
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.preprocess, self.model = load_split_model()
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        self.model.to(device)
        self.model.eval()

    def search_images_by_url(self, query,
                             image_embedding_path="embeddings/image_coco_validation_2014_embeddings.npy",
                             file_urls="embeddings/image_coco_validation_2014_coco_urls.npy",
                             top_k=5):
        # Load embeddings and filenames
        image_embeddings = np.load(image_embedding_path)
        file_urls = np.load(file_urls)
        text_tokens = self.tokenizer([query]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (text_features.cpu().numpy() @ image_embeddings.T)[0]
            top_indices = similarity.argsort()[-top_k:][::-1]
            results = [(file_urls[i], similarity[i]) for i in top_indices]
            
        urls = [url for url, _ in results]
        return [fetch_image(url) for url in urls], urls  # images, urls_state
