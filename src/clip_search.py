from utils import load_split_model, fetch_image
from typing import List, Tuple
import numpy as np
import open_clip
import torch
import json


class CLIPSearch:
    def __init__(self, model_path: str = "models", prefix: str = "clip_model_part_",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """Initialize CLIP model and tokenizer"""
        self.device = device
        self.preprocess, self.model = load_split_model()
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        self.model.to(device)
        self.model.eval()

    def search_images(self, query: str,
                      image_embedding_path: str = "embeddings/image_coco_validation_2014_embeddings.npy",
                      top_k: int = 5):
        """Return top-k similar image indices and their similarity scores"""
        image_embeddings = np.load(image_embedding_path)
        text_tokens = self.tokenizer([query]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (text_features.cpu().numpy() @ image_embeddings.T)[0]
            top_indices = similarity.argsort()[-top_k:][::-1]
            results = [(i, similarity[i]) for i in top_indices]

        return results

    def get_image_urls(self, top_k_indices: List[Tuple[int, float]],
                       file_urls: str = "embeddings/image_coco_validation_2014_coco_urls.npy"):
        """Get image URLs from indices"""
        file_urls = np.load(file_urls)
        return [(file_urls[index]) for index, similarity in top_k_indices]

    def search_images_by_url(self, query: str,
                             image_embedding_path: str = "embeddings/image_coco_validation_2014_embeddings.npy",
                             file_urls: str = "embeddings/image_coco_validation_2014_coco_urls.npy",
                             top_k: int = 5):
        """Return list of PIL Images from URLs"""
        top_k_indices = self.search_images(query, image_embedding_path, top_k)
        urls = self.get_image_urls(top_k_indices, file_urls)
        images = [fetch_image(url) for url in urls]
        return images, urls , top_k_indices
    def get_image_caption_annotations(self, top_k_indices: List[Tuple[int, float]],
                                      coco_annotations: str = "data/captions_val2014.json",
                                      file_names = "embeddings/image_coco_validation_2014_filenames.npy"):
        """Get image caption annotations from indices"""
        with open(coco_annotations, 'r') as f:
            coco_annotations = json.load(f)
        # Original sections
        file_names = np.load(file_names)
        file_names = [(file_names[index]) for index, similarity in top_k_indices]
        images = coco_annotations.get("images", [])
        filtered_images = []
        filtered_ids = []
        for item in images:
            if item["file_name"] in file_names:
                filtered_images.append(item)
                filtered_ids.append(item["id"])
        annotations = coco_annotations.get("annotations", [])
        filtered_annotations = [ann for ann in annotations if ann["image_id"] in filtered_ids]
        info = coco_annotations.get("info", {})
        licenses = coco_annotations.get("licenses", [])
        annotations = {"info": info,
                       "images": filtered_images,
                       "licenses": licenses,
                       "annotations": filtered_annotations}
        return annotations