from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
import open_clip
import torch
import json
import os

with open('data/captions_val2014.json') as f:
    coco_validation_2014_annotatiaons = json.load(f)

#Images can be downloaded from this link: http://images.cocodataset.org/zips/val2014.zip
image_dir = "PAth_to_the_coco_validataion_2014"

#Path to clip model
model_path = "models/clip-vit-b-32/models--timm--vit_base_patch32_clip_224.openai"
#Define lists to save image embeddings and image urls
image_embeddings = []
file_coco_urls = []

#load clip-vit-b-32
model_cache_dir = Path("models/clip-vit-b-32")
model_cache_dir.mkdir(parents=True, exist_ok=True)
#load clip model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained=None)
model.load_state_dict(torch.load("models/clip-vit-b-32/clip-vit-b-32.pt"))
tokenizer = open_clip.get_tokenizer('ViT-B-32')
model.to(device)
model.eval()

#load a list of available images
images = os.listdir(image_dir)

# Compute embeddings
with torch.no_grad():
    for index in tqdm(range(len(images))):
        if images[index].lower().endswith((".jpg", ".png")):
            path = os.path.join(image_dir, images[index])
            for item in coco_validation_2014_annotatiaons["images"]:
                if item["file_name"] == images[index]:
                    file_coco_urls.append(item["coco_url"])
                    break
            image = preprocess(Image.open(path)).unsqueeze(0).to(device)
            embedding = model.encode_image(image)
            embedding /= embedding.norm(dim=-1, keepdim=True)
            image_embeddings.append(embedding.cpu().numpy())

image_embeddings = np.vstack(image_embeddings)
#save each image embedding and filenames on disk
np.save("embeddings/image_coco_validation_2014_embeddings.npy", image_embeddings)
np.save("embeddings/image_coco_validation_2014_coco_urls.npy", file_coco_urls)
