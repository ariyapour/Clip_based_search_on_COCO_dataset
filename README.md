# CLIP-based Image Search on COCO (validation2014)

This application leverages CLIP embeddings of the COCO dataset to enable semantic image search using natural language queries. Users can enter a text query, and the app retrieves the most relevant images from the dataset.

For each search, the application generates a filtered subset of the COCO dataset, including:

- The selected images (displayed in a gallery)

- Their corresponding COCO-style annotations

Users can then download the results as a ZIP file, which contains:

- An images/ directory with the retrieved images

- An annotations/annotations.json file containing the annotations

This setup allows researchers and developers to quickly extract custom subsets of the COCO dataset for experimentation, fine-tuning, or other computer vision tasks based on semantic content rather than predefined categories.

## What this project contains

- `data/captions_val2014.json` — COCO captions/annotations for the val2014 split (used to map filenames to coco URLs).
- `embeddings/` — binary .npy files produced by the embedding script: `image_coco_validation_2014_embeddings.npy` and `image_coco_validation_2014_coco_urls.npy` and `embeddings/image_coco_validation_2014_filenames.npy`.
- `models/` — possibly-split CLIP model parts (named `clip_model_part_aa`, ...). `src/utils.py` contains helpers to merge these into a single in-memory model.
- `src/` — application source code:
  - `app.py` — Gradio UI to enter a text query, show a gallery of results and allow downloading selected images as a ZIP.
  - `clip_embed_images.py` — script to compute image embeddings for a local copy of COCO val2014 and save embeddings + COCO URLs.
  - `clip_search_images.py` — search routine that loads precomputed embeddings and runs text-to-image similarity using a CLIP model.
  - `utils.py` — helpers to assemble/load the split model parts and other small utilities.

## High-level architecture and dataflow

1. Compute image embeddings (offline):
	- `src/clip_embed_images.py` loads a CLIP model via `utils.load_split_model()`, processes local image files (val2014), and computes normalized image embeddings.
	- It saves two files under `embeddings/`: a NumPy array of embeddings and a matching list of COCO image URLs.
2. Query-time search (online):
	- `src/clip_search_images.py` loads the saved embeddings (`embeddings/*.npy`) and the CLIP text encoder. Given a text query, it encodes the query, computes cosine similarities with the stored image embeddings and returns the top-k image URLs and scores.
3. UI
	- `src/app.py` provides a Gradio interface. When a user submits a query, the backend calls the search function and displays the top-k images in a gallery. Selected images can be downloaded as a ZIP.

Why the split-model code: The `models/` directory may contain a large pretrained CLIP model file that was split into multiple parts for storage/transfer. `src/utils.py` shows a simple approach that concatenates those parts in memory and loads the PyTorch state dict.

## Quickstart (assumptions)

This project assumes you have a local copy of the COCO `val2014` images available (the embedding script expects a local image folder). It also expects either a saved CLIP model available in `models/` (possibly split into `clip_model_part_*`) or that you will use a local OpenCLIP model installation.

1. (Optional) Install requirements. There is no pinned requirements file in the repo, but the code uses the following packages:

	- Python 3.8+
	- torch
	- open_clip (or open_clip_torch / open_clip depending on your setup)
	- numpy
	- pillow
	- gradio
	- requests
	- tqdm

	Example pip install (adjust for your environment):

```bash
pip install torch numpy pillow gradio requests tqdm git+https://github.com/mlfoundations/open_clip.git
```

2. Prepare COCO images

	- Download the COCO val2014 images and put them in a directory (example path: `/path/to/val2014`). The embedding script expects `image_dir` to point at that folder.

3. Compute embeddings (CPU or GPU)

	- Edit `src/clip_embed_images.py` and set `image_dir` to your local path and `model_path` if needed. Then run:

```bash
python src/clip_embed_images.py
```

	This produces `embeddings/image_coco_validation_2014_embeddings.npy` and `embeddings/image_coco_validation_2014_coco_urls.npy`.

4. Run the Gradio demo

```bash
python src/app.py
```

Open the local URL printed by Gradio in your browser, enter a query, and inspect the gallery.

## Key implementation notes and project-specific patterns

- Model loading: `src/utils.py::load_split_model()` concatenates files named with the prefix `clip_model_part_` inside `models/` into a bytes buffer and then loads the PyTorch state dict from that buffer. This is intentionally memory-resident; be careful with very large models on low-RAM machines.
- Embeddings: `src/clip_embed_images.py` normalizes embeddings after encoding: `embedding /= embedding.norm(dim=-1, keepdim=True)`; search uses the same normalization on text features to compute cosine similarity via dot product.
- Search uses NumPy arrays for fast similarity computation: text feature (1 x D) dot image_embeddings.T yielding a vector of similarities.
- UI: `src/app.py` fetches images from their original COCO URLs at display/download time rather than storing local copies.

## Troubleshooting & tips

- If the `models/` folder contains split parts but `utils.load_split_model()` fails, ensure the parts are sorted and named consistently (`clip_model_part_aa`, `clip_model_part_ab`, ...).
- For quick experimentation without local models you can try using an installed CLIP checkpoint via `open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_something')` — adjust code accordingly.
- If running on CPU, embedding computation will be slow; prefer GPU when available.

## Files of interest (quick links)

- `src/clip_embed_images.py` — compute and save embeddings (offline).
- `src/clip_search_images.py` — run query-time search using embeddings.
- `src/utils.py` — model assembly helper for split files.
- `src/app.py` — Gradio frontend wiring.

## Contribution ideas / next steps

- Add a `requirements.txt` or `pyproject.toml` for reproducible installs.
- Add argument parsing to `src/clip_embed_images.py` for input/output paths and batch sizes.
- Add unit tests for `utils.load_split_model()` and `clip_search_images.search_images_by_url()`.

If any details (preferred install method, specific model checkpoints, or CI/test commands) are missing, tell me and I can add them to this README.
