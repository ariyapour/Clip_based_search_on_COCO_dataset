# src/app.py
import gradio as gr
from clip_search_images import search_images_by_url
from PIL import Image
import requests
from io import BytesIO
import zipfile
import tempfile
import os
from urllib.parse import urlparse

def search_by_url_and_display(query, top_k):
    """Return list of PIL Images from URLs"""
    results = search_images_by_url(query, top_k=top_k)
    urls = [url for url, _ in results]
    return [fetch_image(url) for url in urls], urls  # images, urls_state

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


with gr.Blocks() as demo:
    gr.Markdown("## CLIP Image Search for COCO Dataset.")

    query = gr.Textbox(label="Search Query")
    top_k = gr.Number(label="Number of Results", value=10, precision=0)  # user specifies top-k
    search_btn = gr.Button("🔍 Search")

    gallery = gr.Gallery(label="Top Matches", columns=5, show_label=True)
    urls_state = gr.State()

    download_file = gr.File(label="Download ZIP")
    download_btn = gr.Button("⬇️ Download Selected Images as ZIP")

    # Connect search: returns images + urls_state
    search_btn.click(
        fn=search_by_url_and_display,
        inputs=[query,top_k],
        outputs=[gallery, urls_state]
    )

    # Connect download: returns temporary ZIP path
    download_btn.click(
        fn=download_selected_images_as_zip,
        inputs=urls_state,
        outputs=download_file
    )

if __name__ == "__main__":
    demo.launch()