# src/app.py
import gradio as gr
from clip_search_images import search_images_by_url
from PIL import Image
import requests
from io import BytesIO

def search_by_url_and_display(query):
    results = search_images_by_url(query, top_k=10)

    return [fetch_image(url) for url, _ in results]

# Function to fetch image from URL
def fetch_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content))
demo = gr.Interface(
    fn=search_by_url_and_display,
    inputs=gr.Textbox(label="Search Query"),
    outputs=gr.Gallery(label="Top Matches", columns=5, show_label=True),
    title="CLIP-Based Image Search",
    description="Type a text query to find visually relevant images from a small Flickr8k sample."
)

if __name__ == "__main__":
    demo.launch(allowed_paths=["/media/aref/DATA/datasets/coco/val2014"])
