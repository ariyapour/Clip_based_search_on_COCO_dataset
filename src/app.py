# src/app.py
from utils import download_selected_images_as_zip
from clip_search import CLIPSearch
import gradio as gr

clip_engine = CLIPSearch()

def search_by_url_and_display(query, top_k):
    """Return list of PIL Images from URLs"""
    results = clip_engine.search_images_by_url(query, top_k=top_k)
    return results

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