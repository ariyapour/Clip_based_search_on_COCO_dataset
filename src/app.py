# src/app.py
from utils import download_selected_images_as_zip, download_images_with_annotations_as_zip
from clip_search import CLIPSearch
import gradio as gr

clip_engine = CLIPSearch()

def search_by_url_and_display(query, top_k):
    """Return list of PIL Images from URLs"""
    images, urls, top_k_indices  = clip_engine.search_images_by_url(query, top_k=top_k)
    annotations = clip_engine.get_image_caption_annotations(top_k_indices)
    return images, {"urls": urls, "annotations": annotations}

with gr.Blocks() as demo:
    gr.Markdown("## CLIP Image Search for COCO Dataset.")

    query = gr.Textbox(label="Search Query", value="People cooking in a kitchen")
    top_k = gr.Number(label="Number of files to get", value=10, precision=0)  # user specifies top-k
    search_btn = gr.Button("🔍 Search")

    gallery = gr.Gallery(label="Top Matches", columns=5, show_label=True)
    urls_state = gr.State()
    
    gr.HTML("""
    <style>
    .small-download {
        width: 100%;
        
        max-height: 250px;
        height: 60px;
    }
    </style>
    """)  # inject CSS at the top

    download_file = gr.File(
        label="Download ZIP",
        file_types=[".zip"],
        elem_classes="small-download"
    )
    
    download_btn = gr.Button("⬇️ Download Selected subset of dataset with Annotations")

    # Connect search: returns images + urls_state
    search_btn.click(
        fn=search_by_url_and_display,
        inputs=[query,top_k],
        outputs=[gallery, urls_state]
    )

    # Connect download: returns temporary ZIP path
    download_btn.click(
        fn=lambda state: download_images_with_annotations_as_zip(
            urls=state.get("urls", []),
            annotations_dict=state.get("annotations", {})
        ),
        inputs=urls_state,
        outputs=download_file
    )

if __name__ == "__main__":
    demo.launch()