import gradio as gr

CUSTOM_CSS = """
    .audio {
        height: 200px !important;
    }
"""

with gr.Blocks() as demo:
    gr.Markdown("<center><h1>Unified Text and Audio Control for Music Mixing Style Transfer</h1></center>")
    
    gr.Markdown("")
    gr.Markdown("<h2>Upload Raw Tracks</h2>")
    with gr.Row():
        with gr.Column():
            raw_file_1 = gr.Audio(label="Raw Track 1", type="filepath", sources="upload", elem_classes="audio")
            raw_file_5 = gr.Audio(label="Raw Track 5", type="filepath", sources="upload", elem_classes="audio")
        with gr.Column():
            raw_file_2 = gr.Audio(label="Raw Track 2", type="filepath", sources="upload", elem_classes="audio")
            raw_file_6 = gr.Audio(label="Raw Track 6", type="filepath", sources="upload", elem_classes="audio")
        with gr.Column():
            raw_file_3 = gr.Audio(label="Raw Track 3", type="filepath", sources="upload", elem_classes="audio")
            raw_file_7 = gr.Audio(label="Raw Track 7", type="filepath", sources="upload", elem_classes="audio")
        with gr.Column():
            raw_file_4 = gr.Audio(label="Raw Track 4", type="filepath", sources="upload", elem_classes="audio")
            raw_file_8 = gr.Audio(label="Raw Track 8", type="filepath", sources="upload", elem_classes="audio")

    gr.Markdown("")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("<h2>Audio Control</h2>")
            ref_file = gr.Audio(label="Ref song", type="filepath", sources="upload", elem_classes="audio")
            gr.Slider(label="Ref Start Time (s)", minimum=0, maximum=180, step=0.1)
            gr.Button("Generate Mixing with Audio Control")
        with gr.Column():
            gr.Markdown("<h2>Text Control</h2>")
            with gr.Row():
                gr.Button("Brightness")
                gr.Button("Compressed")
                gr.Button("Loudness")
                gr.Button("Panning")
            gr.Dropdown(
                choices=["Track 1", "Track 2", "Track 3", "Track 4", "Track 5", "Track 6", "Track 7", "Track 8", "Mastering"], 
                label="Modified Tracks", interactive=True
            )
            gr.Slider(label="Text Control Strength", minimum=0, maximum=100, step=0.01)
            gr.Slider(label="Style Strength", minimum=0, maximum=100, step=0.01)
            gr.Button("Generate Mixing with Text Control")

    gr.Markdown("<h2>Output Mixed Track</h2>")
    with gr.Row():
        output_audio = gr.Audio(label="Output Mixed Track", type="filepath", sources="upload", elem_classes="audio")

demo.launch(theme=gr.themes.Default(), css=CUSTOM_CSS)