# app.py
import gradio as gr
from main import colorize_image_pil  # ✅ use the correct function

def colorize_gradio(input_img):
    return colorize_image_pil(input_img)  # ✅ call the correct function

demo = gr.Interface(
    fn=colorize_gradio,
    inputs=gr.Image(type="pil", label="Upload B&W Image"),
    outputs=gr.Image(type="pil", label="Colorized Image"),
    title="B&W to Color Image Colorizer",
    description="Upload a grayscale photo and see it colorized — no files saved!"
)

demo.launch()
