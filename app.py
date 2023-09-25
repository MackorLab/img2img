import gradio as gr
import modin.pandas as pd
import torch
import numpy as np
from PIL import Image
from diffusers import DiffusionPipeline


device = "cuda"
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe = pipe.to(device)
def resize(height, width, img):
    img = Image.open(img)
    img = img.resize((height, width))
    return img
def infer(source_img, prompt, negative_prompt, height, width, guide, steps, seed, strength):
    generator = torch.Generator(device).manual_seed(seed)
    source_image = resize(height, width, source_img)
    source_image.save('source.png')
    image = pipe(prompt, negative_prompt=negative_prompt, image=source_image, strength=strength, guidance_scale=guide, num_inference_steps=steps).images[0]
    return image
gr.Interface(fn=infer, inputs=[
        gr.Image(source="upload", type="filepath", label="Raw Image. Must Be .png"),
        gr.inputs.Textbox(label='Что вы хотите, чтобы ИИ генерировал'),
        gr.inputs.Textbox(label='Что вы не хотите, чтобы ИИ генерировал?', default='(deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, wrinkles, old face'),
        gr.Slider(512, 1024, 768, step=1, label='Ширина картинки'),
        gr.Slider(512, 1024, 768, step=1, label='Высота картинки'),
        gr.Slider(2, 15, value=7, label='Шкала навигации'),
        gr.Slider(1, 100, value=25, step=1, label='Количество итераций'),
        gr.Slider(label="Зерно", minimum=0, maximum=987654321987654321, step=1, randomize=True),
        gr.Slider(label='Сила', minimum=0, maximum=1, step=.05, value=.5),

    ],
     outputs='image', title = "DIAMONIK7777 - img2img - SDXL - Refiner",article = "<br><br><br><br><br>").launch(debug=True, max_threads=True, share=True, inbrowser=True)
