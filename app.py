import gradio as gr
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
import torch
import random
import numpy as np
import spaces




if torch.cuda.is_available():
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16")

    unet = UNet2DConditionModel.from_pretrained("UCLA-AGI/SPIN-Diffusion-iter3", subfolder="unet", torch_dtype=torch.float16)
    pipe.unet = unet
    pipe = pipe.to("cuda")
else:
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32)

    unet = UNet2DConditionModel.from_pretrained("UCLA-AGI/SPIN-Diffusion-iter3", subfolder="unet", torch_dtype=torch.float32)
    pipe.unet = unet
    pipe = pipe.to("cpu")
    

@spaces.GPU(enable_queue=True)
def generate(prompt: str, num_images: int=5, guidance_scale=7.5):
    # Ensure num_images is an integer
    num_images = int(num_images)
    images = pipe(prompt, guidance_scale=guidance_scale, num_inference_steps=50, num_images_per_prompt=num_images).images
    images = [x.resize((512, 512)) for x in images]
    return images



with gr.Blocks() as demo:
    gr.Markdown("# SPIN-Diffusion 1.0 Demo")
    gr.Markdown("A **self-play** fine-tuned diffusion model from [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5), using winner images from the [yuvalkirstain/pickapic_v2](https://huggingface.co/datasets/yuvalkirstain/pickapic_v2) dataset.")
    with gr.Row():
        prompt_input = gr.Textbox(label="Enter your prompt", placeholder="Type something...", lines=2)
        generate_btn = gr.Button("Generate images")
    guidance_scale = gr.Slider(label="Guidance Scale", minimum=0, maximum=50, value=7.5, step=0.1)
    num_images_input = gr.Number(label="Number of images", value=5, minimum=1, maximum=10, step=1)
    gallery = gr.Gallery(label="Generated images", elem_id="gallery", columns=5, object_fit="contain")
    
    gr.Markdown("```If a generated image appears entirely black, it has been filtered out by the NSFW safety checker. Please try generating additional images.```")

    # Define your example prompts
    examples = [
        ["The Eiffel Tower at sunset"],
        ["A futuristic city skyline"],
        ["A cat wearing a wizard hat"],
        ["A futuristic city at sunset"],
        ["A landscape with mountains and lakes"],
        ["A portrait of a robot in Renaissance style"],
    ]
    
    # Add the Examples component linked to the prompt_input
    gr.Examples(examples=examples, inputs=prompt_input, fn=generate, outputs=gallery)
    
    generate_btn.click(fn=generate, inputs=[prompt_input, num_images_input, guidance_scale], outputs=gallery)
    
demo.queue().launch()
