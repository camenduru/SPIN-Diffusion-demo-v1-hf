import gradio as gr
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
import torch
import random
import numpy as np
import spaces


MODEL="UCLA-AGI/SPIN-Diffusion-iter3"

def set_seed(seed=5775709):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


set_seed()



def get_pipeline(device='cuda'):
    model_id = "runwayml/stable-diffusion-v1-5"
    #pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker = None, requires_safety_checker = False)
    if torch.cuda.is_available():
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)

        # load finetuned model
        unet_id = MODEL
        unet = UNet2DConditionModel.from_pretrained(unet_id, subfolder="unet", torch_dtype=torch.float32)
        pipe.unet = unet
        pipe = pipe.to(device)
        return pipe

@spaces.GPU(enable_queue=True)
def generate(prompt: str, num_images: int=5, guidance_scale=7.5):
    pipe = get_pipeline()
    generator = torch.Generator(pipe.device).manual_seed(5775709)
    # Ensure num_images is an integer
    num_images = int(num_images)
    images = pipe(prompt, generator=generator, guidance_scale=guidance_scale, num_inference_steps=50, num_images_per_prompt=num_images).images
    images = [x.resize((512, 512)) for x in images]
    return images



with gr.Blocks() as demo:
    gr.Markdown("# SPIN-Diffusion 1.0 Demo")

    with gr.Row():
        prompt_input = gr.Textbox(label="Enter your prompt", placeholder="Type something...", lines=2)
        generate_btn = gr.Button("Generate images")
    guidance_scale = gr.Slider(label="Guidance Scale", minimum=0, maximum=50, value=9, step=0.1)
    num_images_input = gr.Number(label="Number of images", value=5, minimum=1, maximum=10, step=1)
    gallery = gr.Gallery(label="Generated images", elem_id="gallery", columns=5, object_fit="contain")
    
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

if __name__ == "__main__":
    demo.launch(share=True)
