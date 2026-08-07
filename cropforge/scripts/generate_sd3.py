import os
import torch
from diffusers import StableDiffusionPipeline
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

def main():
    # Use standard pre-trained Stable Diffusion v1.5 from Hugging Face
    model_id = "runwayml/stable-diffusion-v1-5"
    
    print(f"Loading {model_id} from Hugging Face...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        if device == "cuda":
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(device)
            
        print("Pipeline loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Prompt for tomato leaf disease generation
    prompt = "A high-resolution, realistic photo of a tomato leaf with early blight disease, captured in a greenhouse, natural lighting"
    print(f"Generating image for prompt: '{prompt}'")
    
    image = pipe(
        prompt,
        num_inference_steps=30,
        guidance_scale=7.5,
    ).images[0]
    
    # Save image
    output_dir = r"D:\Crop-Forge\outputs\sd"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "generated_tomato_leaf.png")
    
    image.save(output_path)
    print(f"Successfully generated and saved image to: {output_path}")

if __name__ == "__main__":
    main()
