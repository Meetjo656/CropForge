import os
from dotenv import load_dotenv, find_dotenv
from huggingface_hub import InferenceClient

load_dotenv(find_dotenv(), override=True)

# Get HF Token
token = os.getenv("HF_TOKEN", "").strip().strip('"').strip("'")
if not token:
    print("Warning: HF_TOKEN environment variable not set. Please set HF_TOKEN in your .env file or environment.")

def main():
    print("Connecting to Hugging Face Cloud Inference API (Zero Local Download)...")
    
    # Initialize Serverless Cloud Client
    client = InferenceClient(api_key=token)
    
    prompt = "A high-resolution, realistic photo of a tomato leaf with early blight disease, close-up, sharp focus, natural lighting"
    print(f"Generating image on Hugging Face Cloud for prompt: '{prompt}'...")
    
    try:
        # Use FLUX.1-schnell or Stable Diffusion v1.5 on Hugging Face Cloud
        image = client.text_to_image(
            prompt=prompt,
            model="black-forest-labs/FLUX.1-schnell"
        )
        
        output_dir = r"D:\Crop-Forge\outputs\cloud"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "generated_tomato_leaf_cloud.png")
        
        image.save(output_path)
        print(f"SUCCESS! Image generated via Cloud API and saved to: {output_path}")
        
    except Exception as e:
        print(f"Cloud Inference Error: {e}")

if __name__ == "__main__":
    main()
