"""
Example script demonstrating all features of the Multimodal Vision-Language Pipeline
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from MMV.multimodal_pipeline import MultimodalVisionPipeline
from PIL import Image
import requests
from io import BytesIO

def download_sample_image():
    """Download a sample image for demonstration."""
    url = "https://images.unsplash.com/photo-1502810365585-56ffa361fdde?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=500&q=80"
    response = requests.get(url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    return image

def main():
    print("=== Multimodal Vision-Language Pipeline Demo ===\n")
    
    # Initialize pipeline
    print("1. Initializing pipeline...")
    pipeline = MultimodalVisionPipeline()
    
    # Load sample image
    print("2. Loading sample image...")
    try:
        image = download_sample_image()
        print("   Sample image loaded successfully!\n")
    except Exception as e:
        print(f"   Failed to load sample image: {e}")
        print("   Creating a simple test image instead...")
        image = Image.new('RGB', (224, 224), color='red')
    
    # Demonstrate image classification
    print("3. Demonstrating image classification...")
    try:
        classifications = pipeline.classify_image(image, top_k=3)
        print("   Top classifications:")
        for label, confidence in classifications:
            print(f"     {label}: {confidence:.4f}")
    except Exception as e:
        print(f"   Classification failed: {e}")
    
    print("\n4. Demonstrating image captioning...")
    try:
        caption = pipeline.generate_caption(image, max_length=30)
        print(f"   Generated caption: {caption}")
    except Exception as e:
        print(f"   Captioning failed: {e}")
    
    print("\n5. Demonstrating CLIP image-text similarity...")
    try:
        texts = [
            "a beautiful landscape",
            "a city skyline",
            "a colorful abstract art",
            "a close-up of flowers"
        ]
        similarities = pipeline.clip_image_text_similarity(image, texts)
        print("   Image-text similarities:")
        for text, sim in zip(texts, similarities):
            print(f"     '{text}': {sim:.4f}")
    except Exception as e:
        print(f"   CLIP similarity failed: {e}")
    
    print("\n6. Demonstrating Visual Question Answering...")
    try:
        question = "What is the main subject of this image?"
        # Open-ended VQA
        open_ended_answer = pipeline.vqa(image, question)
        print(f"   Open-ended VQA: {open_ended_answer}")
        
        # Multiple-choice VQA
        question = "What type of scene is this?"
        candidates = ["landscape", "portrait", "abstract", "urban"]
        mc_result = pipeline.vqa(image, question, candidates)
        print(f"   Multiple-choice VQA for '{question}':")
        for answer, prob in mc_result.items():
            print(f"     {answer}: {prob:.4f}")
    except Exception as e:
        print(f"   VQA failed: {e}")
    
    print("\n=== Demo completed successfully! ===")

if __name__ == "__main__":
    main()