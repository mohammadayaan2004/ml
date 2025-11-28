"""
Test script for the Multimodal Vision-Language Pipeline
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from MMV.multimodal_pipeline import MultimodalVisionPipeline
from PIL import Image
import numpy as np
import argparse

def test_pipeline():
    """Test the multimodal vision pipeline with sample tasks."""
    print("Initializing Multimodal Vision-Language Pipeline...")
    pipeline = MultimodalVisionPipeline()
    
    # For demonstration, we'll create a simple test image
    # In practice, you would load a real image
    print("\n1. Creating a test image...")
    # Create a simple red square image for testing
    test_image = Image.new('RGB', (224, 224), color='red')
    
    print("\n2. Testing image classification...")
    try:
        classifications = pipeline.classify_image(test_image, top_k=3)
        print("Top classifications:")
        for label, confidence in classifications:
            print(f"  {label}: {confidence:.4f}")
    except Exception as e:
        print(f"Classification failed: {e}")
    
    print("\n3. Testing image captioning...")
    try:
        caption = pipeline.generate_caption(test_image)
        print(f"Generated caption: {caption}")
    except Exception as e:
        print(f"Captioning failed: {e}")
    
    print("\n4. Testing CLIP similarity...")
    try:
        texts = ["a red square", "a blue circle", "a green triangle"]
        similarities = pipeline.clip_image_text_similarity(test_image, texts)
        print("Text similarities:")
        for text, sim in zip(texts, similarities):
            print(f"  '{text}': {sim:.4f}")
    except Exception as e:
        print(f"CLIP similarity failed: {e}")
    
    print("\n5. Testing Visual Question Answering...")
    try:
        question = "What color is the shape?"
        candidates = ["red", "blue", "green", "yellow"]
        vqa_result = pipeline.vqa(test_image, question, candidates)
        print(f"VQA results for '{question}':")
        for answer, prob in vqa_result.items():
            print(f"  {answer}: {prob:.4f}")
    except Exception as e:
        print(f"VQA failed: {e}")
    
    print("\nPipeline test completed!")

if __name__ == "__main__":
    test_pipeline()