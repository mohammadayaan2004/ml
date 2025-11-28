"""
Command-line interface for the Multimodal Vision-Language Pipeline
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
from MMV.multimodal_pipeline import MultimodalVisionPipeline

def main():
    parser = argparse.ArgumentParser(description="Multimodal Vision-Language Pipeline")
    parser.add_argument("--image", type=str, required=True, help="Path or URL to the image")
    parser.add_argument("--task", type=str, choices=["classify", "caption", "vqa", "similarity"], 
                        required=True, help="Task to perform")
    parser.add_argument("--texts", type=str, nargs="+", help="Texts for similarity comparison")
    parser.add_argument("--question", type=str, help="Question for VQA")
    parser.add_argument("--candidates", type=str, nargs="+", help="Candidate answers for VQA")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top classifications to return")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    print("Initializing Multimodal Vision-Language Pipeline...")
    pipeline = MultimodalVisionPipeline()
    
    # Load image
    print(f"Loading image from {args.image}...")
    try:
        image = pipeline.load_image(args.image)
        print("Image loaded successfully!")
    except Exception as e:
        print(f"Failed to load image: {e}")
        return
    
    # Perform requested task
    if args.task == "classify":
        print(f"\nClassifying image (top {args.top_k} predictions)...")
        try:
            results = pipeline.classify_image(image, top_k=args.top_k)
            for i, (label, confidence) in enumerate(results, 1):
                print(f"{i}. {label}: {confidence:.4f}")
        except Exception as e:
            print(f"Classification failed: {e}")
            
    elif args.task == "caption":
        print("\nGenerating image caption...")
        try:
            caption = pipeline.generate_caption(image)
            print(f"Caption: {caption}")
        except Exception as e:
            print(f"Captioning failed: {e}")
            
    elif args.task == "similarity":
        if not args.texts:
            print("Error: --texts argument required for similarity task")
            return
        print("\nComputing image-text similarities...")
        try:
            similarities = pipeline.clip_image_text_similarity(image, args.texts)
            for text, sim in zip(args.texts, similarities):
                print(f"'{text}': {sim:.4f}")
        except Exception as e:
            print(f"Similarity computation failed: {e}")
            
    elif args.task == "vqa":
        if not args.question:
            print("Error: --question argument required for VQA task")
            return
        print(f"\nAnswering question: {args.question}")
        try:
            if args.candidates:
                result = pipeline.vqa(image, args.question, args.candidates)
                print("Candidate answers with probabilities:")
                for answer, prob in result.items():
                    print(f"  {answer}: {prob:.4f}")
            else:
                result = pipeline.vqa(image, args.question)
                print(f"Answer: {result}")
        except Exception as e:
            print(f"VQA failed: {e}")

if __name__ == "__main__":
    main()