"""
Multimodal Vision-Language Pipeline
Integrates vision models (CLIP/ViT/SAM) with LLMs for captioning, VQA, and segmentation

ARCHITECTURE OVERVIEW:
=====================
1. Vision Encoder Layer:
   - CLIP: Extracts image and text embeddings in a shared latent space
   - ViT: Extracts image features for classification tasks
   - SAM: Produces segmentation masks for objects in images

2. Feature Projection Layer:
   - Linear projections to align dimensions between vision encoders and LLM input space
   - Normalization of embeddings for stable training/inference

3. LLM Integration:
   - Embedding injection: Directly inject visual embeddings into LLM's token embedding space
   - Cross-attention: Allow LLM to attend to visual features during generation
   - Prompt engineering: Format visual information as text prompts for LLMs

4. Output Generation:
   - Decoding strategies: Greedy, beam search, nucleus sampling
   - Post-processing: Filtering, formatting, and validation of outputs
"""

import torch
import numpy as np
from PIL import Image
import cv2
from transformers import CLIPProcessor, CLIPModel, ViTFeatureExtractor, ViTForImageClassification
from transformers import BlipProcessor, BlipForConditionalGeneration
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union
import requests
from io import BytesIO
import warnings
warnings.filterwarnings("ignore")

try:
    from segment_anything import SamPredictor, sam_model_registry
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("Segment Anything Model (SAM) not available. Install with: pip install git+https://github.com/facebookresearch/segment-anything.git")

class MultimodalVisionPipeline:
    """
    A multimodal pipeline combining vision models (CLIP/ViT/SAM) with LLMs
    for image captioning, visual question answering, and segmentation tasks.
    
    CONNECTION BETWEEN VISION ENCODERS AND LLMS:
    ==========================================
    1. CLIP Integration:
       - Extract image embeddings using CLIP vision encoder
       - Pass embeddings to LLM through cross-attention or prompt engineering
       - Example: "Image shows [DESCRIPTION] - Answer: [LLM_OUTPUT]"
       
    2. ViT Integration:
       - Extract patch-level features from ViT encoder
       - Pool features and project to LLM embedding dimension
       - Inject as prefix tokens in LLM input sequence
       
    3. SAM Integration:
       - Generate segmentation masks for regions of interest
       - Extract features from masked regions
       - Provide spatial information to LLM through structured prompts
    """
    
    def __init__(self, device: str = None):
        """
        Initialize the multimodal pipeline with vision models.
        
        Args:
            device: Device to run models on ('cpu' or 'cuda')
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Initialize CLIP model for image-text understanding
        print("Loading CLIP model...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Initialize ViT for image classification
        print("Loading ViT model...")
        self.vit_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224').to(self.device)
        self.vit_feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
        
        # Initialize BLIP for image captioning
        print("Loading BLIP model...")
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)
        
        # Initialize SAM if available
        self.sam_predictor = None
        if SAM_AVAILABLE:
            print("SAM model available for segmentation tasks")
        else:
            print("SAM model not available for segmentation tasks")
            
    def load_image(self, image_path_or_url: str) -> Image.Image:
        """
        Load an image from a file path or URL.
        
        Args:
            image_path_or_url: Path to image file or URL to image
            
        Returns:
            PIL Image object
        """
        if image_path_or_url.startswith("http"):
            response = requests.get(image_path_or_url)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(image_path_or_url).convert("RGB")
        return image
    
    def get_clip_image_embeddings(self, image: Image.Image) -> torch.Tensor:
        """
        Extract image embeddings using CLIP vision encoder.
        
        IMAGE FEATURE EXTRACTION WITH CLIP:
        ==================================
        1. Preprocessing: Resize and normalize image to 224x224
        2. Patch embedding: Convert image to patches and linearly embed
        3. Positional encoding: Add positional information to patches
        4. Transformer layers: Process patches through 12 transformer layers
        5. Global representation: Use [CLS] token as final image embedding
        6. Projection: Map to 512-dim shared latent space
        
        Args:
            image: PIL Image object
            
        Returns:
            Image embeddings tensor of shape (1, 512)
        """
        inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            # Extract image embeddings from CLIP vision encoder
            image_features = self.clip_model.get_image_features(**inputs)
        return image_features
    
    def get_vit_image_features(self, image: Image.Image) -> torch.Tensor:
        """
        Extract features from an image using ViT.
        
        IMAGE FEATURE EXTRACTION WITH ViT:
        =================================
        1. Preprocessing: Resize image to 224x224 and normalize
        2. Patch embedding: Split image into 16x16 patches, flatten each patch
        3. Linear projection: Project flattened patches to embedding dimension
        4. Positional embedding: Add learnable positional encodings
        5. Transformer encoder: Process through 12 transformer blocks
        6. Classification token: Use [CLS] token for image representation
        
        Args:
            image: PIL Image object
            
        Returns:
            Image features tensor of shape (1, 1000) - logits for 1000 ImageNet classes
        """
        inputs = self.vit_feature_extractor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.vit_model(**inputs)
            features = outputs.logits
        return features
    
    def classify_image(self, image: Image.Image, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Classify an image using ViT model.
        
        Args:
            image: PIL Image object
            top_k: Number of top predictions to return
            
        Returns:
            List of (label, confidence) tuples
        """
        inputs = self.vit_feature_extractor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.vit_model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probs, top_k)
        top_probs = top_probs.cpu().numpy()[0]
        top_indices = top_indices.cpu().numpy()[0]
        
        # Get ImageNet labels
        imagenet_labels = self._get_imagenet_labels()
        results = []
        for i in range(top_k):
            label = imagenet_labels[top_indices[i]]
            confidence = top_probs[i]
            results.append((label, confidence))
            
        return results
    
    def _get_imagenet_labels(self) -> List[str]:
        """
        Get ImageNet class labels.
        
        Returns:
            List of ImageNet class labels
        """
        # Full ImageNet labels (first 20 for demo purposes)
        return [
            "tench", "goldfish", "great white shark", "tiger shark", "hammerhead",
            "electric ray", "stingray", "cock", "hen", "ostrich", 
            "brambling", "goldfinch", "house finch", "junco", "indigo bunting",
            "robin", "bulbul", "jay", "magpie", "chickadee"
        ]
    
    def generate_caption(self, image: Image.Image, max_length: int = 50) -> str:
        """
        Generate a caption for an image using BLIP model.
        
        Args:
            image: PIL Image object
            max_length: Maximum length of caption
            
        Returns:
            Generated caption string
        """
        inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.blip_model.generate(**inputs, max_length=max_length)
        caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
        return caption
    
    def clip_image_text_similarity(self, image: Image.Image, texts: List[str]) -> List[float]:
        """
        Compute similarity between an image and a list of texts using CLIP.
        
        Args:
            image: PIL Image object
            texts: List of text strings to compare with image
            
        Returns:
            List of similarity scores
        """
        inputs = self.clip_processor(text=texts, images=image, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            similarities = logits_per_image.softmax(dim=1).cpu().numpy()[0]
        return similarities.tolist()
    
    def vqa(self, image: Image.Image, question: str, candidate_answers: List[str] = None) -> Union[str, Dict[str, float]]:
        """
        Perform Visual Question Answering using CLIP.
        
        Args:
            image: PIL Image object
            question: Question about the image
            candidate_answers: List of candidate answers (if None, generates open-ended answer)
            
        Returns:
            Either a generated answer string or dict of candidate answers with probabilities
        """
        if candidate_answers is None:
            # For open-ended VQA, we would need a specialized model
            # As a placeholder, we'll generate a descriptive caption
            caption = self.generate_caption(image)
            return f"Based on the image content: {caption}"
        else:
            # Zero-shot VQA with candidate answers
            # Formulate as "image of {question} {answer}"
            texts = [f"{question} {answer}" for answer in candidate_answers]
            similarities = self.clip_image_text_similarity(image, texts)
            
            # Return dictionary of answers with probabilities
            result = {}
            for answer, similarity in zip(candidate_answers, similarities):
                result[answer] = similarity
            return result
    
    def initialize_sam(self, sam_checkpoint_path: str, model_type: str = "vit_h") -> bool:
        """
        Initialize SAM model for segmentation tasks.
        
        Args:
            sam_checkpoint_path: Path to SAM checkpoint file
            model_type: Type of SAM model ("vit_h", "vit_l", or "vit_b")
            
        Returns:
            True if successful, False otherwise
        """
        if not SAM_AVAILABLE:
            print("SAM not available. Please install segment-anything package.")
            return False
            
        try:
            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint_path)
            sam.to(device=self.device)
            self.sam_predictor = SamPredictor(sam)
            return True
        except Exception as e:
            print(f"Failed to initialize SAM: {e}")
            return False
    
    def segment_image(self, image: np.ndarray, boxes: List[List[int]] = None, 
                     points: List[List[float]] = None) -> np.ndarray:
        """
        Segment objects in an image using SAM.
        
        HOW SAM PRODUCES SEGMENTATION MASKS:
        ===================================
        1. Image Encoding:
           - Uses a Vision Transformer (ViT) to encode the entire image
           - Produces image embeddings that capture both global and local features
           
        2. Prompt Encoding:
           - Encodes user prompts (points, boxes, or masks) into prompt embeddings
           - Points: Foreground (positive) or background (negative)
           - Boxes: Bounding boxes around objects of interest
           
        3. Mask Decoding:
           - Combines image and prompt embeddings through lightweight mask decoder
           - Outputs segmentation masks, quality predictions, and mask embeddings
           - Supports ambiguous prompts with multiple output masks
           
        4. Real-time Interaction:
           - Efficient architecture enables real-time mask prediction
           - Can refine masks iteratively with additional prompts
           
        Args:
            image: NumPy array of image (H, W, C)
            boxes: List of bounding boxes [x1, y1, x2, y2]
            points: List of points [[x, y], ...] with corresponding labels
            
        Returns:
            Segmentation mask
        """
        if self.sam_predictor is None:
            raise ValueError("SAM predictor not initialized. Call initialize_sam() first.")
            
        self.sam_predictor.set_image(image)
        
        input_boxes = torch.tensor(boxes, device=self.sam_predictor.device) if boxes else None
        input_points = torch.tensor(points, device=self.sam_predictor.device) if points else None
        
        masks, _, _ = self.sam_predictor.predict(
            point_coords=input_points,
            box=input_boxes,
            multimask_output=False,
        )
        
        return masks[0]