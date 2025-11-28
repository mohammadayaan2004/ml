# Machine Learning Projects

This repository contains various machine learning projects and implementations.

## Projects

### 1. Retrieval-Augmented Generation (RAG) Chatbot
Location: [RAG/](RAG/)

A document-aware LLM assistant that combines a vector database with a language model to provide grounded responses with fewer hallucinations and citeable outputs.

### 2. Multimodal Vision-Language Pipeline
Location: [MMV/](MMV/)

A fast and effective implementation of a multimodal pipeline combining vision models (CLIP/ViT/SAM) with LLMs for:
- Image captioning
- Visual Question Answering (VQA)
- Image segmentation with SAM
- Image classification with ViT
- Image-text similarity with CLIP

Features:
- Image Classification: Uses Vision Transformer (ViT) to classify images
- Image Captioning: Generates descriptive captions using BLIP model
- Visual Question Answering: Answers questions about images using CLIP
- Image-Text Similarity: Computes similarity between images and text using CLIP
- Image Segmentation: Segments objects in images using Segment Anything Model (SAM)
- Enhanced Documentation: Detailed explanations of how vision encoders connect with LLMs
- Code Examples: Comprehensive examples for loading models, extracting embeddings, and LLM integration

## Installation

To set up any of the projects, navigate to the respective directory and install the required packages:

```bash
# For RAG Chatbot
cd RAG
pip install -r requirements.txt

# For Multimodal Vision-Language Pipeline
cd MMV
pip install -r requirements.txt
```

For the SAM segmentation feature in the MMV pipeline, you'll also need to download a model checkpoint:
```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

## Usage

### Enhanced Demo
Run the enhanced demo to see detailed explanations of the pipeline architecture:
```bash
python enhanced_demo.py
```

### Detailed Architecture Documentation
See [pipeline_architecture.md](MMV/pipeline_architecture.md) for a comprehensive overview of:
- Vision encoder layer (CLIP/ViT/SAM)
- Feature projection layer
- LLM input formatting
- Output generation

### Code Examples
See [llm_integration_examples.py](MMV/llm_integration_examples.py) for examples of:
- Loading pretrained CLIP/ViT models
- Extracting image embeddings
- Passing embeddings to LLMs
- Running SAM for segmentation

### Command-Line Interface
```bash
# Image classification
python cli_multimodal.py --image path/to/image.jpg --task classify --top-k 5

# Image captioning
python cli_multimodal.py --image path/to/image.jpg --task caption

# Visual Question Answering
python cli_multimodal.py --image path/to/image.jpg --task vqa --question "What is in the image?"

# Image-text similarity
python cli_multimodal.py --image path/to/image.jpg --task similarity --texts "a red car" "a blue sky"
```

Refer to the README files in each project directory for detailed usage instructions:
- [RAG/README.md](RAG/README.md)
- [MMV/README.md](MMV/README.md)