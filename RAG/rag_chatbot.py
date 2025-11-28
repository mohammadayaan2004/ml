import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import List, Tuple, Dict
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGChatbot:
    """
    Retrieval-Augmented Generation (RAG) Chatbot implementation
    
    Combines a vector database with a language model to provide document-grounded responses.
    """
    
    def __init__(self, model_name: str = "gpt2", embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the RAG Chatbot
        
        Args:
            model_name: Name of the generative model to use
            embedding_model: Name of the sentence transformer model for embeddings
        """
        # Initialize the embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize the generative model
        logger.info(f"Loading generative model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Move model to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.to('cuda')
            logger.info("Model moved to GPU")
        else:
            logger.info("Using CPU for inference")
            
        # Initialize FAISS index
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)
        
        # Store documents
        self.documents = []
        
    def add_documents(self, documents: List[str]):
        """
        Add documents to the vector database
        
        Args:
            documents: List of document strings to add
        """
        self.documents.extend(documents)
        
        # Create embeddings for documents
        logger.info(f"Encoding {len(documents)} documents...")
        embeddings = self.embedding_model.encode(documents)
        
        # Add embeddings to FAISS index
        self.index.add(embeddings.astype(np.float32))
        logger.info(f"Added {len(documents)} documents to index. Total documents: {len(self.documents)}")
        
    def retrieve_documents(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: Query string
            k: Number of documents to retrieve
            
        Returns:
            List of tuples (document, similarity_score)
        """
        # Create embedding for query
        logger.info(f"Retrieving top {k} documents for query: {query}")
        query_embedding = self.embedding_model.encode([query])
        
        # Search in FAISS index
        distances, indices = self.index.search(query_embedding.astype(np.float32), k)
        
        logger.info(f"Retrieved {len(indices[0])} documents")
        
        # Return documents with scores
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents):
                results.append((self.documents[idx], float(distances[0][i])))
                
        return results
    
    def generate_response(self, query: str, max_length: int = 200) -> str:
        """
        Generate a response to a query using RAG
        
        Args:
            query: Query string
            max_length: Maximum length of generated response
            
        Returns:
            Generated response string
        """
        # Retrieve relevant documents
        retrieved_docs = self.retrieve_documents(query, k=3)
        
        # Create context from retrieved documents
        context = "\n".join([doc for doc, _ in retrieved_docs])
        
        # Create prompt with context
        prompt = f"Answer the question based on the context below. Be concise and only use information from the context.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
        
        # Tokenize input
        logger.info("Generating response...")
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        # Move inputs to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=len(inputs["input_ids"][0]) + 50,  # Limit generation length
                num_return_sequences=1,
                temperature=0.5,  # Lower temperature for more focused responses
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                eos_token_id=self.tokenizer.eos_token_id,
                early_stopping=True
            )
            
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the answer part (remove the prompt)
        if "Answer:" in response:
            # Find the last occurrence of "Answer:" and extract everything after it
            answer_start = response.rfind("Answer:") + len("Answer:")
            response = response[answer_start:].strip()
            
        # Clean up the response - remove any trailing context or extra text
        response_lines = response.split('\n')
        if len(response_lines) > 3:  # If response is too long, take only first few lines
            response = '\n'.join(response_lines[:3]).strip()
            
        return response
    
    def chat(self, query: str) -> str:
        """
        Main chat interface
        
        Args:
            query: User query
            
        Returns:
            Chatbot response
        """
        try:
            response = self.generate_response(query)
            return response
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"Sorry, I encountered an error: {str(e)}"

# Example usage
if __name__ == "__main__":
    # Initialize chatbot
    chatbot = RAGChatbot()
    
    # Add sample documents
    sample_docs = [
        "The Earth revolves around the Sun in approximately 365.25 days.",
        "Water boils at 100 degrees Celsius at sea level.",
        "The human body has 206 bones.",
        "Photosynthesis is the process by which plants convert sunlight into energy.",
        "The speed of light in a vacuum is approximately 299,792,458 meters per second."
    ]
    
    chatbot.add_documents(sample_docs)
    
    # Test query
    query = "How long does it take Earth to revolve around the Sun?"
    response = chatbot.chat(query)
    print(f"Query: {query}")
    print(f"Response: {response}")