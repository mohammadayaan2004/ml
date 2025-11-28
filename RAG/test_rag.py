#!/usr/bin/env python3
"""
Test script for the RAG Chatbot
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_chatbot import RAGChatbot


def test_rag_chatbot():
    print("Testing RAG Chatbot...")
    
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
    
    print(f"Adding {len(sample_docs)} sample documents...")
    chatbot.add_documents(sample_docs)
    print("Documents added successfully!")
    
    # Test queries
    test_queries = [
        "How long does it take Earth to revolve around the Sun?",
        "What is the boiling point of water?",
        "How many bones are in the human body?",
        "What is photosynthesis?",
        "What is the speed of light?"
    ]
    
    print("\nTesting queries:")
    for query in test_queries:
        print(f"\nQuery: {query}")
        response = chatbot.chat(query)
        print(f"Response: {response}")
    
    print("\nTest completed!")


if __name__ == "__main__":
    test_rag_chatbot()