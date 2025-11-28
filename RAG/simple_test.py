#!/usr/bin/env python3
"""
Simple test script for the RAG Chatbot fix
"""

from rag_chatbot import RAGChatbot

def test_fix():
    print("Testing RAG Chatbot fix...")
    
    # Initialize chatbot
    chatbot = RAGChatbot()
    
    # Add sample documents
    sample_docs = [
        "The Earth revolves around the Sun in approximately 365.25 days.",
        "Water boils at 100 degrees Celsius at sea level.",
    ]
    
    print(f"Adding {len(sample_docs)} sample documents...")
    chatbot.add_documents(sample_docs)
    print("Documents added successfully!")
    
    # Test the specific query that was failing
    query = "how many day earth take for 1 revolution"
    print(f"\nQuery: {query}")
    response = chatbot.chat(query)
    print(f"Response: {response}")
    
    # Test another query
    query2 = "How long does it take Earth to revolve around the Sun?"
    print(f"\nQuery: {query2}")
    response2 = chatbot.chat(query2)
    print(f"Response: {response2}")

if __name__ == "__main__":
    test_fix()