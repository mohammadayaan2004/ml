#!/usr/bin/env python3
"""
Command-line interface for the RAG Chatbot
"""

import sys
import argparse
from rag_chatbot import RAGChatbot

def main():
    parser = argparse.ArgumentParser(description="RAG Chatbot CLI")
    parser.add_argument("--docs", nargs="+", help="Documents to add to the chatbot")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    
    args = parser.parse_args()
    
    # Initialize chatbot
    chatbot = RAGChatbot()
    
    # Add documents if provided
    if args.docs:
        print(f"Adding {len(args.docs)} documents...")
        chatbot.add_documents(args.docs)
        print("Documents added successfully!")
    
    if args.interactive:
        print("\n🤖 RAG Chatbot Interactive Mode")
        print("Type 'quit' to exit, 'help' for commands\n")
        
        while True:
            try:
                query = input("You: ").strip()
                
                if query.lower() in ['quit', 'exit']:
                    print("Goodbye! 👋")
                    break
                elif query.lower() == 'help':
                    print("\nCommands:")
                    print("  quit/exit - Exit the chatbot")
                    print("  help - Show this help message")
                    print("  Or just ask any question!\n")
                    continue
                elif not query:
                    continue
                
                # Get response
                response = chatbot.chat(query)
                print(f"Bot: {response}\n")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except EOFError:
                print("\n\nGoodbye! 👋")
                break
    else:
        # Single query mode
        if len(sys.argv) > 1:
            query = " ".join(sys.argv[1:])
            response = chatbot.chat(query)
            print(f"Query: {query}")
            print(f"Response: {response}")
        else:
            print("Please provide a query or use --interactive mode")
            parser.print_help()

if __name__ == "__main__":
    main()