#!/usr/bin/env python3
"""
Module for providing embedding functionality for the RAG system.
"""
from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embedding_function():
    """
    Returns a function that generates embeddings using HuggingFaceEmbeddings.
    
    Uses an efficient, all-purpose embedding model from Sentence Transformers.
    """
    try:
        # Using a popular and versatile embedding model
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return embeddings
    except Exception as e:
        print(f"Error initializing embedding model: {str(e)}")
        print("Falling back to a smaller model...")
        try:
            # Fallback to a smaller model if the first one fails
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2")
            return embeddings
        except Exception as e:
            print(f"Error with fallback embedding model: {str(e)}")
            raise Exception("Failed to initialize embedding models. Please check your installation.")

if __name__ == "__main__":
    # Test the function
    embedding_function = get_embedding_function()
    print(f"Successfully initialized embedding function: {embedding_function.__class__.__name__}")
