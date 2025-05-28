#!/usr/bin/env python3
"""
Document processor for RAG system.
Handles loading, splitting, and storing documents in the vector database.
"""

import os
import glob
import shutil
from typing import List, Dict, Any, Optional
from pathlib import Path

# Import text splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import from langchain-community
from langchain_community.document_loaders import (
    PyPDFLoader,
    DirectoryLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain_chroma import Chroma

# Import from langchain-core
from langchain_core.documents import Document

from get_embedding_function import get_embedding_function

# Constants
CHROMA_PATH = "chroma"
DATA_PATH = "data"


def load_documents(data_dir: str = DATA_PATH) -> List[Document]:
    """
    Load documents from the data directory.
    
    Args:
        data_dir: Directory containing documents to process
        
    Returns:
        List of loaded documents
    """
    # Dictionary to track loaded document counts by type
    loaded_docs_count = {
        "pdf": 0,
        "docx": 0,
        "txt": 0
    }
    
    all_documents = []
    
    try:
        # Check if data directory exists
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            print(f"Created data directory at {data_dir}")
            print("Please add documents to this directory and run again.")
            return []
        
        # Process PDF files
        for pdf_path in glob.glob(f"{data_dir}/**/*.pdf", recursive=True):
            try:
                loader = PyPDFLoader(pdf_path)
                documents = loader.load()
                
                # Add source information to metadata
                for doc in documents:
                    doc.metadata["source"] = pdf_path
                    doc.metadata["id"] = os.path.basename(pdf_path)
                    doc.metadata["title"] = os.path.basename(pdf_path)
                
                all_documents.extend(documents)
                loaded_docs_count["pdf"] += 1
            except Exception as e:
                print(f"Error loading PDF {pdf_path}: {str(e)}")
        
        # Process DOCX files
        for docx_path in glob.glob(f"{data_dir}/**/*.docx", recursive=True):
            try:
                loader = Docx2txtLoader(docx_path)
                documents = loader.load()
                
                # Add source information to metadata
                for doc in documents:
                    doc.metadata["source"] = docx_path
                    doc.metadata["id"] = os.path.basename(docx_path)
                    doc.metadata["title"] = os.path.basename(docx_path)
                
                all_documents.extend(documents)
                loaded_docs_count["docx"] += 1
            except Exception as e:
                print(f"Error loading DOCX {docx_path}: {str(e)}")
        
        # Process TXT files
        for txt_path in glob.glob(f"{data_dir}/**/*.txt", recursive=True):
            try:
                loader = TextLoader(txt_path)
                documents = loader.load()
                
                # Add source information to metadata
                for doc in documents:
                    doc.metadata["source"] = txt_path
                    doc.metadata["id"] = os.path.basename(txt_path)
                    doc.metadata["title"] = os.path.basename(txt_path)
                
                all_documents.extend(documents)
                loaded_docs_count["txt"] += 1
            except Exception as e:
                print(f"Error loading TXT {txt_path}: {str(e)}")
        
        # Print summary
        print(f"Loaded {loaded_docs_count['pdf']} PDF documents")
        print(f"Loaded {loaded_docs_count['docx']} Word documents")
        print(f"Loaded {loaded_docs_count['txt']} text files")
        print(f"Loaded {len(all_documents)} total documents")
        
        return all_documents
    
    except Exception as e:
        print(f"Error loading documents: {str(e)}")
        return []


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into chunks.
    
    Args:
        documents: List of documents to split
        
    Returns:
        List of document chunks
    """
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
        )
        
        chunks = text_splitter.split_documents(documents)
        
        # Add line reference metadata to chunks
        for i, chunk in enumerate(chunks):
            # Calculate approximate line number ranges
            # Since we don't have actual line numbers from the text_splitter,
            # we'll estimate based on character position and assume average chars per line
            content = chunk.page_content
            avg_chars_per_line = 80  # Assumption of average characters per line
            
            # Assign an estimated line start and end
            # This is a rough approximation
            line_start = i * (1000 - 100) // avg_chars_per_line + 1  # Adjusting for overlap
            line_end = line_start + (len(content) // avg_chars_per_line)
            
            # Add to metadata
            chunk.metadata["line_start"] = line_start
            chunk.metadata["line_end"] = line_end
            
            # Add chunk index for reference
            chunk.metadata["chunk_index"] = i
        
        print(f"Split {len(documents)} documents into {len(chunks)} chunks")
        return chunks
    except Exception as e:
        print(f"Error splitting documents: {str(e)}")
        return documents


def store_documents(chunks: List[Document]) -> None:
    """
    Store document chunks in the vector database.
    
    Args:
        chunks: List of document chunks to store
    """
    try:
        # Remove existing database if it exists
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)
            print(f"Removed existing database at {CHROMA_PATH}")
        
        # Get embedding function
        embedding_function = get_embedding_function()
        
        # Create and persist database
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_function,
            persist_directory=CHROMA_PATH,
        )
        
        # Make sure the database is persisted
        db.persist()
        print(f"Stored {len(chunks)} document chunks in database")
        
    except Exception as e:
        print(f"Error storing documents: {str(e)}")
        raise


def process_documents():
    """
    Main function to process documents and build the vector database.
    """
    try:
        print("Starting document processing...")
        documents = load_documents()
        
        if not documents:
            print("No documents found to process.")
            return
        
        chunks = split_documents(documents)
        store_documents(chunks)
        print("Document processing completed successfully!")
        
    except Exception as e:
        print(f"Error processing documents: {str(e)}")


if __name__ == "__main__":
    process_documents()
