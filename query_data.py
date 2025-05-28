"""
RAG query module.
Handles querying the vector database and generating responses with the LLM.
"""

import argparse
import os
import json
from typing import Dict, List, Any, Optional

from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama

from get_embedding_function import get_embedding_function

# Constants
CHROMA_PATH = "chroma"

# RAG prompt template
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Question: {question}

Answer the question based only on the information provided in the context.
If the context doesn't contain the information needed to answer the question,
respond with "I don't have enough information to answer this question."

Where appropriate, cite your sources using [SOURCE_ID].
"""


def query_rag(
    query_text: str, 
    k: int = 5, 
    output_format: str = "text", 
    model: str = "llama3.2:3b"
) -> Dict[str, Any]:
    """
    Query the RAG system and return the answer with sources.
    
    Args:
        query_text: The question to ask
        k: Number of chunks to retrieve
        output_format: "text" or "json"
        model: Ollama model to use
        
    Returns:
        Dict containing the question, answer, and sources
    """
    response_data = {
        "question": query_text,
        "answer": "",
        "sources": [],
        "error": None
    }
    
    try:
        # Check if the database exists
        if not os.path.exists(CHROMA_PATH):
            error_msg = f"Database not found at {CHROMA_PATH}. Please run process_documents.py first."
            response_data["error"] = error_msg
            print(error_msg)
            return response_data
        
        # Get embedding function
        embedding_function = get_embedding_function()
        
        # Load the database
        try:
            db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        except Exception as e:
            error_msg = f"Error loading database: {str(e)}"
            response_data["error"] = error_msg
            print(error_msg)
            return response_data
        
        # Search the database
        try:
            results = db.similarity_search_with_score(query_text, k=k)
        except Exception as e:
            error_msg = f"Error searching database: {str(e)}"
            response_data["error"] = error_msg
            print(error_msg)
            return response_data
        
        # Format context and collect sources
        context_parts = []
        sources = []
        
        # Sort results by score (higher is better)
        results.sort(key=lambda x: x[1], reverse=True)
        
        for i, (doc, score) in enumerate(results):
            # Create source ID from filename or index
            source_id = doc.metadata.get("id", f"source-{i}")
            title = doc.metadata.get("title", "Unknown")
            line_start = doc.metadata.get("line_start", "Unknown")
            line_end = doc.metadata.get("line_end", "Unknown")
            
            # Add formatted content to context
            context_parts.append(f"{doc.page_content}\n\nSource: [{source_id}]")
            
            # Extract line reference information
            line_ref = ""
            if line_start != "Unknown" and line_end != "Unknown":
                line_ref = f"Lines {line_start}-{line_end}"
            
            # Add source information
            sources.append({
                "id": source_id,
                "title": title,
                "relevance_score": float(score),
                "line_reference": line_ref,
                "text_sample": doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
            })
        
        # Join context parts
        context_text = "\n\n---\n\n".join(context_parts)
        
        # Check if we found any relevant documents
        if not context_parts:
            response_data["answer"] = "I couldn't find any relevant information to answer your question."
            return response_data
        
        # Create prompt
        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        formatted_prompt = prompt.format(context=context_text, question=query_text)
        
        # Query the LLM
        try:
            llm = Ollama(model=model)
            response_text = llm.invoke(formatted_prompt)
            response_data["answer"] = response_text
            response_data["sources"] = sources
        except Exception as e:
            error_msg = f"Error querying LLM: {str(e)}"
            response_data["error"] = error_msg
            print(error_msg)
            # Return partial results with error
            return response_data
        
        # Print formatted output for text mode
        if output_format == "text":
            print("\n" + "="*50)
            print(f"Question: {query_text}")
            print("="*50)
            print(f"Answer: {response_text}")
            print("-"*50)
            print("Sources:")
            for i, source in enumerate(sources):
                print(f"[{i+1}] {source['id']}")
                print(f"    Title: {source['title']}")
                line_info = f"    {source['line_reference']}" if source['line_reference'] else ""
                if line_info:
                    print(line_info)
                print(f"    Relevance: {source['relevance_score']:.4f}")
                print(f"    Content excerpt: {source['text_sample']}")
                print("")
    
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        response_data["error"] = error_msg
        print(error_msg)
    
    return response_data


def main():
    """Command-line interface for the RAG system."""
    parser = argparse.ArgumentParser(description="Query the RAG system")
    parser.add_argument("query_text", type=str, help="The question to ask")
    parser.add_argument(
        "--format", 
        choices=["text", "json"], 
        default="text", 
        help="Output format (text or json)"
    )
    parser.add_argument(
        "--k", 
        type=int, 
        default=5, 
        help="Number of chunks to retrieve"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="llama3.2:3b", 
        help="Ollama model to use"
    )
    
    args = parser.parse_args()
    
    response = query_rag(
        args.query_text, 
        k=args.k, 
        output_format=args.format, 
        model=args.model
    )
    
    if args.format == "json":
        print(json.dumps(response, indent=2))


if __name__ == "__main__":
    main()
