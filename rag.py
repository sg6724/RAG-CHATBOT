"""
rag_query.py - RAG Query Processing with FAISS
Handles: Information retrieval, prompt template, response generation
"""
import os
from dotenv import load_dotenv
import google.generativeai as genai
from embeddings import search_similar

# Load environment
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file!")

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)

print("🤖 Initializing query system...")

# Prompt template
PROMPT_TEMPLATE = """You are a helpful AI assistant that interacts with the user for general questions and answers context-based questions using the provided context.

Context:
{context}

Question: {question}

Instructions:
- Do general interactions conversationally
- Answer using the provided context above
- Cite sources using [Source: filename] format when referencing information
- If the context doesn't contain enough information to answer the question, clearly state: "I don't have enough information to answer this question."
- Be specific, detailed, and accurate in your answer
- Format your answer in a clear, readable way
- Give me structured answers visually appealling format

Answer:"""

print("✅ Query system ready!")


def query_documents(question: str, top_k: int = 5) -> dict:
    """
    Query the RAG system with a question
    
    Args:
        question: User's question
        top_k: Number of relevant chunks to retrieve
        
    Returns:
        dict with 'answer' and 'sources'
    """
    try:
        # Retrieve similar chunks
        results = search_similar(question, top_k)
        
        if not results:
            return {
                "answer": "No documents found. Please upload some documents first.",
                "sources": [],
                "question": question
            }
        
        # Build context
        context = "\n\n".join([
            f"[Source: {r['source']}]\n{r['text']}"
            for r in results
        ])
        
        # Create prompt
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        
        # Generate response with Gemini
        try:
            model = genai.GenerativeModel('gemini-2.5-flash')
        except:
            try:
                model = genai.GenerativeModel('gemini-1.5-flash')
            except:
                model = genai.GenerativeModel('gemini-1.5-pro')
        
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.2,
                top_p=0.8,
                top_k=40,
                max_output_tokens=2048
            )
        )
        
        # Format sources for response
        sources = []
        for r in results:
            sources.append({
                "text": r["text"][:200] + "..." if len(r["text"]) > 200 else r["text"],
                "source": r["source"],
                "type": r["type"],
                "page": r.get("page")
            })
        
        return {
            "answer": response.text,
            "sources": sources,
            "question": question
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "answer": f"Error processing query: {str(e)}",
            "sources": [],
            "question": question
        }


def query_with_filter(question: str, source_filter: str = None, top_k: int = 5) -> dict:
    """
    Query with optional source filtering
    
    Args:
        question: User's question
        source_filter: Filter by specific source (optional)
        top_k: Number of chunks to retrieve
        
    Returns:
        dict with 'answer' and 'sources'
    """
    # Retrieve with filter
    results = search_similar(question, top_k, source_filter)
    
    if not results:
        return {
            "answer": "No documents found matching your criteria.",
            "sources": [],
            "question": question
        }
    
    # Build context
    context = "\n\n".join([
        f"[Source: {r['source']}]\n{r['text']}"
        for r in results
    ])
    
    # Create prompt
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    
    # Generate response
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
    except:
        model = genai.GenerativeModel('gemini-1.5-flash')
    
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            temperature=0.2,
            top_p=0.8,
            top_k=40,
            max_output_tokens=2048
        )
    )
    
    # Format sources
    sources = []
    for r in results:
        sources.append({
            "text": r["text"][:200] + "..." if len(r["text"]) > 200 else r["text"],
            "source": r["source"],
            "type": r["type"],
            "page": r.get("page")
        })
    
    return {
        "answer": response.text,
        "sources": sources,
        "question": question
    }
