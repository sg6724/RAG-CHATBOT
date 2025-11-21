"""
embeddings.py - Document Processing and Embedding Creation with FAISS
Handles: Text chunking, file loading, URL scraping, FAISS storage
"""
import os
import pickle
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_core.documents import Document
import faiss
import requests
from bs4 import BeautifulSoup

# Load environment
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file!")

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)

print("🔧 Initializing FAISS embedding system...")

# Storage files
FAISS_INDEX_PATH = "./faiss_index.bin"
METADATA_PATH = "./metadata.pkl"
EMBEDDING_DIM = 768

# Initialize FAISS index
if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH):
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(METADATA_PATH, "rb") as f:
        metadata_store = pickle.load(f)
    print(f"✅ Loaded existing index with {index.ntotal} vectors")
else:
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    metadata_store = []
    print("✅ Created new FAISS index")

# Text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)

print("✅ Embedding system ready!")


def get_embedding(text: str):
    """Get embedding from Gemini"""
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=text,
        task_type="retrieval_document",
        output_dimensionality=EMBEDDING_DIM
    )
    return np.array(result['embedding'], dtype='float32')


def save_index():
    """Save FAISS index and metadata to disk"""
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata_store, f)


def add_text(text: str, source_name: str) -> int:
    """
    Add plain text to FAISS index
    
    Args:
        text: Text content to add
        source_name: Name/identifier for this text
        
    Returns:
        Number of chunks created
    """
    doc = Document(
        page_content=text,
        metadata={"source": source_name, "type": "text"}
    )
    
    chunks = text_splitter.split_documents([doc])
    
    # Generate embeddings
    embeddings = []
    for chunk in chunks:
        emb = get_embedding(chunk.page_content)
        embeddings.append(emb)
        metadata_store.append({
            "text": chunk.page_content,
            "source": source_name,
            "type": "text"
        })
    
    # Add to FAISS
    if embeddings:
        embeddings_array = np.array(embeddings)
        index.add(embeddings_array)
    
    # Save
    save_index()
    
    return len(chunks)


def add_file(file_path: str, filename: str) -> int:
    """
    Add file (PDF/DOCX/TXT) to FAISS index
    
    Args:
        file_path: Path to file
        filename: Original filename
        
    Returns:
        Number of chunks created
    """
    # Determine file type
    file_ext = filename.split(".")[-1].lower()
    
    # Select appropriate loader
    loaders = {
        "pdf": PyPDFLoader,
        "docx": Docx2txtLoader,
        "txt": TextLoader
    }
    
    loader_class = loaders.get(file_ext)
    if not loader_class:
        raise ValueError(f"Unsupported file type: {file_ext}")
    
    # Load and process
    loader = loader_class(file_path)
    documents = loader.load()
    
    # Add metadata
    for doc in documents:
        doc.metadata["source"] = filename
        doc.metadata["type"] = file_ext
    
    chunks = text_splitter.split_documents(documents)
    
    # Generate embeddings
    embeddings = []
    for chunk in chunks:
        emb = get_embedding(chunk.page_content)
        embeddings.append(emb)
        metadata_store.append({
            "text": chunk.page_content,
            "source": filename,
            "type": file_ext,
            "page": chunk.metadata.get("page")
        })
    
    # Add to FAISS
    if embeddings:
        embeddings_array = np.array(embeddings)
        index.add(embeddings_array)
    
    # Save
    save_index()
    
    return len(chunks)


def add_url(url: str) -> int:
    """
    Scrape and add content from URL to FAISS index
    
    Args:
        url: URL to scrape
        
    Returns:
        Number of chunks created
    """
    # Fetch content
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    
    # Parse HTML
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Remove unwanted elements
    for element in soup(["script", "style", "nav", "footer"]):
        element.decompose()
    
    # Extract text
    text = soup.get_text(separator="\n", strip=True)
    
    # Create document
    doc = Document(
        page_content=text,
        metadata={"source": url, "type": "url"}
    )
    
    chunks = text_splitter.split_documents([doc])
    
    # Generate embeddings
    embeddings = []
    for chunk in chunks:
        emb = get_embedding(chunk.page_content)
        embeddings.append(emb)
        metadata_store.append({
            "text": chunk.page_content,
            "source": url,
            "type": "url"
        })
    
    # Add to FAISS
    if embeddings:
        embeddings_array = np.array(embeddings)
        index.add(embeddings_array)
    
    # Save
    save_index()
    
    return len(chunks)


def search_similar(query_text: str, top_k: int = 5, source_filter: str = None):
    """
    Search for similar chunks in FAISS
    
    Args:
        query_text: Query text
        top_k: Number of results to return
        source_filter: Optional source filter
        
    Returns:
        List of dicts with text, source, type, page, score
    """
    if index.ntotal == 0:
        return []
    
    # Get query embedding
    query_embedding = get_embedding(query_text)
    query_embedding = query_embedding.reshape(1, -1)
    
    # Search FAISS (get more if filtering)
    search_k = top_k * 5 if source_filter else top_k
    search_k = min(search_k, index.ntotal)
    
    distances, indices = index.search(query_embedding, search_k)
    
    # Get results
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        
        meta = metadata_store[idx]
        
        # Apply source filter if specified
        if source_filter and meta.get("source") != source_filter:
            continue
        
        results.append({
            "text": meta["text"],
            "source": meta["source"],
            "type": meta["type"],
            "page": meta.get("page"),
            "score": float(1 / (1 + dist))  # Convert distance to similarity
        })
        
        # Stop if we have enough results
        if len(results) >= top_k:
            break
    
    return results


def get_document_count() -> int:
    """Get total number of document chunks in FAISS index"""
    return index.ntotal


def clear_database():
    """Clear all documents from FAISS index"""
    global index, metadata_store
    
    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    metadata_store = []
    
    # Remove files
    if os.path.exists(FAISS_INDEX_PATH):
        os.remove(FAISS_INDEX_PATH)
    if os.path.exists(METADATA_PATH):
        os.remove(METADATA_PATH)
    
    print("🗑️ Database cleared")
