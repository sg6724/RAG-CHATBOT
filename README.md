# 🧠RAG Chatbot (FAISS Version)

A simple, fast, and Windows-friendly retrieval-augmented generation chatbot that answers questions using your own files, web pages, or pasted text — powered by Google Gemini embeddings, Gemini 2.0 Flash, FastAPI, Streamlit, and FAISS for vector storage.

***

## 🚀 Features

- **Upload**: Add text, PDFs, DOCX, or TXT files, or scrape a URL.
- **Query**: Ask questions about your uploaded documents.
- **Citations**: Get clear source references for every answer.
- **Simple UI**: Use a modern Streamlit interface.
- **Blazing fast setup**: No ChromaDB, no LangChain, no compilation headaches. Installs and runs quickly on Windows, Linux, or Mac.

***

## 🛠️ Setup

### 1. Clone and enter the project directory

```bash
git clone <your-repo-url>
cd rag-chatbot
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Add your Gemini API key

- Create a file named `.env`:

```
GOOGLE_API_KEY=your_gemini_api_key_here
```



### 6. Open the app in your browser

Deployed Link : https://frontend-pvf5.vercel.app/

***

## ✍️ Usage

1. **Add Data**
   - In the sidebar, upload text, choose a file, or provide a URL.
   - Supported file types: PDF, DOCX, TXT.
2. **Ask Questions**
   - Use the chat box to ask about your uploaded documents.
   - RAG will fetch relevant context and answer based only on your data.
3. **View Sources**
   - Expand “View Sources” under answers for chunk citations and context.
4. **Database Management**
   - See document count, and clear the database with one click if needed.

***

## 🖇 Technology Stack

- **Embeddings**: Google Gemini text-embedding-004 (768-dim)
- **Language Model**: Gemini 2.0 Flash (`gemini-2.0-flash-exp`)
- **Vector Database**: FAISS (CPU, persistent, lightweight)
- **Backend**: FastAPI
- **Frontend**: HTML , CSS , JS
- **Document Parsing**: PyPDF, python-docx, BeautifulSoup4
- **Utilities**: dotenv, requests

***

## ⚡ Installation/Runtime Tips

- **Super fast install**: No ChromaDB or Onnxruntime bloat. Works perfectly on Windows！
- **GPU/CPU**: FAISS runs efficiently on CPU; no special hardware required.
- **Persistence**: Index and metadata are saved (`faiss_index.bin`, `metadata.pkl`) and reused on restart.
- **For problems**: If you hit any “missing compiler” or “build error,” check you are running the FAISS version and using the right `requirements.txt`.

***

## 📦 requirements.txt

```
fastapi
uvicorn[standard]
python-multipart
google-generativeai
faiss-cpu
langchain
langchain-community
pypdf
python-docx
beautifulsoup4
requests
python-dotenv
```

***
