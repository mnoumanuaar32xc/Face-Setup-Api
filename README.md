# Face-Setup-Api

# step-by-step guide to create a FastAPI project named Fase_setup_Api inside your folder:

D:\Nouman\Training\Fast Api


✅ Step-by-Step Setup in VS Code
# 1. Open Your Project Folder in VS Code

Open VS Code

Click File → Open Folder

Select the folder:D:\Nouman\Training\Fast Api
Inside this folder, create your project folder:
D:\Nouman\Training\Fast Api\Fase_setup_Api

You can create it manually or from VS Code.

# 2. Create Project Structure

Inside Fase_setup_Api, create the following folders/files:

Fase_setup_Api
│
├── app
│   ├── __init__.py
│   ├── main.py
│   ├── routers
│   │     ├── __init__.py
│   │     └── sample_router.py
│   └── models
│         ├── __init__.py
│         └── sample_model.py
│
├── venv  (this will be created automatically later)
│
└── requirements.txt


✅ Step-by-Step Commands
# 3. Create Virtual Environment

Open VS Code terminal:

cd "D:\Nouman\Training\Fast Api\Fase_setup_Api"
Then run:

# python -m venv venv
Activate it:

# 4. Install FastAPI + Uvicorn
pip install fastapi uvicorn
pip freeze > requirements.txt

✅ Step-by-Step: Create Your Main API File
# 5. Open the file:
app/main.py
Add this code:

from fastapi import FastAPI
from app.routers import sample_router

app = FastAPI(
    title="Fase Setup API",
    version="1.0.0"
)

@app.get("/")
def root():
    return {"message": "Welcome to Fase Setup API"}

app.include_router(sample_router.router)



# ✅ Step-by-Step: Create a Router

Create file:
app/routers/sample_router.py

Add this:
from fastapi import APIRouter

router = APIRouter(
    prefix="/sample",
    tags=["Sample Routes"]
)

@router.get("/")
def sample_data():
    return {"info": "This is a sample router from Fase Setup API"}


# ✅ Step-by-Step: Create a Sample Model (Optional)

Create file:
app/models/sample_model.py

Add:

from pydantic import BaseModel

class SampleItem(BaseModel):
    name: str
    age: int
 
# ✅ 6. Run Your API

From terminal:

uvicorn app.main:app --reload


# After running, open these URLs:

✔ API Root

http://127.0.0.1:8000/

✔ Swagger Docs (API UI)

http://127.0.0.1:8000/docs



#################################################################################
# What Is a Vector Database?
A Vector DB stores and searches high-dimensional vectors generated from embeddings (text, images, etc.).


This is used for:

Semantic search

ChatGPT-style retrieval

Document similarity

Recommendation systems

Popular vector DB options include:

✔ Chroma DB (local, free, easy to use)
✔ FAISS (local, fast, by Facebook/Meta)
✔ Pinecone (cloud, scalable)
✔ Weaviate (open-source, cloud/local)

🔥 Which Vector DB Do You want to use?

Before I give you exact setup steps, tell me:

Choose 1 option:

ChromaDB (best for beginners, no server setup needed)

FAISS (fastest local vector DB)

Pinecone (cloud, scalable)

Weaviate

Milvus
# FAISS

# FAISS is possible locally, and it is one of the best choices for local vector search.
FAISS was created by Facebook/Meta, and it is extremely fast for:

Semantic Search

Document similarity

Image similarity

Embedding-based retrieval

You can fully use FAISS locally with FastAPI.
Below is a complete explanation + step-by-step setup for your project.

✅ FAISS Works Locally — 100% Possible

FAISS comes in two versions:

1️⃣ CPU version (works on all machines — Windows/Linux/macOS)
pip install faiss-cpu
# Step 1: Install FAISS + embeddings

Run this in your virtual environment:
pip install faiss-cpu sentence-transformers numpy

# Step 2: Create FAISS service file

Create:
app/services/vector_faiss.py

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create FAISS index (cosine similarity)
dimension = 384  # embedding size of MiniLM
index = faiss.IndexFlatIP(dimension)

# Store metadata in Python list
documents = []

def add_document(text: str):
    embedding = model.encode([text])
    embedding = embedding.astype('float32')

    index.add(embedding)
    documents.append(text)

    return {"status": "added", "id": len(documents)-1}


def search(query: str, top_k: int = 3):
    query_vec = model.encode([query]).astype('float32')
    scores, ids = index.search(query_vec, top_k)

    results = []
    for score, doc_id in zip(scores[0], ids[0]):
        if doc_id != -1:
            results.append({
                "id": int(doc_id),
                "text": documents[doc_id],
                "score": float(score)
            })

    return results



