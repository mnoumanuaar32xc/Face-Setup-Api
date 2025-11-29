# Face-Setup-Api
# Object 1
# this is a samll project which we implement Fast API for upload , delete and get all images and store in Fassi Vector DB. and create a MVC .net Core Application for Upload and read the text from images


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
# 🟦 Step 3: Create FastAPI router

Create file:

app/routers/vector_faiss_router.py

Add:

from fastapi import APIRouter
from app.services.vector_faiss import add_document, search

router = APIRouter(prefix="/faiss", tags=["FAISS Vector DB"])

@router.post("/add")
def add_doc(text: str):
    return add_document(text)

@router.get("/search")
def search_doc(query: str, top_k: int = 3):
    return search(query, top_k)

#  🟦 Step 4: Include the FAISS router in main.py

In your app/main.py:
from app.routers import vector_faiss_router
app.include_router(vector_faiss_router.router)

# 🟦 Step 5: Run FastAPI
uvicorn app.main:app --reload

# Examples to Test
➤ Add a document
POST http://127.0.0.1:8000/faiss/add?text=This is my first document

➤ Search the vector database


############################################################

Below is a complete and clean FastAPI setup for your requirement:

✅ Create a new ImageProcessing.py service class
✅ Create new FastAPI endpoints in your router & main.py
✅ Accept an uploaded image from your MVC app
✅ Convert image → embedding
✅ Store it inside FAISS vector DB
✅ Provide an API to retrieve all stored images

📁 1. Create new file

app/services/ImageProcessing.py

Copy–paste the FULL working implementation:

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from PIL import Image
import io
import base64

# Load image embedding model
model = SentenceTransformer("clip-ViT-B-32")

# FAISS index
embedding_dim = 512  # CLIP ViT-B-32 output size
index = faiss.IndexFlatL2(embedding_dim)

# Store metadata (image bytes)
images_store = []


class ImageProcessing:

    @staticmethod
    def add_image(file_bytes: bytes):
        global images_store, index

        # Load image
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")

        # Compute embedding
        embedding = model.encode([img], convert_to_tensor=False)
        embedding = np.array(embedding).astype("float32")

        # Store image
        img_id = len(images_store)
        images_store.append(file_bytes)

        # Add vector to index
        index.add(embedding)

        return {
            "id": img_id,
            "message": "Image stored successfully"
        }

    @staticmethod
    def get_all_images():
        result = []
        for i, img_bytes in enumerate(images_store):
            # encode image in Base64 for returning in JSON
            base64_img = base64.b64encode(img_bytes).decode("utf-8")
            result.append({
                "id": i,
                "image_base64": base64_img
            })
        return result

        📁 2. Create router file

app/routers/image_router.py

from fastapi import APIRouter, UploadFile, File
from app.services.ImageProcessing import ImageProcessing

router = APIRouter(prefix="/image", tags=["Image Processing"])


@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    file_bytes = await file.read()
    return ImageProcessing.add_image(file_bytes)


@router.get("/all")
def get_all_images():
    return ImageProcessing.get_all_images()


📁 3. Update your main.py

Replace your router imports and include:

from fastapi import FastAPI
from app.routers import sample_router, vector_faiss_router, image_router

app = FastAPI(
    title="Fase Setup API",
    version="1.0.0"
)

@app.get("/")
def root():
    return {"message": "Welcome to Fase Setup API"}

# include your routers
app.include_router(sample_router.router)
app.include_router(vector_faiss_router.router)
app.include_router(image_router.router)


# 📱 4. MVC Upload Code (ASP.NET)

Use this to send image → your FastAPI endpoint.

public async Task<IActionResult> Upload(IFormFile imageFile)
{
    using (var client = new HttpClient())
    {
        var form = new MultipartFormDataContent();
        var stream = imageFile.OpenReadStream();
        form.Add(new StreamContent(stream), "file", imageFile.FileName);

        var response = await client.PostAsync("http://127.0.0.1:8000/image/upload", form);
        var result = await response.Content.ReadAsStringAsync();

        return Content(result);
    }
}

####################################################################################

# ✅ 2. Show the extracted text on the MVC page

This means you need:

✔ FastAPI OCR endpoint
✔ Python OCR logic
✔ MVC call to retrieve OCR result and display it
✔ UI box to show OCR text

🧠 STEP 1 — Install OCR library in FastAPI

Run in VS Code terminal:

pip install pytesseract pillow


Install Tesseract OCR engine (required):

🔸 Windows installer:

Download from here:
https://github.com/UB-Mannheim/tesseract/wiki

Install it.

Then add path in your Python code:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

Then add path in your Python code:

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

🧠 STEP 2 — Update ImageProcessing.py (Add OCR)

Add this inside ImageProcessing:

import pytesseract

# Add Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

class ImageProcessing:

    @staticmethod
    def extract_text(file_bytes: bytes):
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        text = pytesseract.image_to_string(img)
        return text

🧠 STEP 3 — Modify upload endpoint to return OCR text
app/routers/image_router.py
from fastapi import APIRouter, UploadFile, File
from app.services.ImageProcessing import ImageProcessing

router = APIRouter(prefix="/image", tags=["Image Processing"])

@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    file_bytes = await file.read()

    # Save image in FAISS
    result = ImageProcessing.add_image(file_bytes)

    # Extract text using OCR
    text = ImageProcessing.extract_text(file_bytes)

    return {
        "id": result["id"],
        "message": result["message"],
        "extracted_text": text
    }

🧠 STEP 4 — Modify MVC Upload to show OCR text
HomeController.cs
[HttpPost]
public async Task<IActionResult> Upload(IFormFile imageFile)
{
    if (imageFile == null || imageFile.Length == 0)
        return Content("No file selected.");

    using (var client = new HttpClient())
    {
        var form = new MultipartFormDataContent();
        var streamContent = new StreamContent(imageFile.OpenReadStream());
        form.Add(streamContent, "file", imageFile.FileName);

        var response = await client.PostAsync("http://127.0.0.1:8000/image/upload", form);

        var result = await response.Content.ReadAsStringAsync();

        return Content(result); // return JSON
    }
}

🧠 STEP 5 — Show extracted text on MVC UI

Modify your JavaScript:

$("#uploadBtn").click(function () {
    var fileInput = $("#imageFile")[0].files[0];
    if (!fileInput) {
        $("#response").text("Please select an image.");
        return;
    }

    var formData = new FormData();
    formData.append("imageFile", fileInput);

    $.ajax({
        url: "/Home/Upload",
        type: "POST",
        contentType: false,
        processData: false,
        data: formData,
        success: function (result) {
            let data = JSON.parse(result);
            $("#response").html(`
                <strong>Uploaded Successfully!</strong><br/>
                <b>OCR Extracted Text:</b><br/>
                <pre style="white-space: pre-wrap; background:#f0f0f0; padding:10px;">
${data.extracted_text}
                </pre>
            `);
        }
    });
});

🧠 STEP 6 — Add text box on page

Add in your Index.cshtml:

<h3>Extracted Text:</h3>
<div id="response" style="white-space:pre-wrap; background:#f7f7f7; padding:10px;"></div>

🎉 FINAL RESULT

✔ When user uploads image
✔ FastAPI extracts text using OCR
✔ MVC receives OCR text
✔ Text is shown under the image upload section

Exactly what you want!



Install Tesseract OCR engine (required):
