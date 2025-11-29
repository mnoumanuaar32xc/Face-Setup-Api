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


