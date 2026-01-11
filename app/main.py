from fastapi import FastAPI
from app.api import routes
from app.core import state

app = FastAPI(title="AI Research Agent")

@app.on_event("startup")
async def startup_event():
    print("🚀 App started. Waiting for indexing...")

app.include_router(routes.router)
