from __future__ import annotations
from fastapi import FastAPI

app = FastAPI()
@app.get("/")
async def root():
    return {"status": "ok"}
@app.get("/search")
async def search(q: str = ""):
    return {"q": q}
@app.post("/login")
async def login(payload: dict):
    return {"ok": True}


