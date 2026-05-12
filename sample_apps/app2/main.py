from __future__ import annotations
from fastapi import FastAPI

app = FastAPI()
@app.get("/products")
async def products(page: int = 1, sort: str = "asc"):
    return {"page": page, "sort": sort}


