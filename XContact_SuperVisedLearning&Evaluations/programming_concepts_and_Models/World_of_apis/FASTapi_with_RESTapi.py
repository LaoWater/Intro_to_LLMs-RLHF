import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# Starting Server
# uvicorn FASTapi_with_RESTapi:app --reload

app = FastAPI()


# Define the data model
class Book(BaseModel):
    title: str
    author: str
    description: str
    genre: str
    price: float


# In-memory database (a simple list)
books = []


@app.get("/books")
def get_books():
    return books


@app.post("/books")
def add_book(book: Book):
    books.append(book)
    return {"message": "Book added successfully"}


@app.get("/books/{book_id}")
def get_book(book_id: int):
    if book_id < 0 or book_id >= len(books):
        raise HTTPException(status_code=404, detail="Book not found")
    return books[book_id]

