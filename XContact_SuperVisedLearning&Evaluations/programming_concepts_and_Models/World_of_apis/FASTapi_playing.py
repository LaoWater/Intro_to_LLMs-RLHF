import requests

BASE_URL = "http://127.0.0.1:8000/books"


# Function to get all books
def get_books():
    response = requests.get(BASE_URL)
    if response.status_code == 200:
        books = response.json()
        print("Books:", books)
        return books
    else:
        print("Failed to retrieve books.")
        return None


# Function to add a new book
def add_book(title, author, description, genre, price):
    new_book = {
        "title": title,
        "author": author,
        "description": description,
        "genre": genre,
        "price": price
    }
    response = requests.post(BASE_URL, json=new_book)
    if response.status_code == 200:
        print("Book added successfully:", response.json())
        return response.json()
    else:
        print("Failed to add book. Status code:", response.status_code)
        return None


# Function to get a specific book by ID
def get_book_by_id(book_id):
    response = requests.get(f"{BASE_URL}/{book_id}")
    if response.status_code == 200:
        book = response.json()
        print("Book details:", book)
        return book
    else:
        print("Book not found.")
        return None


# Function calls to interact with the API
if __name__ == "__main__":
    # Call to add a book
    added_book = add_book(
        title="Python FastAPI Guide",
        author="Jane Doe",
        description="A complete guide to FastAPI",
        genre="Technical",
        price=29.99
    )

    # Call to get all books
    all_books = get_books()

    # Call to get a specific book by ID if any books exist
    if all_books:
        first_book_id = 0  # Assuming the first book's ID is 0
        get_book_by_id(first_book_id)
