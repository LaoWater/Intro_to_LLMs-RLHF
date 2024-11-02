import requests

# Base API URL for JSONPlaceholder users
api_url = "https://jsonplaceholder.typicode.com/users"

# Perform a GET request to fetch all users
response = requests.get(api_url)
if response.status_code == 200:
    users = response.json()
    print("Fetched Users:", users)
else:
    print("Failed to fetch data. Status code:", response.status_code)

# Adding a new user with POST
new_user = {
    "name": "John Doe",
    "username": "johndoe",
    "email": "johndoe@example.com",
    "address": {
        "street": "123 Main St",
        "city": "Metropolis",
        "zipcode": "12345"
    }
}

# Send a POST request to add the new user
post_response = requests.post(api_url, json=new_user)
if post_response.status_code == 201:  # 201 status code means resource created
    created_user = post_response.json()
    print("New User Added:", created_user)
else:
    print("Failed to add new user. Status code:", post_response.status_code)

# Updating the user's information with PUT
updated_user = {
    "name": "John Doe Jr.",
    "username": "johndoejr",
    "email": "johnjr@example.com",
    "address": {
        "street": "123 Main St",
        "city": "Metropolis",
        "zipcode": "54321"
    }
}

# Assuming the user's ID is 1 for demonstration
user_id = 1
put_response = requests.put(f"{api_url}/{user_id}", json=updated_user)
if put_response.status_code == 200:
    print("User Updated:", put_response.json())
else:
    print("Failed to update user. Status code:", put_response.status_code)

# Deleting the user with DELETE
delete_response = requests.delete(f"{api_url}/{user_id}")
if delete_response.status_code == 200:
    print(f"User with ID {user_id} deleted successfully.")
else:
    print(f"Failed to delete user. Status code: {delete_response.status_code}")
