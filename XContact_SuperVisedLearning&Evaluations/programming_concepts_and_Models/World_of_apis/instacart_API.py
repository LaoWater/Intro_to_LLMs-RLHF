import requests

# Define the endpoint URL
url = "https://connect.instacart.com/v2/fulfillment/stores/delivery"

# Set up the headers with the bearer token for authorization
headers = {
    "Accept": "application/json",
    "Authorization": "Bearer <token>",  # Replace <token> with your actual authorization token
    "Content-Type": "application/json"
}

# Define the request payload with either latitude/longitude or address details
data = {
    "find_by": {
        "latitude": 1,  # Replace with actual latitude
        "longitude": 1  # Replace with actual longitude
    }
}

# Send the POST request
response = requests.post(url, headers=headers, json=data)

# Check if the request was successful
if response.status_code == 200:
    # Parse the JSON response
    stores_info = response.json()
    print("Stores found:")
    for store in stores_info.get("stores", []):
        print(f"Name: {store['name']}")
        print(f"Location Code: {store['location_code']}")
        print(f"Supports Alcohol: {store['flags']['alcohol']}")
        print(f"Supports Pickup: {store['flags']['pickup']}")
        print(f"Pickup Only: {store['flags']['pickup_only']}")
        print(f"Long Distance Delivery: {store.get('flags', {}).get('long_distance_delivery', 'N/A')}")
        print("-" * 20)
else:
    print(f"Error {response.status_code}: {response.text}")
