import requests

# Define the GraphQL endpoint
url = "https://swapi-graphql.netlify.app/.netlify/functions/index"

# GraphQL query to fetch Star Wars characters and their species
query = """
{
  allPeople {
    people {
      name
      species {
        name
      }
    }
  }
}
"""

# Send the request
response = requests.post(url, json={'query': query})

# Check if the request was successful
if response.status_code == 200:
    data = response.json()
    print("Character Data:", data)
else:
    print("Failed to fetch data. Status code:", response.status_code)
