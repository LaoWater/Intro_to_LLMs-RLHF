# Assuming `events` is the response from `local_evently.search_events`
events = [
    {'id': '35d193a894c1f80', 'name': 'Moonlight Rollerway', 'description': 'Roller skating at Moonlight Rollerway!', 'location': 'Glendale, CA', 'date': '2024-05-03', 'time': '19:00', 'category': 'Recreation', 'is_free': False},
    {'id': 'e3ab4c25c83a3c7', 'name': 'Bike Tour', 'description': 'Bike tour around Griffith Park with views of the Hollywood sign', 'location': 'Los Feliz, CA', 'date': '2024-04-29', 'time': '10:00', 'category': 'Sports', 'is_free': True},
    {'id': '9a92b384a647018', 'name': 'Dodgers Game', 'description': 'See the Dodgers play the Giants!', 'location': 'Los Angeles, CA', 'date': '2023-09-15', 'time': '19:00', 'category': 'Sports', 'is_free': False}
]

# Print formatted output
for event in events:
    print(f"Event ID: {event['id']}")
    print(f"Name: {event['name']}")
    print(f"Description: {event['description']}")
    print(f"Location: {event['location']}")
    print(f"Date: {event['date']}")
    print(f"Time: {event['time']}")
    print(f"Category: {event['category']}")
    print(f"Free Event: {'Yes' if event['is_free'] else 'No'}")
    print("-" * 40)  # Separator for clarity
