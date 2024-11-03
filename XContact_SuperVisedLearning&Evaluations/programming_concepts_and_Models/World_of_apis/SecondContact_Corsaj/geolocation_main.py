import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
import numpy as np

# Define approximate coordinates for the three event locations
event_coords = [
    (34.1425, -118.2551),  # Glendale, CA
    (34.1255, -118.2848),  # Los Feliz, CA
    (34.0522, -118.2437)   # Los Angeles, CA
]

# Extract latitude and longitude for plotting
event_lats = [coord[0] for coord in event_coords]
event_lons = [coord[1] for coord in event_coords]

# Convert coordinates to a numpy array for Delaunay triangulation
points = np.array(event_coords)

# Calculate the centroid (average of the event coordinates)
centroid = np.mean(points, axis=0)

# Plotting
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_title("Triangulation of Event Locations with Centroid")

# Plot each event location
ax.plot(event_lons, event_lats, 'ro', label="Event Locations")

# Plot the centroid as the calculated center
ax.plot(centroid[1], centroid[0], 'bo', label="Calculated Center (User)")

# Draw triangles based on Delaunay triangulation
tri = Delaunay(points)
for simplex in tri.simplices:
    ax.plot(points[simplex, 1], points[simplex, 0], 'g--')

# Draw lines to connect all pairs of points manually
for i in range(len(points)):
    for j in range(i + 1, len(points)):
        ax.plot([points[i, 1], points[j, 1]], [points[i, 0], points[j, 0]], 'g:')

# Customize plot
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.legend()
plt.grid(True)
plt.show()
