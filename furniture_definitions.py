import json
import os

class Furniture:
    """Represents a single piece of furniture."""
    def __init__(self, name, width_m, height_m, original_filename):
        self.name = name
        self.width_m = width_m
        self.height_m = height_m
        self.original_filename = original_filename
        # These will be set during placement
        self.position_px = None
        self.angle = 0
        self.width_px = 0
        self.height_px = 0

    def __repr__(self):
        return f"Furniture({self.name}, {self.width_m:.2f}m x {self.height_m:.2f}m)"

def load_furniture_prototypes(json_path):
    """Loads all available furniture pieces from the JSON file."""
    with open(json_path, 'r') as f:
        measurements = json.load(f)

    prototypes = {}
    for filename, dims in measurements.items():
        # Cleans up name, e.g., "17_L-sofa.png" -> "Lsofa"
        name = ''.join(filter(str.isalpha, filename.split('.')[0]))
        
        furniture = Furniture(name, dims['width_m'], dims['height_m'], filename)
        
        if name not in prototypes:
            prototypes[name] = []
        prototypes[name].append(furniture)
    return prototypes

# Mapping of which furniture types belong in which rooms.
# You can customize these rules.
FURNITURE_ROOM_MAP = {
    "living_room": ["Lsofa", "sofa", "tv", "table", "diningtable", "kitchen", "stove", "sink", "chair"],
    "bedroom_master": ["bed", "bedside", "study", "chair"],
    "bedroom_guest": ["singlebed", "bedside", "study"],
    "bedroom": ["singlebed", "bedside", "study"], # Fallback for single bedrooms
    "bathroom": ["bathtub", "shower", "sink", "commode"],
    "hallway": [],
    "empty_room": [],
}
