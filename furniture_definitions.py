import json
import os
import copy

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
        self.essential = False  # Flag for essential furniture items

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

# Define which items are essential for each room type
ESSENTIAL_FURNITURE = {
    "bedroom_master": ["bed", "study"],
    "bedroom_guest": ["singlebed", "study"],
    "bedroom": ["singlebed", "study"],
    "bathroom": ["bathtub", "sink"],
    "living_room": ["sofa", "tv"]
}

# Mapping of which furniture types belong in which rooms.
# You can customize these rules.
FURNITURE_ROOM_MAP = {
    "living_room": ["Lsofa", "sofa", "tv", "table", "diningtable", "kitchen", "stove", "sink"], # Removed chair
    "bedroom_master": ["bed", "bedside", "study", "table"], # Removed chair
    "bedroom_guest": ["singlebed", "bedside", "study", "table"], # Removed chair
    "bedroom": ["singlebed", "bedside", "study", "table"], # Removed chair, added table
    "bathroom": ["bathtub", "shower", "sink", "commode"],
    "hallway": [],
    "empty_room": [],
}

def ensure_essential_furniture(rooms_info, furniture_prototypes):
    """
    Ensures there's enough essential furniture for all rooms by duplicating items when needed.
    Returns a modified copy of the furniture_prototypes.
    """
    # Count rooms by type
    room_type_count = {}
    for room_name, room_data in rooms_info.items():
        room_type = room_data['type']
        room_type_count[room_type] = room_type_count.get(room_type, 0) + 1
    
    # Create a copy of the prototypes to modify
    prototypes_copy = copy.deepcopy(furniture_prototypes)
    
    # For each room type, ensure we have enough of each essential item
    for room_type, count in room_type_count.items():
        if room_type in ESSENTIAL_FURNITURE:
            for essential_item in ESSENTIAL_FURNITURE[room_type]:
                if essential_item in prototypes_copy and prototypes_copy[essential_item]:
                    # Make sure we have enough copies
                    while len(prototypes_copy[essential_item]) < count:
                        original = prototypes_copy[essential_item][0]
                        duplicate = copy.deepcopy(original)
                        duplicate.essential = True
                        prototypes_copy[essential_item].append(duplicate)
                        print(f"Duplicated essential furniture: {essential_item} for {room_type}")
    
    return prototypes_copy
