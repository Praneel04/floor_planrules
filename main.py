import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import copy

# Check for shapely dependency before other imports
try:
    import shapely
except ImportError:
    print("---")
    print("ERROR: The 'shapely' library is required but not installed in the current Python environment.")
    print("This can happen if you have multiple Python versions or are using a virtual environment.")
    print("\nTROUBLESHOOTING STEPS:")
    print("1. Use the specific Python interpreter running this script to install the package. Run this command in your terminal:")
    print(f"   \"{sys.executable}\" -m pip install shapely")
    print("\n2. If you are using a virtual environment (like venv), make sure it is activated before you run the script.")
    print("\n3. If you are using an IDE (like VS Code), check which Python interpreter is selected. It's often displayed in the status bar at the bottom. Ensure it matches the environment where you installed shapely.")
    print("---")
    sys.exit(1)

from room_analyzer import process_floor_plan, print_room_summary
from furniture_definitions import load_furniture_prototypes, FURNITURE_ROOM_MAP
from furniture_placer import FurniturePlacer

def visualize_final_layout(segmented_image, final_layout):
    """Visualize the final floor plan with furniture."""
    overlay = segmented_image.copy()
    
    for room_name, furniture_list in final_layout.items():
        for furniture in furniture_list:
            if furniture.position_px:
                # Create a rotated rectangle for the furniture
                rect = (furniture.position_px, (furniture.width_px, furniture.height_px), furniture.angle)
                box = cv2.boxPoints(rect)
                box = np.intp(box)
                
                cv2.drawContours(overlay, [box], 0, (255, 255, 255), 2)
                cv2.putText(overlay, furniture.name, furniture.position_px, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    plt.figure(figsize=(12, 12))
    plt.imshow(overlay)
    plt.title("Final Floor Plan with Furniture")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def calculate_dynamic_scale(rooms_info, furniture_prototypes, default_ratio):
    """
    Calculates a dynamic pixel_to_meter_ratio based on fitting the kitchen
    furniture to the shorter side of the living room.
    """
    # Find the living room
    living_room = None
    for room_name, room_data in rooms_info.items():
        if room_data['type'] == 'living_room':
            living_room = room_data
            break
    
    # Find the kitchen furniture prototype
    kitchen_protos = furniture_prototypes.get('kitchen')

    if not living_room or not kitchen_protos:
        print("\nWARNING: Could not find living room or kitchen furniture for dynamic scaling.")
        print(f"Falling back to default pixel_to_meter_ratio: {default_ratio}")
        return default_ratio

    kitchen_proto = kitchen_protos[0]
    # Get room's shorter side in pixels from its rotated bounding box
    _center, (w_px, h_px), _angle = living_room['rotated_rect']
    room_short_side_px = min(w_px, h_px)

    # Get kitchen's longer side in meters
    item_long_side_m = max(kitchen_proto.width_m, kitchen_proto.height_m)

    if room_short_side_px < 1:
        print("\nWARNING: Living room has zero size. Cannot calculate dynamic scale.")
        return default_ratio

    # Calculate new ratio to make the kitchen's length match the wall's length
    new_ratio = item_long_side_m / room_short_side_px
    print(f"\nDynamically calculated pixel_to_meter_ratio: {new_ratio:.4f}")
    print(f"(Based on fitting {kitchen_proto.name} ({item_long_side_m}m) to living room's shorter side ({room_short_side_px:.1f}px))")
    return new_ratio

def update_room_units(rooms_info, pixel_to_meter_ratio):
    """Updates all room dimension units after a new ratio is calculated."""
    for room_key in rooms_info:
        r_data = rooms_info[room_key]
        r_data['area_units'] = r_data['area_pixels'] * (pixel_to_meter_ratio ** 2)
        
        # Rotated rect width and height are already in pixels
        _c, (w_px, h_px), _angle = r_data['rotated_rect']
        r_data['width_units'] = w_px * pixel_to_meter_ratio
        r_data['height_units'] = h_px * pixel_to_meter_ratio
    return rooms_info

def differentiate_room_subtypes(rooms_info):
    """Identifies rooms of the same type and re-labels them based on size."""
    # Find all bedrooms
    bedrooms = []
    for room_key, room_data in rooms_info.items():
        if room_data['type'] == 'bedroom':
            bedrooms.append((room_key, room_data['area_pixels']))
    
    if len(bedrooms) > 1:
        # Sort by area, largest first
        bedrooms.sort(key=lambda x: x[1], reverse=True)
        
        # Re-label the largest as master
        master_key = bedrooms[0][0]
        rooms_info[master_key]['type'] = 'bedroom_master'
        print(f"\nIdentified {master_key} as Master Bedroom.")
        
        # Re-label others as guest
        for i in range(1, len(bedrooms)):
            guest_key = bedrooms[i][0]
            rooms_info[guest_key]['type'] = 'bedroom_guest'
            print(f"Identified {guest_key} as Guest Bedroom.")
            
    return rooms_info

def main():
    """Main function to run the furniture placement algorithm."""
    # --- Configuration ---
    base_path = os.path.dirname(__file__)
    segmented_image_path = os.path.join(base_path, 'segmented_rooms.png')
    furniture_json_path = os.path.join(base_path, 'furniture_crops', 'furniture_measurements.json')
    pixel_to_meter_ratio = 0.1 # Default value, will be replaced

    # 1. Analyze Rooms in PIXELS first
    print("1. Analyzing floor plan (in pixels)...")
    # Use a ratio of 1.0 so all 'unit' measurements are actually pixel measurements
    rooms_info = process_floor_plan(segmented_image_path, 1.0)

    # 2. Load Furniture Definitions
    print("\n2. Loading furniture definitions...")
    furniture_prototypes = load_furniture_prototypes(furniture_json_path)
    print(f"Loaded {sum(len(v) for v in furniture_prototypes.values())} furniture items.")

    # 2.5 Differentiate room subtypes
    print("\n2.5. Differentiating room subtypes...")
    rooms_info = differentiate_room_subtypes(rooms_info)

    # 3. Calculate Dynamic Scale
    print("\n3. Calculating dynamic scale...")
    pixel_to_meter_ratio = calculate_dynamic_scale(rooms_info, furniture_prototypes, pixel_to_meter_ratio)
    
    # 4. Update Room Measurements with new scale
    print("\n4. Updating room measurements with new scale...")
    rooms_info = update_room_units(rooms_info, pixel_to_meter_ratio)
    print_room_summary(rooms_info) # Print summary again with correct units

    # 5. Place Furniture
    print("\n5. Placing furniture...")
    # Create a DEEP copy of prototypes for the placer to safely modify
    furniture_for_placer = copy.deepcopy(furniture_prototypes)
    placer = FurniturePlacer(rooms_info, furniture_for_placer, FURNITURE_ROOM_MAP, pixel_to_meter_ratio, debug=True)
    final_layout = placer.place_all()
    print("Furniture placement complete.")

    # 6. Visualize Results
    print("\n6. Visualizing final layout...")
    segmented_image = cv2.imread(segmented_image_path)
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)
    visualize_final_layout(segmented_image, final_layout)

if __name__ == "__main__":
    main()
