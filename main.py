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
from furniture_definitions import load_furniture_prototypes, FURNITURE_ROOM_MAP, ensure_essential_furniture
from furniture_placer import FurniturePlacer

def visualize_final_layout(segmented_image, final_layout, rooms_info):
    """Visualize the final floor plan with furniture schematically."""
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
    plt.title("Final Floor Plan with Furniture (Schematic)")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # After showing the schematic view, show the view with actual furniture images
    visualize_with_actual_furniture(segmented_image, final_layout)

def visualize_with_actual_furniture(segmented_image, final_layout):
    """Visualize the floor plan with actual furniture images from the crops folder."""
    print("\nGenerating visualization with actual furniture images...")
    
    # Create a canvas for the final visualization
    canvas = segmented_image.copy()
    
    # ENHANCEMENT: Add floor coloring similar to ground truth image
    # Use a light wooden floor color
    floor_color = (245, 222, 179)  # Light wooden floor color
    
    # Create a floor layer under the room colors
    for y in range(canvas.shape[0]):
        for x in range(canvas.shape[1]):
            # Only color non-black pixels (rooms)
            if not np.array_equal(canvas[y, x], [0, 0, 0]):
                # Apply a subtle wooden texture by varying color slightly
                variation = np.random.randint(-10, 10, 3)
                floor_color_var = np.clip(np.array(floor_color) + variation, 0, 255)
                # Blend the floor color with existing room color
                canvas[y, x] = canvas[y, x] * 0.3 + floor_color_var * 0.7

    # Find TV position for orienting sofas
    tv_position = None
    for room_name, furniture_list in final_layout.items():
        if room_name.startswith('living_room'):
            for f in furniture_list:
                if 'tv' in f.name.lower() and f.position_px:
                    tv_position = np.array(f.position_px)
                    break
            if tv_position is not None:
                break
    
    # Path to furniture crops folder
    furniture_path = os.path.join(os.path.dirname(__file__), 'furniture_crops')
    
    # Process each piece of furniture
    for room_name, furniture_list in final_layout.items():
        for furniture in furniture_list:
            if not furniture.position_px:
                continue
            
            # Load and prepare the furniture image
            # Find the matching image file based on original filename
            image_file = os.path.join(furniture_path, furniture.original_filename)
            if not os.path.exists(image_file):
                print(f"Warning: Image file not found for {furniture.name}: {image_file}")
                continue
            
            # Load furniture image
            furniture_img = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
            if furniture_img is None:
                print(f"Warning: Failed to load image for {furniture.name}: {image_file}")
                continue
            
            # Convert BGR to RGB if needed
            if len(furniture_img.shape) >= 3 and furniture_img.shape[2] == 3:
                furniture_img = cv2.cvtColor(furniture_img, cv2.COLOR_BGR2RGB)
            
            # IMPORTANT: Get the target shape from furniture dimensions
            target_width = furniture.width_px
            target_height = furniture.height_px
            
            # Determine whether to rotate/flip the image instead of resizing
            img_h, img_w = furniture_img.shape[:2]
            img_aspect = img_w / img_h
            target_aspect = target_width / target_height
            
            # If the aspect ratios are flipped, rotate the image 90 degrees
            # This preserves detail instead of stretching/squashing the image
            should_rotate = (img_aspect > 1 and target_aspect < 1) or (img_aspect < 1 and target_aspect > 1)
            
            if should_rotate:
                furniture_img = cv2.rotate(furniture_img, cv2.ROTATE_90_CLOCKWISE)
                img_h, img_w = furniture_img.shape[:2]
            
            # CRITICAL CHANGE: Special handling for sofas in living room
            angle_to_use = furniture.angle
            if ('sofa' in furniture.name.lower() and tv_position is not None and 
                furniture.position_px is not None and room_name.startswith('living_room')):
                
                # Use the existing furniture angle which should align with the room
                # The sofa placement algorithm already handles this
                pass
            
            # Scale the image to match target size while preserving aspect ratio
            scale_factor = min(target_width / img_w, target_height / img_h)
            new_w = int(img_w * scale_factor)
            new_h = int(img_h * scale_factor)
            
            try:
                resized_img = cv2.resize(furniture_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            except cv2.error:
                print(f"Warning: Error resizing image for {furniture.name}. Skipping.")
                continue
            
            # Create a blank image of the target size with transparent background
            if len(resized_img.shape) >= 3 and resized_img.shape[2] == 4:
                # With alpha channel
                padded_img = np.zeros((int(target_height), int(target_width), 4), dtype=np.uint8)
            else:
                # Without alpha channel
                padded_img = np.zeros((int(target_height), int(target_width), 3), dtype=np.uint8)
            
            # Center the resized image in the target-sized canvas
            y_offset = (int(target_height) - new_h) // 2
            x_offset = (int(target_width) - new_w) // 2
            
            # Place the resized image in the center of the padded canvas
            if y_offset >= 0 and x_offset >= 0 and y_offset + new_h <= padded_img.shape[0] and x_offset + new_w <= padded_img.shape[1]:
                padded_img[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img
            else:
                # Handle edge case where offsets would be negative
                print(f"Warning: Could not pad image for {furniture.name}. Using resized image directly.")
                padded_img = resized_img
            
            # Now rotate the padded image by the furniture angle
            # OpenCV rotation is counterclockwise, so we negate the angle
            rotation_angle = -angle_to_use  # Use the potentially adjusted angle for sofas
            center = (padded_img.shape[1] // 2, padded_img.shape[0] // 2)
            
            M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
            
            # Determine new bounding dimensions after rotation
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            new_w = int((padded_img.shape[0] * sin) + (padded_img.shape[1] * cos))
            new_h = int((padded_img.shape[0] * cos) + (padded_img.shape[1] * sin))
            
            # Adjust transformation matrix to center the result
            M[0, 2] += (new_w / 2) - center[0]
            M[1, 2] += (new_h / 2) - center[1]
            
            # Perform the rotation
            try:
                rotated_img = cv2.warpAffine(padded_img, M, (new_w, new_h), flags=cv2.INTER_LINEAR)
            except cv2.error:
                print(f"Warning: Error rotating image for {furniture.name}. Skipping.")
                continue
            
            # Calculate position to place the rotated furniture image
            x_pos = int(furniture.position_px[0] - (new_w / 2))
            y_pos = int(furniture.position_px[1] - (new_h / 2))
            
            # Make sure position is within canvas bounds
            if x_pos < 0 or y_pos < 0 or x_pos + new_w > canvas.shape[1] or y_pos + new_h > canvas.shape[0]:
                print(f"Warning: {furniture.name} extends outside the canvas boundaries.")
                # Try to adjust position to fit within bounds while maintaining center alignment
                x_pos = max(0, min(x_pos, canvas.shape[1] - new_w))
                y_pos = max(0, min(y_pos, canvas.shape[0] - new_h))
                if x_pos + new_w > canvas.shape[1] or y_pos + new_h > canvas.shape[0]:
                    # If still out of bounds after adjustment, skip
                    continue
                
            # If the image has an alpha channel, use it for blending
            if len(rotated_img.shape) >= 3 and rotated_img.shape[2] == 4:
                # Extract RGB and alpha channels
                rgb_img = rotated_img[:, :, :3]
                alpha = rotated_img[:, :, 3] / 255.0
                
                # Get the region of interest in the canvas
                try:
                    roi = canvas[y_pos:y_pos+new_h, x_pos:x_pos+new_w]
                    
                    # Check dimensions match
                    if roi.shape[:2] != rgb_img.shape[:2]:
                        print(f"Warning: Dimension mismatch for {furniture.name}. Skipping.")
                        continue
                    
                    # Blend using the alpha channel
                    for c in range(0, 3):
                        roi[:, :, c] = roi[:, :, c] * (1 - alpha) + rgb_img[:, :, c] * alpha
                    
                    # Update the canvas with the blended ROI
                    canvas[y_pos:y_pos+new_h, x_pos:x_pos+new_w] = roi
                except ValueError:
                    print(f"Warning: ROI error for {furniture.name}. Skipping.")
                    continue
            else:
                # Simple overlay without alpha blending
                # Ensure ROI doesn't exceed image dimensions
                # Ensure ROI doesn't exceed image dimensions
                y_end = min(y_pos + new_h, canvas.shape[0])
                x_end = min(x_pos + new_w, canvas.shape[1])
                h_to_use = y_end - y_pos
                w_to_use = x_end - x_pos
                
                if h_to_use <= 0 or w_to_use <= 0:
                    continue
                
                try:
                    canvas[y_pos:y_end, x_pos:x_end] = rotated_img[:h_to_use, :w_to_use]
                except ValueError:
                    print(f"Warning: Dimension mismatch for {furniture.name}. Skipping.")
                    continue
    
    # Display the visualization
    plt.figure(figsize=(12, 12))
    plt.imshow(canvas)
    plt.title("Final Floor Plan with Actual Furniture")
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

    # 4.5 Ensure essential furniture is available for all rooms
    print("\n4.5 Ensuring essential furniture is available for all rooms...")
    furniture_prototypes = ensure_essential_furniture(rooms_info, furniture_prototypes)
    
    # Count rooms by type for reference
    room_type_count = {}
    for room_name, room_data in rooms_info.items():
        room_type = room_data['type']
        room_type_count[room_type] = room_type_count.get(room_type, 0) + 1
    print(f"Room types detected: {room_type_count}")

    # 5. Place Furniture
    print("\n5. Placing furniture...")
    # Create a DEEP copy of prototypes for the placer to safely modify
    furniture_for_placer = copy.deepcopy(furniture_prototypes)
    
    placer = FurniturePlacer(rooms_info, furniture_for_placer, FURNITURE_ROOM_MAP, 
                           pixel_to_meter_ratio, debug=True)
    final_layout = placer.place_all()
    
    # Print summary of placed furniture by room
    print("\nFurniture Placement Summary:")
    for room_name, furniture_list in final_layout.items():
        print(f"{room_name}: {len(furniture_list)} pieces - {', '.join(f.name for f in furniture_list)}")
    print("Furniture placement complete.")

    # 6. Visualize Results
    print("\n6. Visualizing final layout...")
    segmented_image = cv2.imread(segmented_image_path)
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)
    visualize_final_layout(segmented_image, final_layout, rooms_info)  # Pass rooms_info here

if __name__ == "__main__":
    main()
