import numpy as np
import cv2
import matplotlib.pyplot as plt

# Define the color map used during training
color_maps = np.array([
    [0, 0, 0],         # background
    [51,30,0],     # bedroom/orange
    [50, 99, 155],     # hallway/purple
    [0,31,51],     # blue/bathroom
    [31, 51, 0],       # green/living
    [100, 80, 71]      # brown/empty-room
], dtype=np.uint8)

# Room type mapping
room_types = {
    0: "background",
    1: "bedroom",
    2: "hallway",
    3: "bathroom",
    4: "living_room",
    5: "empty_room"
}

def extract_room_dimensions(segmented_image, color_maps, room_types, pixel_to_meter_ratio=1.0):
    """
    Extract dimensions and properties for each room from segmented image
    """
    rooms_info = {}
    for class_idx, color in enumerate(color_maps):
        if class_idx == 0:
            continue
        mask = np.all(segmented_image == color, axis=-1)
        if not np.any(mask):
            continue
        num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
        for room_id in range(1, num_labels):
            room_mask = (labels == room_id)
            if np.sum(room_mask) < 50:
                continue
            room_props = analyze_room(room_mask, pixel_to_meter_ratio)
            room_key = f"{room_types[class_idx]}_{room_id}"
            rooms_info[room_key] = {
                'type': room_types[class_idx],
                'class_idx': class_idx,
                'color': color.tolist(),
                **room_props
            }
    return rooms_info


def analyze_room(room_mask, pixel_to_meter_ratio=1.0):
    """
    Analyze a single room mask to extract dimensions and properties
    """
    contours, _ = cv2.findContours(
        room_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {}
    main_contour = max(contours, key=cv2.contourArea)

    # Area
    area_pixels = cv2.contourArea(main_contour)
    area_units = area_pixels * (pixel_to_meter_ratio ** 2)

    # Rotated bounding rectangle
    rot_rect = cv2.minAreaRect(main_contour)
    (cx, cy), (w, h), angle = rot_rect
    width_units = w * pixel_to_meter_ratio
    height_units = h * pixel_to_meter_ratio

    # Centroid (from rotated rect)
    centroid = (int(cx), int(cy))

    # Aspect ratio and shape
    aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 1.0
    epsilon = 0.02 * cv2.arcLength(main_contour, True)
    approx = cv2.approxPolyDP(main_contour, epsilon, True)
    if len(approx) <= 4:
        shape = "rectangular"
    elif len(approx) <= 6:
        shape = "L_shaped"
    else:
        shape = "irregular"

    # Size category
    if area_units < 50:
        size_category = "small"
    elif area_units < 200:
        size_category = "medium"
    else:
        size_category = "large"

    return {
        'area_pixels': area_pixels,
        'area_units': area_units,
        'width_units': width_units,
        'height_units': height_units,
        'rotated_rect': rot_rect,
        'bounding_box': cv2.boxPoints(rot_rect).tolist(),
        'centroid': centroid,
        'aspect_ratio': aspect_ratio,
        'shape': shape,
        'size_category': size_category,
        'contour': main_contour,
        'vertices': len(approx)
    }


def visualize_room_analysis(image, rooms_info):
    """
    Visualize the room analysis results with rotated boxes
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    ax1.imshow(image)
    ax1.set_title("Segmented Floor Plan")
    ax1.axis('off')

    overlay = image.copy()
    for room_name, room_data in rooms_info.items():
        pts = np.array(room_data['bounding_box'], dtype=np.int32)
        cv2.drawContours(overlay, [pts], 0, (255, 255, 255), 2)
        cx, cy = room_data['centroid']
        cv2.circle(overlay, (cx, cy), 5, (255, 255, 255), -1)
        label = f"{room_data['type']}\n{room_data['area_units']:.1f}u²\n{room_data['width_units']:.1f}x{room_data['height_units']:.1f}"
        cv2.putText(overlay, room_data['type'], (int(cx), int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    ax2.imshow(overlay)
    ax2.set_title("Room Analysis Overlay (Rotated Boxes)")
    ax2.axis('off')
    plt.tight_layout()
    plt.show()


def print_room_summary(rooms_info):
    """
    Print a summary of all detected rooms
    """
    print("="*60)
    print("ROOM ANALYSIS SUMMARY")
    print("="*60)
    for rname, rdata in rooms_info.items():
        print(f"\n{rname.upper()}")
        print(f"  Type: {rdata['type']}")
        print(f"  Dimensions: {rdata['width_units']:.1f} x {rdata['height_units']:.1f} units")
        print(f"  Area: {rdata['area_units']:.1f} square units")
        print(f"  Shape: {rdata['shape']}")
        print(f"  Size Category: {rdata['size_category']}")
        print(f"  Aspect Ratio: {rdata['aspect_ratio']:.2f}")
        print(f"  Centroid: {rdata['centroid']}")


def process_floor_plan(segmented_image_path, pixel_to_meter_ratio=0.1, visualize=True):
    segmented = cv2.imread(segmented_image_path)
    segmented = cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB)
    rooms_info = extract_room_dimensions(segmented, color_maps, room_types, pixel_to_meter_ratio)
    if visualize:
        print_room_summary(rooms_info)
        visualize_room_analysis(segmented, rooms_info)
    return rooms_info

# Example usage:
# rooms_data = process_floor_plan('your_segmented_image.png', pixel_to_meter_ratio=0.1)
if __name__ == "__main__":
    import os
    segmented_image_path = os.path.join(os.path.dirname(__file__), 'segmented_rooms.png')
    rooms_data = process_floor_plan(segmented_image_path, pixel_to_meter_ratio=0.1, visualize=True)