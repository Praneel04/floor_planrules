"""
Layout generator module for creating multiple furniture layouts from a single floor plan.
"""
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import copy
from pathlib import Path

# Import modules from main project
# Ensure we're in the right directory
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

from room_analyzer import process_floor_plan, print_room_summary
from furniture_definitions import load_furniture_prototypes, FURNITURE_ROOM_MAP, ensure_essential_furniture
from furniture_placer import FurniturePlacer
from main import visualize_final_layout, calculate_dynamic_scale, update_room_units, differentiate_room_subtypes

def generate_layouts(num_layouts=5, base_seed=None, output_dir=None):
    """Generate multiple layout variations from a single floor plan."""
    
    # Set paths
    base_path = os.path.dirname(__file__)
    segmented_image_path = os.path.join(base_path, 'segmented_rooms.png')
    furniture_json_path = os.path.join(base_path, 'furniture_crops', 'furniture_measurements.json')
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Set base seed
    if base_seed is None:
        base_seed = int(time.time())
    
    # Load segmented floor plan image
    segmented_image = cv2.imread(segmented_image_path)
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)
    
    # Process rooms once to get room info
    print("Analyzing floor plan...")
    rooms_info = process_floor_plan(segmented_image_path, 1.0)
    rooms_info = differentiate_room_subtypes(rooms_info)
    
    # Load furniture prototypes once
    print("Loading furniture definitions...")
    furniture_prototypes = load_furniture_prototypes(furniture_json_path)
    
    # Calculate dynamic scale
    print("Calculating dynamic scale...")
    pixel_to_meter_ratio = calculate_dynamic_scale(rooms_info, furniture_prototypes, 0.1)
    rooms_info = update_room_units(rooms_info, pixel_to_meter_ratio)
    
    # Style variations for the layouts
    style_options = [
        {'variation_level': 'low', 'style_preference': 'traditional', 'space_efficiency': 0.7},
        {'variation_level': 'medium', 'style_preference': 'modern', 'space_efficiency': 0.8},
        {'variation_level': 'high', 'style_preference': 'mixed', 'space_efficiency': 0.9},
    ]
    
    # Generate layouts
    layouts = []
    for i in range(num_layouts):
        print(f"\nGenerating layout {i+1}/{num_layouts}")
        
        # Set seed for this layout
        layout_seed = base_seed + i
        random.seed(layout_seed)
        np.random.seed(layout_seed)
        
        # Set style for this layout - cycle through options or random
        if i < len(style_options):
            layout_options = style_options[i]
        else:
            # Random style for additional layouts
            layout_options = {
                'variation_level': random.choice(['low', 'medium', 'high']),
                'style_preference': random.choice(['traditional', 'modern', 'mixed']),
                'space_efficiency': random.uniform(0.7, 0.9),
                'alignment_strictness': random.uniform(0.6, 1.0),
                'grouping_preference': random.uniform(0.5, 0.9),
            }
        
        print(f"Layout style: {layout_options['style_preference']}, Variation: {layout_options['variation_level']}")
        
        # Get essential furniture
        furniture_copy = ensure_essential_furniture(rooms_info, copy.deepcopy(furniture_prototypes))
        
        # Place furniture
        placer = FurniturePlacer(rooms_info, furniture_copy, FURNITURE_ROOM_MAP, 
                               pixel_to_meter_ratio, debug=True, layout_options=layout_options)
        final_layout = placer.place_all()
        
        # Store layout info
        layouts.append({
            'seed': layout_seed,
            'options': layout_options,
            'layout': final_layout
        })
        
        # Visualize this layout
        plt.figure(figsize=(12, 12))
        visualize_final_layout(segmented_image.copy(), final_layout)
        if output_dir:
            plt.savefig(os.path.join(output_dir, f"layout_{i+1}_seed_{layout_seed}.png"))
            plt.close()
        else:
            plt.show()
    
    return layouts

if __name__ == "__main__":
    # Process command line args
    num_layouts = 3  # Default
    output_dir = None  # Default to showing instead of saving
    
    if len(sys.argv) > 1:
        try:
            num_layouts = int(sys.argv[1])
        except ValueError:
            print("Warning: Invalid number of layouts. Using default (3).")
    
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
        
    # Generate layouts
    layouts = generate_layouts(num_layouts, output_dir=output_dir)
    print(f"Generated {len(layouts)} layout variations.")
