import numpy as np
from shapely.geometry import Polygon, Point, LineString
from shapely.affinity import rotate, translate
import copy

class FurniturePlacer:
    """
    Places furniture in rooms based on a set of rules.
    Requires `shapely` library: pip install shapely
    """
    def __init__(self, rooms_info, furniture_prototypes, furniture_map, pixel_to_meter_ratio, debug=False):
        self.rooms_info = rooms_info
        self.furniture_prototypes = furniture_prototypes
        self.furniture_map = furniture_map
        self.pixel_to_meter_ratio = pixel_to_meter_ratio
        self.placed_furniture = {}
        self.debug = debug
        self.placement_stats = {}  # Track statistics about placement success/failure

        # Convert all furniture dimensions from meters to pixels
        for name in self.furniture_prototypes:
            for f_item in self.furniture_prototypes[name]:
                f_item.width_px = f_item.width_m / self.pixel_to_meter_ratio
                f_item.height_px = f_item.height_m / self.pixel_to_meter_ratio

    def place_all(self):
        """Orchestrates furniture placement for all rooms."""
        # Initialize placement stats by room type, not room name
        for room_data in self.rooms_info.values():
            room_type = room_data['type']
            if room_type not in self.placement_stats:
                self.placement_stats[room_type] = {
                    'attempted': 0,
                    'placed': 0,
                    'failed': 0
                }
        
        # Now place furniture in each room
        for room_name, room_data in self.rooms_info.items():
            self.placed_furniture[room_name] = self._place_in_room(room_data)
            
        # Print placement statistics if debug is enabled
        if self.debug:
            print("\n--- Furniture Placement Statistics ---")
            for room_type, stats in self.placement_stats.items():
                success_rate = stats['placed'] / stats['attempted'] * 100 if stats['attempted'] > 0 else 0
                print(f"{room_type}: Placed {stats['placed']}/{stats['attempted']} items ({success_rate:.1f}%)")
                
        return self.placed_furniture

    def _get_furniture_for_room(self, room_type):
        """Gets a list of furniture objects for a given room type, consuming them from the pool."""
        furniture_needed = self.furniture_map.get(room_type, [])
        available_furniture = []
        
        # First try to get essential items
        for f_type in furniture_needed:
            if self.furniture_prototypes.get(f_type) and self.furniture_prototypes[f_type]:
                # Look for items marked as essential first
                essential_items = [f for f in self.furniture_prototypes[f_type] if hasattr(f, 'essential') and f.essential]
                if essential_items:
                    available_furniture.append(essential_items[0])
                    self.furniture_prototypes[f_type].remove(essential_items[0])
                    continue
                
                # Fall back to any available item
                available_furniture.append(self.furniture_prototypes[f_type].pop(0))
                
        return available_furniture

    def _get_walls(self, bounding_box):
        """Extracts wall segments from a room's bounding box."""
        walls = []
        for i in range(len(bounding_box)):
            p1 = np.array(bounding_box[i])
            p2 = np.array(bounding_box[(i + 1) % len(bounding_box)])
            walls.append((p1, p2))
        walls.sort(key=lambda w: np.linalg.norm(w[1] - w[0]), reverse=True)
        return walls

    def _place_in_room(self, room_data):
        """Delegates furniture placement to the correct function based on room type."""
        # --- Custom Rule for Living Room ---
        if room_data['type'] == 'living_room':
            return self._place_in_living_room(room_data)

        # --- Custom Rule for Bedrooms ---
        if room_data['type'] in ['bedroom_master', 'bedroom_guest', 'bedroom']:
            return self._place_in_bedroom(room_data)
            
        # --- Custom Rule for Bathroom ---
        if room_data['type'] == 'bathroom':
            return self._place_in_bathroom(room_data)

        # --- Default Placement for other rooms ---
        if self.debug: print(f"\n--- Applying default rules for {room_data['type']} ---")
        furniture_to_place = self._get_furniture_for_room(room_data['type'])
        furniture_to_place.sort(key=lambda f: f.width_px * f.height_px, reverse=True)
        
        placed_in_this_room = []
        placed_polygons = []
        
        # Use the actual room contour for placement checks, which is more accurate.
        if room_data.get('contour') is not None and len(room_data['contour']) > 2:
            room_polygon = Polygon(np.squeeze(room_data['contour']))
        else:
            room_polygon = Polygon(room_data['bounding_box']) # Fallback

        for f in furniture_to_place:
            self.placement_stats[room_data['type']]['attempted'] += 1
            is_placed, f_poly = self._place_against_wall(f, room_data, room_polygon, placed_polygons)
            if is_placed:
                placed_in_this_room.append(f)
                placed_polygons.append(f_poly)
                self.placement_stats[room_data['type']]['placed'] += 1
            else:
                self.placement_stats[room_data['type']]['failed'] += 1
        
        # Print debug summary
        if self.debug and not placed_in_this_room:
            print(f"WARNING: No furniture placed in {room_data['type']}.")
            
        # Print summary of what was placed
        if self.debug:
            furniture_counts = {}
            for item in placed_in_this_room:
                furniture_counts[item.name] = furniture_counts.get(item.name, 0) + 1
            print(f"Placed in {room_data['type']}: {', '.join(f'{count}x {name}' for name, count in furniture_counts.items())}")
                
        return placed_in_this_room

    def _place_in_bedroom(self, room_data):
        """Places furniture in a bedroom with enhanced study desk and table placement."""
        if self.debug: print(f"\n--- Applying custom rules for {room_data['type']} ---")
        
        placed_in_this_room = []
        placed_polygons = []
        
        if room_data.get('contour') is not None and len(room_data['contour']) > 2:
            room_polygon = Polygon(np.squeeze(room_data['contour']))
        else:
            room_polygon = Polygon(room_data['bounding_box'])

        # 1. Get all furniture and categorize by type
        all_furniture = self._get_furniture_for_room(room_data['type'])
        
        # Separate furniture by type for strategic placement
        bed_items = []
        study_items = []
        bedside_items = []
        other_furniture = []
        
        for item in all_furniture:
            if 'bed' in item.name:
                bed_items.append(item)
            elif 'study' in item.name:
                study_items.append(item)
            elif 'bedside' in item.name or 'table' in item.name:
                bedside_items.append(item)
            else:
                other_furniture.append(item)
        
        # 2. Place the bed first (compulsory)
        bed = bed_items[0] if bed_items else None
        if bed:
            self.placement_stats[room_data['type']]['attempted'] += 1
            if self.debug: print(f"Prioritizing placement of {bed.name}.")
            
            # Get the longest walls - beds are often placed against longer walls
            walls = self._get_walls(room_data['bounding_box'])
            long_walls = sorted(walls, key=lambda w: np.linalg.norm(w[1] - w[0]), reverse=True)[:2]
            
            # Try to place bed against the longest walls first
            placed = False
            for wall in long_walls:
                is_placed, bed_poly = self._place_item_on_wall(bed, wall, room_data, room_polygon, placed_polygons)
                if is_placed:
                    placed_in_this_room.append(bed)
                    placed_polygons.append(bed_poly)
                    self.placement_stats[room_data['type']]['placed'] += 1
                    placed = True
                    break
            
            # If failed with long walls, try any wall
            if not placed:
                is_placed, bed_poly = self._place_against_wall(bed, room_data, room_polygon, placed_polygons)
                if is_placed:
                    placed_in_this_room.append(bed)
                    placed_polygons.append(bed_poly)
                    self.placement_stats[room_data['type']]['placed'] += 1
                else:
                    self.placement_stats[room_data['type']]['failed'] += 1
                    if self.debug: print(f"CRITICAL: Failed to place compulsory item '{bed.name}' in {room_data['type']}.")
        else:
            if self.debug: print(f"WARNING: No bed available for {room_data['type']}.")
        
        # 3. Place bedside tables near the bed
        if bed in placed_in_this_room and bedside_items:
            if self.debug: print(f"Placing bedside tables near the bed.")
            
            # Find the direction perpendicular to bed orientation
            bed_pos = np.array(bed.position_px)
            bed_angle_rad = np.deg2rad(bed.angle)
            side_vec_norm = np.array([np.cos(bed_angle_rad + np.pi/2), np.sin(bed_angle_rad + np.pi/2)])
            
            # Try to place bedside tables on both sides of the bed
            for side_idx, side_direction in enumerate([-1, 1]):  # Left and right sides
                if not bedside_items:
                    break
                    
                bedside = bedside_items.pop(0)
                self.placement_stats[room_data['type']]['attempted'] += 1
                
                # Calculate position: slightly offset from the bed's side
                side_offset = (bed.width_px / 2) + (bedside.width_px / 2) + 5  # 5px gap
                side_pos = bed_pos + (side_vec_norm * side_direction * side_offset)
                
                # Try to place with same orientation as bed
                is_placed, table_poly = self._place_item_at_pos(bedside, side_pos, bed.angle, 
                                                             room_polygon, placed_polygons)
                if is_placed:
                    placed_in_this_room.append(bedside)
                    placed_polygons.append(table_poly)
                    self.placement_stats[room_data['type']]['placed'] += 1
                    if self.debug: print(f"Placed bedside table {side_idx+1} next to bed.")
                else:
                    # Fallback to general placement method
                    is_placed, table_poly = self._place_against_wall(bedside, room_data, room_polygon, placed_polygons)
                    if is_placed:
                        placed_in_this_room.append(bedside)
                        placed_polygons.append(table_poly)
                        self.placement_stats[room_data['type']]['placed'] += 1
                    else:
                        self.placement_stats[room_data['type']]['failed'] += 1
                        bedside_items.append(bedside)  # Return to the list of items to place
        
        # 4. Place study desk in a good position - opposite wall from bed or corner
        if study_items:
            study = study_items.pop(0)
            self.placement_stats[room_data['type']]['attempted'] += 1
            
            # Find the wall opposite to where the bed is placed
            placed = False
            if bed in placed_in_this_room:
                bed_pos = np.array(bed.position_px)
                walls = self._get_walls(room_data['bounding_box'])
                
                # Sort walls by distance from bed
                walls_with_distance = [(wall, np.linalg.norm(np.array([(wall[0][0] + wall[1][0])/2, 
                                                               (wall[0][1] + wall[1][1])/2]) - bed_pos)) 
                                 for wall in walls]
                walls_with_distance.sort(key=lambda x: x[1], reverse=True)
                
                # Try the wall furthest from the bed first
                for wall, _ in walls_with_distance[:2]:  # Try the two furthest walls
                    is_placed, study_poly = self._place_item_on_wall(study, wall, room_data, 
                                                                   room_polygon, placed_polygons)
                    if is_placed:
                        placed_in_this_room.append(study)
                        placed_polygons.append(study_poly)
                        self.placement_stats[room_data['type']]['placed'] += 1
                        if self.debug: print(f"Placed {study.name} on wall opposite to bed.")
                        placed = True
                        break
            
            # If not placed yet, try general placement
            if not placed:
                is_placed, study_poly = self._place_against_wall(study, room_data, room_polygon, placed_polygons)
                if is_placed:
                    placed_in_this_room.append(study)
                    placed_polygons.append(study_poly)
                    self.placement_stats[room_data['type']]['placed'] += 1
                    if self.debug: print(f"Placed {study.name} using general placement.")
                else:
                    self.placement_stats[room_data['type']]['failed'] += 1
                    if self.debug: print(f"Failed to place study desk in {room_data['type']}.")
        
        # 5. Try to place additional study items if space permits
        for study in study_items:
            self.placement_stats[room_data['type']]['attempted'] += 1
            is_placed, study_poly = self._place_against_wall(study, room_data, room_polygon, placed_polygons)
            if is_placed:
                placed_in_this_room.append(study)
                placed_polygons.append(study_poly)
                self.placement_stats[room_data['type']]['placed'] += 1
                if self.debug: print(f"Placed additional {study.name}.")
            else:
                self.placement_stats[room_data['type']]['failed'] += 1
        
        # 6. Place remaining furniture
        remaining_furniture = bedside_items + other_furniture
        remaining_furniture.sort(key=lambda f: f.width_px * f.height_px, reverse=True)
        
        for item in remaining_furniture:
            self.placement_stats[room_data['type']]['attempted'] += 1
            is_placed, item_poly = self._place_against_wall(item, room_data, room_polygon, placed_polygons)
            if is_placed:
                placed_in_this_room.append(item)
                placed_polygons.append(item_poly)
                self.placement_stats[room_data['type']]['placed'] += 1
            else:
                self.placement_stats[room_data['type']]['failed'] += 1
                
        if self.debug and not placed_in_this_room:
            print(f"WARNING: No furniture placed in {room_data['type']}.")
            
        # Print summary of what was placed
        if self.debug:
            furniture_counts = {}
            for item in placed_in_this_room:
                furniture_counts[item.name] = furniture_counts.get(item.name, 0) + 1
            print(f"Placed in {room_data['type']}: {', '.join(f'{count}x {name}' for name, count in furniture_counts.items())}")
                
        return placed_in_this_room

    def _place_in_living_room(self, room_data):
        """Places furniture in the living room according to specific, detailed rules."""
        if self.debug: print(f"\n--- Applying custom rules for Living Room ---")
        
        placed_in_this_room = []
        placed_polygons = []
        room_polygon = Polygon(room_data['bounding_box'])
        
        # 1. Get all furniture and separate it by type
        all_furniture = self._get_furniture_for_room(room_data['type'])
        
        def pop_items(name_list, source_list):
            items = []
            remaining = []
            for item in source_list:
                if item.name in name_list:
                    items.append(item)
                else:
                    remaining.append(item)
            items.sort(key=lambda f: f.width_px * f.height_px, reverse=True)
            return items, remaining

        kitchen_items, all_furniture = pop_items(['kitchen'], all_furniture)
        stove_items, all_furniture = pop_items(['stove'], all_furniture)
        tv_items, all_furniture = pop_items(['tv'], all_furniture)
        sofas, all_furniture = pop_items(['Lsofa', 'sofa'], all_furniture)
        tables, all_furniture = pop_items(['table', 'diningtable'], all_furniture)
        dining_tables, tables = pop_items(['diningtable'], tables)

        kitchen = kitchen_items[0] if kitchen_items else None
        stove = stove_items[0] if stove_items else None
        tv = tv_items[0] if tv_items else None
        
        # 2. Place Kitchen and Stove - FIXED PLACEMENT LOGIC
        if kitchen:
            walls = self._get_walls(room_data['bounding_box'])
            walls.sort(key=lambda w: np.linalg.norm(w[1] - w[0])) # Shortest to longest
            short_walls = walls[:2]

            if short_walls:
                target_wall = short_walls[0] # Pick one short wall
                wall_vec = target_wall[1] - target_wall[0]
                wall_len = np.linalg.norm(wall_vec)
                
                # Force orientation: ALWAYS make the kitchen's longer side its width
                # This ensures the longer side is parallel with the wall
                kitchen_long_side = max(kitchen.width_px, kitchen.height_px)
                kitchen_short_side = min(kitchen.width_px, kitchen.height_px)
                kitchen.width_px, kitchen.height_px = kitchen_long_side, kitchen_short_side

                is_placed, kitchen_poly = self._place_item_on_wall(kitchen, target_wall, room_data, room_polygon, placed_polygons, fixed_orientation=True)
                if is_placed:
                    if self.debug: print(f"Placed kitchen on a short wall with longer side parallel to wall.")
                    placed_in_this_room.append(kitchen)
                    placed_polygons.append(kitchen_poly)
                    
                    # FIXED STOVE PLACEMENT: Place stove directly beside kitchen with same orientation
                    if stove:
                        # Force stove to have EXACTLY the same orientation as kitchen
                        # First get the kitchen's orientation from its dimensions
                        kitchen_is_horizontal = kitchen.width_px >= kitchen.height_px
                        
                        # Set stove dimensions to match kitchen orientation
                        stove_long = max(stove.width_px, stove.height_px)
                        stove_short = min(stove.width_px, stove.height_px)
                        
                        if kitchen_is_horizontal:
                            # Make stove horizontal like kitchen
                            stove.width_px, stove.height_px = stove_long, stove_short
                        else:
                            # Make stove vertical like kitchen
                            stove.width_px, stove.height_px = stove_short, stove_long
                        
                        # Get the kitchen's wall-parallel direction vector
                        wall_unit_vec = wall_vec / wall_len
                        kitchen_pos = np.array(kitchen.position_px)
                        
                        # Try to place stove at different positions along the kitchen edge
                        # Try positions at different percentages along the kitchen width
                        placements = []
                        
                        # Try right side of kitchen (positive direction)
                        offset = kitchen.width_px / 2 + stove.width_px / 2 + 5  # 5px gap
                        pos_right = kitchen_pos + (wall_unit_vec * offset)
                        placements.append((pos_right, "right side"))
                        
                        # Try left side of kitchen (negative direction)
                        pos_left = kitchen_pos - (wall_unit_vec * offset)
                        placements.append((pos_left, "left side"))
                        
                        # Try positions with smaller and larger offsets
                        pos_right_closer = kitchen_pos + (wall_unit_vec * (offset - 10))
                        placements.append((pos_right_closer, "right side (closer)"))
                        
                        pos_left_closer = kitchen_pos - (wall_unit_vec * (offset - 10))
                        placements.append((pos_left_closer, "left side (closer)"))
                        
                        # Try placing stove at these positions
                        for stove_pos, position_name in placements:
                            is_placed, stove_poly = self._place_item_at_pos(
                                stove, stove_pos, kitchen.angle, room_polygon, placed_polygons)
                            
                            if is_placed:
                                if self.debug: print(f"Placed stove beside kitchen ({position_name}).")
                                placed_in_this_room.append(stove)
                                placed_polygons.append(stove_poly)
                                break
                                
                        # If direct placement failed, fall back to wall placement
                        if stove not in placed_in_this_room:
                            if self.debug: print("Direct stove placement failed. Trying wall placement.")
                            is_placed, stove_poly = self._place_against_wall(stove, room_data, room_polygon, placed_polygons)
                            if is_placed:
                                if self.debug: print("Placed stove using fallback method.")
                                placed_in_this_room.append(stove)
                                placed_polygons.append(stove_poly)
                else:
                    # Fallback to general wall placement for kitchen
                    is_placed, kitchen_poly = self._place_against_wall(kitchen, room_data, room_polygon, placed_polygons)
                    if is_placed:
                        placed_in_this_room.append(kitchen)
                        placed_polygons.append(kitchen_poly)

        # 3. Place TV on the longest wall, away from the kitchen
        if tv:
            walls = self._get_walls(room_data['bounding_box'])
            kitchen_pos = np.array(kitchen.position_px) if kitchen and kitchen.position_px else np.array([0,0])
            best_wall = sorted(walls, key=lambda w: np.linalg.norm(kitchen_pos - ((w[0]+w[1])/2)), reverse=True)[0]
            is_placed, tv_poly = self._place_item_on_wall(tv, best_wall, room_data, room_polygon, placed_polygons)
            if is_placed:
                if self.debug: print("Placed TV on longest wall away from kitchen.")
                placed_in_this_room.append(tv)
                placed_polygons.append(tv_poly)

        # 4. Place main Sofa, additional sofas, and coffee table
        if sofas and tv in placed_in_this_room:
            # Take largest sofa as main sofa
            main_sofa = sofas.pop(0)
            tv_pos = np.array(tv.position_px)
            centroid = np.array(room_polygon.centroid.coords[0])
            viewing_normal = (centroid - tv_pos) / np.linalg.norm(centroid - tv_pos) if np.linalg.norm(centroid - tv_pos) > 0 else np.array([0,1])
            
            viewing_dist = (main_sofa.height_px / 2) + 100 # 100px buffer
            sofa_pos = tv_pos + viewing_normal * viewing_dist
            
            is_placed, sofa_poly = self._place_item_at_pos(main_sofa, sofa_pos, tv.angle + 180, room_polygon, placed_polygons)
            if is_placed:
                if self.debug: print("Placed main sofa in front of TV.")
                placed_in_this_room.append(main_sofa)
                placed_polygons.append(sofa_poly)
                
                # Place coffee table between TV and sofa
                if tables:
                    coffee_table = tables.pop(0)
                    table_pos = tv_pos + viewing_normal * (viewing_dist / 2)
                    is_placed, table_poly = self._place_item_at_pos(coffee_table, table_pos, tv.angle, room_polygon, placed_polygons)
                    if is_placed:
                        if self.debug: print("Placed coffee table.")
                        placed_in_this_room.append(coffee_table)
                        placed_polygons.append(table_poly)
                    else:
                        tables.insert(0, coffee_table) # Put it back if it didn't fit
        
        # 5. Place dining table in center space
        if dining_tables:
            dining_table = dining_tables.pop(0)
            
            # Get the room's dominant orientation from walls
            walls = self._get_walls(room_data['bounding_box'])
            
            # Calculate the most common wall angle (dominant room orientation)
            wall_angles = []
            for wall in walls:
                wall_vec = wall[1] - wall[0]
                angle = np.rad2deg(np.arctan2(wall_vec[1], wall_vec[0])) % 180  # Normalize to 0-180
                wall_angles.append(angle)
            
            # Get main orientation angles (align with walls)
            main_angles = []
            for angle in wall_angles:
                # Get the base angle and its perpendicular
                main_angles.append(angle)
                main_angles.append((angle + 90) % 180)
            
            # Try to place dining table in the center of the remaining space
            room_center = np.array(room_polygon.centroid.coords[0])
            
            # Calculate position away from other furniture
            tv_pos = np.array(tv.position_px) if tv and tv.position_px else room_center
            kitchen_pos = np.array(kitchen.position_px) if kitchen and kitchen.position_px else room_center
            
            # Calculate vector away from both TV and kitchen
            vec_from_tv = room_center - tv_pos
            vec_from_kitchen = room_center - kitchen_pos
            combined_vec = vec_from_tv + vec_from_kitchen
            if np.linalg.norm(combined_vec) > 0:
                combined_vec = combined_vec / np.linalg.norm(combined_vec)
                target_pos = room_center + combined_vec * 80  # Place slightly offset from center
            else:
                target_pos = room_center  # Fallback to center
                
            # Try both orientations with the room's dominant angles
            orientations = [(dining_table.width_px, dining_table.height_px), 
                           (dining_table.height_px, dining_table.width_px)]
            
            placed = False
            # First try aligning with room walls
            for o_idx, (w, h) in enumerate(orientations):
                dining_table.width_px, dining_table.height_px = w, h
                # Try the dominant room angles first
                for angle in main_angles:
                    is_placed, dt_poly = self._place_item_at_pos(dining_table, target_pos, angle, room_polygon, placed_polygons)
                    if is_placed:
                        if self.debug: print(f"Placed dining table aligned with room at angle {angle:.1f}°")
                        placed_in_this_room.append(dining_table)
                        placed_polygons.append(dt_poly)
                        placed = True
                        break
                if placed:
                    break
                    
            # If still not placed, try arbitrary angles as fallback
            if not placed:
                for o_idx, (w, h) in enumerate(orientations):
                    dining_table.width_px, dining_table.height_px = w, h
                    for angle in [0, 45, 90, 135]:
                        is_placed, dt_poly = self._place_item_at_pos(dining_table, target_pos, angle, room_polygon, placed_polygons)
                        if is_placed:
                            if self.debug: print(f"Placed dining table at fallback angle {angle}°")
                            placed_in_this_room.append(dining_table)
                            placed_polygons.append(dt_poly)
                            placed = True
                            break
                    if placed:
                        break
                        
            # If central placement fails entirely, try against a wall
            if not placed:
                is_placed, dt_poly = self._place_against_wall(dining_table, room_data, room_polygon, placed_polygons)
                if is_placed:
                    if self.debug: print("Placed dining table against a wall (last resort).")
                    placed_in_this_room.append(dining_table)
                    placed_polygons.append(dt_poly)
                    
        # 6. Place remaining furniture (other tables, chairs, etc.)
        remaining_to_place = sofas + tables + dining_tables + all_furniture
        remaining_to_place.sort(key=lambda f: f.width_px * f.height_px, reverse=True)
        
        for f in remaining_to_place:
            is_placed, f_poly = self._place_against_wall(f, room_data, room_polygon, placed_polygons)
            if is_placed:
                placed_in_this_room.append(f)
                placed_polygons.append(f_poly)

        return placed_in_this_room

    def _place_in_bathroom(self, room_data):
        """Places furniture in a bathroom with specific rules for sink orientation."""
        if self.debug: print(f"\n--- Applying custom rules for bathroom ---")
        
        placed_in_this_room = []
        placed_polygons = []
        
        if room_data.get('contour') is not None and len(room_data['contour']) > 2:
            room_polygon = Polygon(np.squeeze(room_data['contour']))
        else:
            room_polygon = Polygon(room_data['bounding_box'])

        # Get all bathroom furniture items
        all_furniture = self._get_furniture_for_room(room_data['type'])
        
        # Categorize bathroom furniture by type
        sink_items = []
        bathtub_items = []
        shower_items = []
        commode_items = []
        other_furniture = []
        
        for item in all_furniture:
            if 'sink' in item.name:
                sink_items.append(item)
            elif 'bathtub' in item.name:
                bathtub_items.append(item)
            elif 'shower' in item.name:
                shower_items.append(item)
            elif 'commode' in item.name:
                commode_items.append(item)
            else:
                other_furniture.append(item)
        
        # Place the sink first with specific orientation
        for sink in sink_items:
            self.placement_stats[room_data['type']]['attempted'] += 1
            if self.debug: print(f"Placing sink with longest side against wall.")
            
            # Ensure the sink's longest side is parallel to the wall
            sink_long_side = max(sink.width_px, sink.height_px)
            sink_short_side = min(sink.width_px, sink.height_px)
            
            # Force orientation: width (parallel to wall) = long side, height (perpendicular) = short side
            sink.width_px, sink.height_px = sink_long_side, sink_short_side
            
            # Try all walls with this fixed orientation
            walls = self._get_walls(room_data['bounding_box'])
            placed = False
            
            for wall in walls:
                is_placed, sink_poly = self._place_item_on_wall(sink, wall, room_data, room_polygon, 
                                                              placed_polygons, fixed_orientation=True)
                if is_placed:
                    placed_in_this_room.append(sink)
                    placed_polygons.append(sink_poly)
                    self.placement_stats[room_data['type']]['placed'] += 1
                    if self.debug: print(f"Successfully placed sink with longer side ({sink_long_side:.1f}px) against wall.")
                    placed = True
                    break
                    
            if not placed:
                if self.debug: print(f"Failed to place sink with fixed orientation, trying default placement.")
                is_placed, sink_poly = self._place_against_wall(sink, room_data, room_polygon, placed_polygons)
                if is_placed:
                    placed_in_this_room.append(sink)
                    placed_polygons.append(sink_poly)
                    self.placement_stats[room_data['type']]['placed'] += 1
                else:
                    self.placement_stats[room_data['type']]['failed'] += 1
        
        # Place bathtub against a wall
        for bathtub in bathtub_items:
            self.placement_stats[room_data['type']]['attempted'] += 1
            if self.debug: print(f"Placing bathtub against wall.")
            
            # Bathtubs typically have their longer side against the wall
            bathtub_long_side = max(bathtub.width_px, bathtub.height_px)
            bathtub_short_side = min(bathtub.width_px, bathtub.height_px)
            bathtub.width_px, bathtub.height_px = bathtub_long_side, bathtub_short_side
            
            # Try to place against walls, prioritizing longer walls
            walls = self._get_walls(room_data['bounding_box'])
            is_placed, bathtub_poly = self._place_against_wall(bathtub, room_data, room_polygon, placed_polygons)
            if is_placed:
                placed_in_this_room.append(bathtub)
                placed_polygons.append(bathtub_poly)
                self.placement_stats[room_data['type']]['placed'] += 1
                if self.debug: print(f"Placed {bathtub.name} against wall.")
            else:
                self.placement_stats[room_data['type']]['failed'] += 1
        
        # Place shower in a corner if possible
        for shower in shower_items:
            self.placement_stats[room_data['type']]['attempted'] += 1
            if self.debug: print(f"Placing shower, preferably in a corner.")
            
            # Showers are often placed in corners
            walls = self._get_walls(room_data['bounding_box'])
            
            # Try to find corners (where walls meet)
            corners = []
            for i, wall1 in enumerate(walls):
                for j, wall2 in enumerate(walls):
                    if i != j:
                        # Check if walls share a point
                        if np.allclose(wall1[0], wall2[0]) or np.allclose(wall1[0], wall2[1]) or \
                           np.allclose(wall1[1], wall2[0]) or np.allclose(wall1[1], wall2[1]):
                            # Get the shared point
                            if np.allclose(wall1[0], wall2[0]) or np.allclose(wall1[0], wall2[1]):
                                corner = wall1[0]
                            else:
                                corner = wall1[1]
                            corners.append((corner, wall1, wall2))
            
            placed = False
            # Try to place shower in corners
            if corners:
                for corner_point, wall1, wall2 in corners:
                    # Place slightly offset from corner
                    wall1_vec = wall1[1] - wall1[0]
                    wall2_vec = wall2[1] - wall2[0]
                    wall1_unit = wall1_vec / np.linalg.norm(wall1_vec)
                    wall2_unit = wall2_vec / np.linalg.norm(wall2_vec)
                    
                    # Get position offset from corner
                    offset = shower.width_px / 2
                    pos = corner_point + (wall1_unit + wall2_unit) * offset
                    
                    # Try both orientations
                    for orientation_idx in range(2):
                        if orientation_idx == 1:
                            shower.width_px, shower.height_px = shower.height_px, shower.width_px
                        
                        # Try a few angles
                        for angle_offset in [0, 45]:
                            wall1_angle = np.rad2deg(np.arctan2(wall1_vec[1], wall1_vec[0]))
                            angle = wall1_angle + angle_offset
                            
                            is_placed, shower_poly = self._place_item_at_pos(shower, pos, angle, 
                                                                          room_polygon, placed_polygons)
                            if is_placed:
                                placed_in_this_room.append(shower)
                                placed_polygons.append(shower_poly)
                                self.placement_stats[room_data['type']]['placed'] += 1
                                if self.debug: print(f"Placed {shower.name} near corner.")
                                placed = True
                                break
                        
                        if placed:
                            break
                    
                    if placed:
                        break
            
            # If not placed in a corner, try general wall placement
            if not placed:
                is_placed, shower_poly = self._place_against_wall(shower, room_data, room_polygon, placed_polygons)
                if is_placed:
                    placed_in_this_room.append(shower)
                    placed_polygons.append(shower_poly)
                    self.placement_stats[room_data['type']]['placed'] += 1
                    if self.debug: print(f"Placed {shower.name} against wall.")
                else:
                    self.placement_stats[room_data['type']]['failed'] += 1
        
        # Place commode against a wall
        for commode in commode_items:
            self.placement_stats[room_data['type']]['attempted'] += 1
            is_placed, commode_poly = self._place_against_wall(commode, room_data, room_polygon, placed_polygons)
            if is_placed:
                placed_in_this_room.append(commode)
                placed_polygons.append(commode_poly)
                self.placement_stats[room_data['type']]['placed'] += 1
                if self.debug: print(f"Placed {commode.name} against wall.")
            else:
                self.placement_stats[room_data['type']]['failed'] += 1
        
        # Place other bathroom items
        for item in other_furniture:
            self.placement_stats[room_data['type']]['attempted'] += 1
            is_placed, item_poly = self._place_against_wall(item, room_data, room_polygon, placed_polygons)
            if is_placed:
                placed_in_this_room.append(item)
                placed_polygons.append(item_poly)
                self.placement_stats[room_data['type']]['placed'] += 1
            else:
                self.placement_stats[room_data['type']]['failed'] += 1
        
        # Print summary of what was placed
        if self.debug:
            furniture_counts = {}
            for item in placed_in_this_room:
                furniture_counts[item.name] = furniture_counts.get(item.name, 0) + 1
            print(f"Placed in bathroom: {', '.join(f'{count}x {name}' for name, count in furniture_counts.items())}")
                
        return placed_in_this_room

    def _find_closest_wall_to_item(self, item, walls):
        """Find which wall an item is placed against"""
        if not item.position_px:
            return None
            
        item_pos = np.array(item.position_px)
        closest_wall = min(walls, key=lambda w: self._point_to_wall_distance(item_pos, w))
        return closest_wall
        
    def _point_to_wall_distance(self, point, wall):
        """Calculate distance from a point to a wall segment"""
        p1, p2 = wall
        line = LineString([p1, p2])
        p = Point(point)
        return p.distance(line)

    def _place_against_wall(self, furniture, room_data, room_polygon, placed_polygons):
        """A simple strategy to place furniture against a wall."""
        if self.debug: print(f"-> Attempting to place {furniture.name}")
        walls = self._get_walls(room_data['bounding_box'])
        
        for i, wall in enumerate(walls):
            if self.debug: print(f" - Trying wall {i+1}/{len(walls)}")
            is_placed, f_poly = self._place_item_on_wall(furniture, wall, room_data, room_polygon, placed_polygons, fixed_orientation=False)
            if is_placed:
                return True, f_poly
        
        if self.debug: print(f"-> FAILED to place {furniture.name}")
        return False, None

    def _place_item_on_wall(self, furniture, wall, room_data, room_polygon, placed_polygons, fixed_orientation=False):
        """Tries to place a single item on a specific wall, trying multiple positions."""
        wall_vec = wall[1] - wall[0]
        wall_len = np.linalg.norm(wall_vec)
        
        if fixed_orientation:
            orientations = [(furniture.width_px, furniture.height_px)]
        else:
            orientations = [(furniture.width_px, furniture.height_px), (furniture.height_px, furniture.width_px)]
            if furniture.width_px == furniture.height_px:
                orientations = orientations[:1]

        for f_w, f_h in orientations:
            if f_w > wall_len:
                if self.debug: print(f"  - Orientation {f_w:.1f}x{f_h:.1f}px on wall (len {wall_len:.1f}px): FAILED (too wide)")
                continue

            wall_angle = np.rad2deg(np.arctan2(wall_vec[1], wall_vec[0]))
            
            # Try placing at multiple points along the wall to find a free spot
            for t in [0.5, 0.25, 0.75, 0.1, 0.9]: # Try center, then quarters, then edges
                pos = wall[0] + wall_vec * t
                
                normal = self._get_wall_normal(pos, room_polygon, wall_vec)
                pos_adjusted = pos + normal * (f_h / 2)

                temp_f = type('obj', (object,), {'width_px': f_w, 'height_px': f_h})()
                f_poly = self._get_furniture_polygon(temp_f, pos_adjusted, wall_angle)

                collision = any(f_poly.intersects(p) for p in placed_polygons)
                is_inside = room_polygon.contains(f_poly.centroid)
                
                if not collision and is_inside:
                    furniture.position_px = (int(pos_adjusted[0]), int(pos_adjusted[1]))
                    furniture.angle = wall_angle
                    furniture.width_px, furniture.height_px = f_w, f_h
                    if self.debug: print(f"  - Orientation {f_w:.1f}x{f_h:.1f}px at wall pos {t*100:.0f}%: SUCCESS")
                    return True, f_poly
            
            # If all positions for this orientation failed, log it
            if self.debug:
                print(f"  - Orientation {f_w:.1f}x{f_h:.1f}px: FAILED (no free space found along wall)")

        return False, None
    
    def _get_wall_normal(self, point, room_polygon, wall_vec):
        """Gets a normal vector pointing into the room from a wall."""
        normal = np.array([-wall_vec[1], wall_vec[0]])
        # Normalize the vector to length 1
        norm_val = np.linalg.norm(normal)
        if norm_val > 0:
            normal = normal / norm_val
        
        # Ensure normal points into the room polygon by checking against the centroid.
        centroid = np.array(room_polygon.centroid.coords[0])
        vec_to_centroid = centroid - point
        if np.dot(normal, vec_to_centroid) < 0:
            normal = -normal # Flip the normal if it's pointing away from the center
        return normal

    def _place_item_at_pos(self, furniture, pos, angle, room_polygon, placed_polygons):
        """Tries to place an item at a specific position and angle."""
        f_poly = self._get_furniture_polygon(furniture, pos, angle)
        collision = any(f_poly.intersects(p) for p in placed_polygons)
        if not collision and room_polygon.contains(f_poly.centroid):
            furniture.position_px = (int(pos[0]), int(pos[1]))
            furniture.angle = angle
            return True, f_poly
        return False, None
        
    def _get_furniture_polygon(self, furniture, center_pos, angle):
        """Creates a shapely Polygon for a furniture item."""
        w, h = furniture.width_px, furniture.height_px
        corners = [(-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)]
        poly = Polygon(corners)
        poly = rotate(poly, angle, origin='center', use_radians=False)
        poly = translate(poly, xoff=center_pos[0], yoff=center_pos[1])
        return poly
