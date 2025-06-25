import numpy as np
from shapely.geometry import Polygon, Point
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

        # Convert all furniture dimensions from meters to pixels
        for name in self.furniture_prototypes:
            for f_item in self.furniture_prototypes[name]:
                f_item.width_px = f_item.width_m / self.pixel_to_meter_ratio
                f_item.height_px = f_item.height_m / self.pixel_to_meter_ratio

    def place_all(self):
        """Orchestrates furniture placement for all rooms."""
        for room_name, room_data in self.rooms_info.items():
            self.placed_furniture[room_name] = self._place_in_room(room_data)
        return self.placed_furniture

    def _get_furniture_for_room(self, room_type):
        """Gets a list of furniture objects for a given room type, consuming them from the pool."""
        furniture_needed = self.furniture_map.get(room_type, [])
        available_furniture = []
        for f_type in furniture_needed:
            # Pop one item of the required type from the central pool. This fixes the empty rooms bug.
            if self.furniture_prototypes.get(f_type) and self.furniture_prototypes[f_type]:
                available_furniture.append(self.furniture_prototypes[f_type].pop(0))
        return available_furniture

    def _place_in_room(self, room_data):
        """Delegates furniture placement to the correct function based on room type."""
        # --- Custom Rule for Living Room ---
        if room_data['type'] == 'living_room':
            return self._place_in_living_room(room_data)

        # --- Custom Rule for Bedrooms ---
        if room_data['type'] in ['bedroom_master', 'bedroom_guest', 'bedroom']:
            return self._place_in_bedroom(room_data)

        # --- Default Placement for other rooms (e.g., bathroom) ---
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
            is_placed, f_poly = self._place_against_wall(f, room_data, room_polygon, placed_polygons)
            if is_placed:
                placed_in_this_room.append(f)
                placed_polygons.append(f_poly)
        
        return placed_in_this_room

    def _place_in_bedroom(self, room_data):
        """Places furniture in a bedroom, prioritizing the bed."""
        if self.debug: print(f"\n--- Applying custom rules for {room_data['type']} ---")
        
        placed_in_this_room = []
        placed_polygons = []
        
        if room_data.get('contour') is not None and len(room_data['contour']) > 2:
            room_polygon = Polygon(np.squeeze(room_data['contour']))
        else:
            room_polygon = Polygon(room_data['bounding_box'])

        # 1. Get all furniture and separate the bed
        all_furniture = self._get_furniture_for_room(room_data['type'])
        
        bed = None
        other_furniture = []
        for item in all_furniture:
            if 'bed' in item.name and bed is None: # Catches 'bed' and 'singlebed', takes the first one
                bed = item
            else:
                other_furniture.append(item)
        
        # 2. Place the bed first (compulsory)
        if bed:
            if self.debug: print(f"Prioritizing placement of {bed.name}.")
            is_placed, bed_poly = self._place_against_wall(bed, room_data, room_polygon, placed_polygons)
            if is_placed:
                placed_in_this_room.append(bed)
                placed_polygons.append(bed_poly)
            else:
                if self.debug: print(f"CRITICAL: Failed to place compulsory item '{bed.name}' in {room_data['type']}.")
        
        # 3. Place remaining furniture
        other_furniture.sort(key=lambda f: f.width_px * f.height_px, reverse=True)
        for f in other_furniture:
            is_placed, f_poly = self._place_against_wall(f, room_data, room_polygon, placed_polygons)
            if is_placed:
                placed_in_this_room.append(f)
                placed_polygons.append(f_poly)
                
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

        kitchen = kitchen_items[0] if kitchen_items else None
        stove = stove_items[0] if stove_items else None
        tv = tv_items[0] if tv_items else None
        
        # 2. Place Kitchen and Stove
        if kitchen:
            walls = self._get_walls(room_data['bounding_box'])
            walls.sort(key=lambda w: np.linalg.norm(w[1] - w[0])) # Shortest to longest
            short_walls = walls[:2]

            if short_walls:
                target_wall = short_walls[0] # Pick one short wall
                
                # Force orientation: make the kitchen's longer side its width.
                if kitchen.height_px > kitchen.width_px:
                    kitchen.width_px, kitchen.height_px = kitchen.height_px, kitchen.width_px

                is_placed, kitchen_poly = self._place_item_on_wall(kitchen, target_wall, room_data, room_polygon, placed_polygons, fixed_orientation=True)
                if is_placed:
                    if self.debug: print(f"Placed kitchen on a short wall with fixed orientation.")
                    placed_in_this_room.append(kitchen)
                    placed_polygons.append(kitchen_poly)
                    if stove:
                        # --- NEW STOVE LOGIC ---
                        # Force stove orientation to be horizontal
                        if stove.height_px > stove.width_px:
                            stove.width_px, stove.height_px = stove.height_px, stove.width_px
                        
                        # Fallback to general placement, which will find a spot near the kitchen
                        is_placed, stove_poly = self._place_against_wall(stove, room_data, room_polygon, placed_polygons)
                        if is_placed:
                            if self.debug: print("Placed stove near kitchen.")
                            placed_in_this_room.append(stove)
                            placed_polygons.append(stove_poly)

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

        # 4. Place main Sofa and a coffee Table
        if sofas and tv in placed_in_this_room:
            main_sofa = sofas.pop(0) # Take the largest sofa
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

        # 5. Place remaining furniture (other sofas, tables, chairs, etc.)
        remaining_to_place = sofas + tables + all_furniture
        remaining_to_place.sort(key=lambda f: f.width_px * f.height_px, reverse=True)
        
        for f in remaining_to_place:
            is_placed, f_poly = self._place_against_wall(f, room_data, room_polygon, placed_polygons)
            if is_placed:
                placed_in_this_room.append(f)
                placed_polygons.append(f_poly)

        return placed_in_this_room

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
                pos += normal * (f_h / 2)

                temp_f = type('obj', (object,), {'width_px': f_w, 'height_px': f_h})()
                f_poly = self._get_furniture_polygon(temp_f, pos, wall_angle)

                collision = any(f_poly.intersects(p) for p in placed_polygons)
                is_inside = room_polygon.contains(f_poly.centroid)
                
                if not collision and is_inside:
                    furniture.position_px = (int(pos[0]), int(pos[1]))
                    furniture.angle = wall_angle
                    furniture.width_px, furniture.height_px = f_w, f_h
                    if self.debug: print(f"  - Orientation {f_w:.1f}x{f_h:.1f}px at wall pos {t*100:.0f}%: SUCCESS")
                    return True, f_poly
            
            # If all positions for this orientation failed, log it
            if self.debug:
                print(f"  - Orientation {f_w:.1f}x{f_h:.1f}px: FAILED (no free space found along wall)")

        return False, None

    def _place_item_at_pos(self, furniture, pos, angle, room_polygon, placed_polygons):
        """Tries to place an item at a specific position and angle."""
        f_poly = self._get_furniture_polygon(furniture, pos, angle)
        collision = any(f_poly.intersects(p) for p in placed_polygons)
        if not collision and room_polygon.contains(f_poly.centroid):
            furniture.position_px = (int(pos[0]), int(pos[1]))
            furniture.angle = angle
            return True, f_poly
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

    def _get_wall_vector_from_item(self, item, walls):
        """Finds the wall vector an item is placed against."""
        item_pos = np.array(item.position_px)
        # Find the wall whose center is closest to the item's position
        closest_wall = min(walls, key=lambda w: np.linalg.norm(item_pos - (w[0]+w[1])/2))
        return closest_wall[1] - closest_wall[0]

    def _get_walls(self, bounding_box):
        """Extracts wall segments from a room's bounding box."""
        walls = []
        for i in range(len(bounding_box)):
            p1 = np.array(bounding_box[i])
            p2 = np.array(bounding_box[(i + 1) % len(bounding_box)])
            walls.append((p1, p2))
        walls.sort(key=lambda w: np.linalg.norm(w[1] - w[0]), reverse=True)
        return walls

    def _get_furniture_polygon(self, furniture, center_pos, angle):
        """Creates a shapely Polygon for a furniture item."""
        w, h = furniture.width_px, furniture.height_px
        corners = [(-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)]
        poly = Polygon(corners)
        poly = rotate(poly, angle, origin='center', use_radians=False)
        poly = translate(poly, xoff=center_pos[0], yoff=center_pos[1])
        return poly
