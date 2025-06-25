import numpy as np
from shapely.geometry import Polygon, Point
from shapely.affinity import rotate, translate
import copy

class FurniturePlacer:
    """
    Places furniture in rooms based on a set of rules.
    Requires `shapely` library: pip install shapely
    """
    def __init__(self, rooms_info, furniture_prototypes, furniture_map, pixel_to_meter_ratio):
        self.rooms_info = rooms_info
        self.furniture_prototypes = furniture_prototypes
        self.furniture_map = furniture_map
        self.pixel_to_meter_ratio = pixel_to_meter_ratio
        self.placed_furniture = {}

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
        """Gets a list of furniture objects for a given room type."""
        furniture_needed = self.furniture_map.get(room_type, [])
        available_furniture = []
        for f_type in furniture_needed:
            if self.furniture_prototypes.get(f_type):
                # Use deepcopy to ensure each room gets its own furniture instances
                available_furniture.append(copy.deepcopy(self.furniture_prototypes[f_type].pop(0)))
        return available_furniture

    def _place_in_room(self, room_data):
        """Applies placement rules for a single room."""
        furniture_to_place = self._get_furniture_for_room(room_data['type'])
        
        placed_in_this_room = []
        placed_polygons = []
        room_polygon = Polygon(room_data['bounding_box'])

        # Place larger items first (a simple optimization heuristic)
        furniture_to_place.sort(key=lambda f: f.width_px * f.height_px, reverse=True)

        for f in furniture_to_place:
            # Default rule: try to place against any wall
            is_placed = self._place_against_wall(f, room_data, room_polygon, placed_polygons)
            if is_placed:
                placed_in_this_room.append(f)
        
        return placed_in_this_room

    def _place_against_wall(self, furniture, room_data, room_polygon, placed_polygons):
        """A simple strategy to place furniture against a wall."""
        walls = self._get_walls(room_data['bounding_box'])
        
        for wall in walls:
            wall_vec = wall[1] - wall[0]
            wall_len = np.linalg.norm(wall_vec)
            
            # Try both orientations
            orientations = [(furniture.width_px, furniture.height_px), (furniture.height_px, furniture.width_px)]
            if furniture.width_px == furniture.height_px:
                orientations = orientations[:1]

            for f_w, f_h in orientations:
                if f_w > wall_len:
                    continue

                wall_angle = np.rad2deg(np.arctan2(wall_vec[1], wall_vec[0]))
                pos = (wall[0] + wall[1]) / 2
                
                normal = np.array([-wall_vec[1], wall_vec[0]])
                normal = normal / np.linalg.norm(normal)

                # Check if normal points inward
                test_point = pos + normal * 5
                if not room_polygon.contains(Point(test_point)):
                    normal = -normal

                pos += normal * (f_h / 2)

                temp_f = type('obj', (object,), {'width_px': f_w, 'height_px': f_h})()
                f_poly = self._get_furniture_polygon(temp_f, pos, wall_angle)

                collision = any(f_poly.intersects(p) for p in placed_polygons)
                
                if not collision and room_polygon.contains(f_poly.centroid):
                    furniture.position_px = (int(pos[0]), int(pos[1]))
                    furniture.angle = wall_angle
                    furniture.width_px, furniture.height_px = f_w, f_h
                    placed_polygons.append(f_poly)
                    return True
        
        return False

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
