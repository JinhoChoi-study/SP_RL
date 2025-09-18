import numpy as np
from shapely.geometry import box
import carla
from shapely import affinity
import math

class SafetyPotential:
    def __init__(self, vehicle_width=2.8, step_length=1.0, lam=0.5, body_penalty=1.0):
        self.vehicle_width = vehicle_width
        self.step_length = step_length
        self.lam = lam
        self.body_penalty = body_penalty
        self.w_max = 2
        self.k = 0.1
        self.theta0 = 15  

    def rotated_box(self, cx, cy, yaw_deg, width, length):
        rect = box(-length / 2, -width / 2, length / 2, width / 2)
        rect = affinity.rotate(rect, yaw_deg, origin=(0, 0))
        rect = affinity.translate(rect, cx, cy)
        return rect

    def compute_potential(self, ego_path, surrounding_vehicles):
        ego_rects = self.path_to_weighted_polygons(ego_path, is_ego=True)
        ego_dir = self.get_direction_vector(ego_path, is_ego=True)
        vehicle_overlap_areas = []
        angles = []
        weights = []

        for vehicle_path in surrounding_vehicles.values():
            if not isinstance(vehicle_path, list) or len(vehicle_path) < 7:
                continue

            vehicle_rects, body_rects = self.path_to_weighted_polygons(vehicle_path, return_body=True, is_ego=False)
            veh_dir = self.get_direction_vector(vehicle_path, is_ego=False)

            cosine = np.dot(ego_dir, veh_dir)
            cosine = np.clip(cosine, -1.0, 1.0)
            angle_deg = np.degrees(np.arccos(cosine))
            dir_weight = self.calculate_direction_weight(angle_deg)

            total_overlap = 0.0
            for ego_poly, ego_w in ego_rects:
                for veh_poly, veh_w in vehicle_rects:
                    intersection = ego_poly.intersection(veh_poly)
                    if not intersection.is_empty:
                        overlap_area = intersection.area
                        min_w = min(ego_w, veh_w)
                        total_overlap += overlap_area * min_w

                for body_poly, _ in body_rects:
                    intersection = ego_poly.intersection(body_poly)
                    if not intersection.is_empty:
                        overlap_area = intersection.area
                        total_overlap += overlap_area * ego_w * self.body_penalty

            weighted_total = total_overlap * dir_weight
            vehicle_overlap_areas.append(weighted_total)
            angles.append(angle_deg)
            weights.append(dir_weight)

        if vehicle_overlap_areas:
            max_index = np.argmax(vehicle_overlap_areas)
            return vehicle_overlap_areas[max_index], angles[max_index], weights[max_index]
        else:
            return 0.0, 0.0, 1.0  


    def calculate_direction_weight(self, angle_deg):
        """Shifted sigmoid-based direction weighting"""
        sigma = 1 / (1 + np.exp(-self.k * (angle_deg - self.theta0)))
        return 1

    def path_to_weighted_polygons(self, path_coords, return_body=False, is_ego=False):
        rects = []
        body_rects = []
        n = len(path_coords)
        if n <= 2:
            return ([], []) if return_body else []

        for i, (x, y) in enumerate(path_coords):
            if i < n - 1:
                dx, dy = np.array(path_coords[i + 1]) - np.array(path_coords[i])
            elif i > 0:
                dx, dy = np.array(path_coords[i]) - np.array(path_coords[i - 1])
            else:
                dx, dy = 1.0, 0.0

            direction = np.array([dx, dy])
            norm = np.linalg.norm(direction)
            yaw_deg = np.degrees(np.arctan2(dy, dx)) if norm > 1e-3 else 0.0

            poly = self.rotated_box(x, y, yaw_deg, self.vehicle_width, self.step_length)

            if is_ego:
                time_ratio = i / max(n - 1, 1)
                weight = np.exp(-self.lam * time_ratio)
            else:
                if i < 5:
                    weight = 1.0
                    if return_body:
                        body_rects.append((poly, 1.0))
                else:
                    n_body = max(n - 5, 1)
                    time_ratio = (i - 5) / max(n_body - 1, 1)
                    weight = np.exp(-self.lam * time_ratio)

            rects.append((poly, weight))

        return (rects, body_rects) if return_body else rects

    def get_direction_vector(self, path, is_ego):
        try:
            if is_ego:
                p0 = np.array(path[0])
                p1 = np.array(path[2])
            else:
                p0 = np.array(path[4])
                p1 = np.array(path[6])
        except IndexError:
            return np.array([1.0, 0.0])
        vec = p1 - p0
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-6)

def draw_debug_boxes(world, path, color=(0, 255, 0), life_time=0.3):
    z = 0.3
    vehicle_width = 2.8
    step_length = 1.0
    n = len(path)

    for i, (x, y) in enumerate(path):
        if i < n - 1:
            dx, dy = np.array(path[i + 1]) - np.array(path[i])
        elif i > 0:
            dx, dy = np.array(path[i]) - np.array(path[i - 1])
        else:
            dx, dy = 1.0, 0.0

        direction = np.array([dx, dy])
        norm = np.linalg.norm(direction)
        yaw_deg = np.degrees(np.arctan2(dy, dx)) if norm > 1e-3 else 0.0

        location = carla.Location(x=x, y=y, z=z)
        extent = carla.Vector3D(step_length / 2, vehicle_width / 2, 0.5) 
        rotation = carla.Rotation(pitch=0.0, yaw=yaw_deg, roll=0.0)

        box = carla.BoundingBox(location, extent)
        box.rotation = rotation

        world.debug.draw_box(
            box,
            rotation,
            thickness=0.1,
            color=carla.Color(r=color[0], g=color[1], b=color[2]),
            life_time=life_time
        )
