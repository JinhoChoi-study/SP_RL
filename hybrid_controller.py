# hybrid_controller_purepursuit.py

import math
import numpy as np
import carla
from agents.tools.misc import get_speed

class HybridVehicleController:
    def __init__(self, vehicle, wheel_base=2.8, max_steering=0.8):
        self._vehicle = vehicle
        self._world = vehicle.get_world()
        self.max_steer = max_steering
        self.past_steering = 0.0
        self._lat_controller = PurePursuitController(self._world, wheel_base=wheel_base)

    def set_waypoints(self, waypoints):
        self._lat_controller.set_waypoints(waypoints)

    def run_step(self, raw_throttle):
        control, _ = self._lat_controller.run_step(self._vehicle)
        current_steering = control.steer

        control = carla.VehicleControl()
        control.throttle = np.clip(raw_throttle, 0.0, 1.0)
        control.brake = 0.0

        if current_steering > self.past_steering + 0.1:
            current_steering = self.past_steering + 0.1
        elif current_steering < self.past_steering - 0.1:
            current_steering = self.past_steering - 0.1

        steering = np.clip(current_steering, -self.max_steer, self.max_steer)
        control.steer = steering
        control.hand_brake = False
        control.manual_gear_shift = False
        self.past_steering = steering

        return control


class PurePursuitController:
    def __init__(self, world, wheel_base=2.8):
        self.world = world
        self.wheel_base = wheel_base
        self.waypoint_list = []

    def set_waypoints(self, waypoints):
        self.waypoint_list = waypoints

    def get_dynamic_lookahead(self, speed):
        min_speed = 0.0
        max_speed = 13.0 
        min_lookahead = 8.0  
        max_lookahead = 8.0 

        factor = (speed - min_speed) / (max_speed - min_speed)
        factor = np.clip(factor, 0.0, 1.0)
        return min_lookahead + factor * (max_lookahead - min_lookahead)

    def find_lookahead_waypoint(self, vehicle, waypoints, lookahead_distance):
        vehicle_location = vehicle.get_location()
        vehicle_x, vehicle_y = vehicle_location.x, vehicle_location.y
        vehicle_yaw = np.radians(vehicle.get_transform().rotation.yaw)

        candidate_waypoints = []
        farthest_waypoint = None
        max_distance = 0.0

        for waypoint in waypoints:
            wp_x, wp_y = waypoint
            distance = max(np.linalg.norm([wp_x - vehicle_x, wp_y - vehicle_y]), 0.01)
            if distance > max_distance:
                max_distance = distance
                farthest_waypoint = waypoint
            rel_x, rel_y = wp_x - vehicle_x, wp_y - vehicle_y
            angle_to_wp = np.arctan2(rel_y, rel_x) - vehicle_yaw
            if abs(angle_to_wp) < np.pi / 2:
                if lookahead_distance * 0.8 <= distance <= lookahead_distance:
                    candidate_waypoints.append((distance, waypoint))

        if candidate_waypoints:
            return max(candidate_waypoints, key=lambda x: x[0])[1]
        return farthest_waypoint

    def run_step(self, vehicle, path=None):
        waypoints = path if path is not None else self.waypoint_list
        if not waypoints:
            return carla.VehicleControl(steer=0.0), []

        velocity = vehicle.get_velocity()
        speed = np.linalg.norm([velocity.x, velocity.y])
        lookahead_distance = self.get_dynamic_lookahead(speed)
        lookahead_wp = self.find_lookahead_waypoint(vehicle, waypoints, lookahead_distance)

        if lookahead_wp:
            vehicle_transform = vehicle.get_transform()
            vehicle_yaw = np.radians(vehicle_transform.rotation.yaw)

            cx = vehicle_transform.location.x
            cy = vehicle_transform.location.y

            rear_axle_offset = 2.5
            rx = cx - rear_axle_offset * np.cos(vehicle_yaw)
            ry = cy - rear_axle_offset * np.sin(vehicle_yaw)

            target_x, target_y = lookahead_wp
            dx = target_x - rx
            dy = target_y - ry

            cos_yaw = np.cos(vehicle_yaw)
            sin_yaw = np.sin(vehicle_yaw)
            local_x = dx * cos_yaw + dy * sin_yaw
            local_y = -dx * sin_yaw + dy * cos_yaw
            alpha = math.atan2(local_y, local_x)

            steering_angle = math.atan2(2 * self.wheel_base * math.sin(alpha), lookahead_distance)
            max_steering_rad = np.radians(35.0)  
            steer_ratio = steering_angle / max_steering_rad
            steer_ratio = np.clip(steer_ratio, -1.0, 1.0)

            control = carla.VehicleControl(steer=steer_ratio)
            return control, lookahead_wp

        return carla.VehicleControl(steer=0.0), []
