import carla
import numpy as np

class RoutePredictor:
    def __init__(self, world, map, prediction_horizon=3, time_step=0.3, num_predictions=6, brake_acceleration=5.0):
        self.world = world
        self.map = map
        self.prediction_horizon = prediction_horizon
        self.time_step = time_step
        self.num_predictions = num_predictions
        self.brake_acceleration = brake_acceleration

    def get_fixed_nearest_vehicles(self, ego_vehicle):
        vehicles = self.world.get_actors().filter('vehicle.*')
        ego_location = ego_vehicle.get_location()
        ego_yaw = np.radians(ego_vehicle.get_transform().rotation.yaw)

        vehicle_candidates = []

        for vehicle in vehicles:
            if vehicle.id == ego_vehicle.id:
                continue

            v_loc = vehicle.get_location()
            dx = v_loc.x - ego_location.x
            dy = v_loc.y - ego_location.y
            distance_2d = np.linalg.norm([dx, dy])

            vehicle_candidates.append((distance_2d, vehicle, v_loc))

        vehicle_candidates.sort(key=lambda x: x[0])
        nearest_vehicles = []

        for _, vehicle, v_loc in vehicle_candidates:
            if len(nearest_vehicles) >= self.num_predictions:
                break

            if not vehicle.is_alive:
                continue

            if v_loc.z < -5 or v_loc.z > 10:
                self.world.debug.draw_string(
                    v_loc + carla.Location(z=2.0),
                    text=f"Z-filtered ID:{vehicle.id}",
                    life_time=0.3,
                    color=carla.Color(r=255, g=100, b=0)
                )
                continue

            extent = vehicle.bounding_box.extent
            if extent.x < 0.5 or extent.y < 0.5:
                continue

            dx = v_loc.x - ego_location.x
            dy = v_loc.y - ego_location.y
            local_x = np.cos(-ego_yaw) * dx - np.sin(-ego_yaw) * dy
            local_y = np.cos(-ego_yaw) * dy + np.sin(-ego_yaw) * dx
            angle = np.degrees(np.arctan2(local_y, local_x))

            if -160 <= angle <= 160:
                nearest_vehicles.append(vehicle)

        return nearest_vehicles

    def get_trajectories_and_next_waypoints(self, sorted_vehicles):
        predicted_trajectories = {}
        next_waypoints = {}

        for vehicle in sorted_vehicles:
            vehicle_id = vehicle.id
            future_positions = self.get_future_positions(vehicle)
            future_waypoints = self.get_future_waypoints(vehicle)

            mixed_trajectory = future_positions[:5] + future_waypoints[5:]
            predicted_trajectories[vehicle_id] = mixed_trajectory
            next_waypoints[vehicle_id] = mixed_trajectory[5] if len(mixed_trajectory) > 5 else mixed_trajectory[-1]

        return predicted_trajectories, next_waypoints


    def get_future_positions(self, vehicle):
        transform = vehicle.get_transform()
        velocity = vehicle.get_velocity()
        yaw = np.radians(transform.rotation.yaw)

        speed = np.linalg.norm([velocity.x, velocity.y])  
        a = self.brake_acceleration

        s_total = ((2*speed))
        step_distance = 1.0
        num_points = max(int(s_total // step_distance)+5, 10)

        back_offset = 2.0
        current_x = transform.location.x - back_offset * np.cos(yaw)
        current_y = transform.location.y - back_offset * np.sin(yaw)

        future_positions = []
        for i in range(num_points):
            if i > 0:
                current_x += step_distance * np.cos(yaw)
                current_y += step_distance * np.sin(yaw)
            future_positions.append((current_x, current_y))

        return future_positions

    def get_future_waypoints(self, vehicle):
        transform = vehicle.get_transform()
        velocity = vehicle.get_velocity()
        speed = np.linalg.norm([velocity.x, velocity.y])  
        a = self.brake_acceleration

        s_total = ((2*speed))
        step_distance = 1.0
        num_points = max(int(s_total // step_distance)+5, 10)
        
        back_offset = 2.0
        yaw = np.radians(transform.rotation.yaw)
        offset_location = carla.Location(
            x=transform.location.x - back_offset * np.cos(yaw),
            y=transform.location.y - back_offset * np.sin(yaw),
            z=transform.location.z
        )

        current_wp = self.map.get_waypoint(offset_location, project_to_road=True, lane_type=carla.LaneType.Driving)
        future_positions = []

        for _ in range(num_points):
            if current_wp is None:
                break
            loc = current_wp.transform.location
            future_positions.append((loc.x, loc.y))
            next_wps = current_wp.next(step_distance)
            if next_wps:
                current_wp = next_wps[0]
            else:
                break

        return future_positions
