import carla
import numpy as np

class HybridPlanner:
    def __init__(self, world, route_path="global_route_58_to_40.npy", step_distance=1.0, min_points=8, a=5.0):
        self.world = world
        self.map = world.get_map()
        self.step_distance = step_distance
        self.min_points = min_points
        self.a = a 
        self.global_route = np.load(route_path).tolist()

    def generate_target_path(self, ego_vehicle):

        transform = ego_vehicle.get_transform()
        velocity = ego_vehicle.get_velocity()
        speed = np.linalg.norm([velocity.x, velocity.y])
        yaw = np.radians(transform.rotation.yaw)

        s_total = max(2*speed, 8.0)
        num_points = max(int(s_total // self.step_distance), self.min_points)

        back_offset = -3.0
        x = transform.location.x - back_offset * np.cos(yaw)
        y = transform.location.y - back_offset * np.sin(yaw)

        # 글로벌 경로에서 가장 가까운 지점부터 샘플링
        dists = [np.hypot(x - gx, y - gy) for gx, gy in self.global_route]
        closest_idx = int(np.argmin(dists))

        sampling_path = []
        prev_x, prev_y = self.global_route[closest_idx]
        sampling_path.append((prev_x, prev_y))

        for i in range(closest_idx + 1, len(self.global_route)):
            gx, gy = self.global_route[i]
            dist = np.hypot(gx - prev_x, gy - prev_y)

            if dist >= self.step_distance * 0.99:
                sampling_path.append((gx, gy))
                prev_x, prev_y = gx, gy

            if len(sampling_path) >= num_points:
                break

        return sampling_path[:num_points]