import numpy as np
import carla
import os

ROUTE_FILE = "global_route_97_to_40.npy"
assert os.path.exists(ROUTE_FILE), f"경로 파일이 존재하지 않습니다: {ROUTE_FILE}"
global_route_xy = np.load(ROUTE_FILE)  

def normalize_clip(x, denom, clip=1.0):
    return np.clip(x / denom, -clip, clip)

def rotate_point(dx, dy, yaw):
    cos_yaw = np.cos(-yaw)
    sin_yaw = np.sin(-yaw)
    x_new = dx * cos_yaw - dy * sin_yaw
    y_new = dx * sin_yaw + dy * cos_yaw
    return x_new, y_new


def compute_remaining_path_distance(ego_loc, route):
    route = np.array(route)
    ego_xy = np.array([ego_loc.x, ego_loc.y])
    dists = np.linalg.norm(route - ego_xy, axis=1)
    nearest_idx = np.argmin(dists)
    seg_dists = np.linalg.norm(np.diff(route, axis=0), axis=1)
    total_dist = np.sum(seg_dists)
    remaining_dist = np.sum(seg_dists[nearest_idx:])
    return remaining_dist, total_dist

def get_state_vector(ego_vehicle, target_waypoint, nearest_vehicles, goal_location, start_location, carla_map):
    velocity = ego_vehicle.get_velocity()
    acceleration = ego_vehicle.get_acceleration()
    ego_loc = ego_vehicle.get_location()
    ego_yaw = np.radians(ego_vehicle.get_transform().rotation.yaw)

    ego_vx, ego_vy = rotate_point(velocity.x, velocity.y, ego_yaw)
    ego_ax, ego_ay = rotate_point(acceleration.x, acceleration.y, ego_yaw)
    ego_speed = np.linalg.norm([ego_vx, ego_vy])
    ego_acc = np.linalg.norm([ego_ax, ego_ay])
    ego_speed_dir = np.arctan2(ego_vy, ego_vx + 1e-6) / np.pi
    ego_acc_dir = np.arctan2(ego_ay, ego_ax + 1e-6) / np.pi

    ego_info = [
        normalize_clip(ego_speed, 20.0),
        ego_speed_dir,
        normalize_clip(ego_acc, 10.0),
        ego_acc_dir
    ]

    rem_dist, tot_dist = compute_remaining_path_distance(ego_loc, global_route_xy)
    norm_goal_dist = np.clip(1.0 - (rem_dist / (1.0 * tot_dist)), 0.0, 1.0)

    if target_waypoint is not None:
        tx, ty = target_waypoint
        dx = tx - ego_loc.x
        dy = ty - ego_loc.y
        rel_x, rel_y = rotate_point(dx, dy, ego_yaw)
        target_distance = normalize_clip(np.linalg.norm([rel_x, rel_y]), 20.0)
        target_direction = np.arctan2(rel_y, rel_x + 1e-6) / np.pi

    else:
        target_distance, target_direction= 1.0, 0.0

    target_info = [target_distance, target_direction]

    surrounding_info = []
    for vehicle in nearest_vehicles[:6]:
        v_loc = vehicle.get_location()
        v_vel = vehicle.get_velocity()
        v_acc = vehicle.get_acceleration()

        dx = v_loc.x - ego_loc.x
        dy = v_loc.y - ego_loc.y
        rel_x, rel_y = rotate_point(dx, dy, ego_yaw)
        distance = np.linalg.norm([rel_x, rel_y])
        direction = np.arctan2(rel_y, rel_x + 1e-6) / np.pi

        v_vx, v_vy = rotate_point(v_vel.x, v_vel.y, ego_yaw)
        v_ax, v_ay = rotate_point(v_acc.x, v_acc.y, ego_yaw)
        speed = np.linalg.norm([v_vx, v_vy])
        acc = np.linalg.norm([v_ax, v_ay])
        speed_dir = np.arctan2(v_vy, v_vx + 1e-6) / np.pi
        acc_dir = np.arctan2(v_ay, v_ax + 1e-6) / np.pi
        rel_yaw = vehicle.get_transform().rotation.yaw - ego_vehicle.get_transform().rotation.yaw
        rel_yaw = np.arctan2(np.sin(np.radians(rel_yaw)), np.cos(np.radians(rel_yaw))) / np.pi

        surrounding_info.extend([
            normalize_clip(distance, 20.0),
            direction,
            normalize_clip(speed, 20.0),
            speed_dir,
            rel_yaw,
            normalize_clip(acc, 10.0),
            acc_dir
        ])

    while len(surrounding_info) < 6 * 7:
        surrounding_info.extend([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    state_vector = np.array(
        ego_info + [norm_goal_dist] + target_info + surrounding_info,
        dtype=np.float32
    )

    expected_dim = 4 + 1 + 2 + 6 * 7  
    assert state_vector.shape[0] == expected_dim, f"상태 벡터 차원 불일치! 기대: {expected_dim}, 실제: {state_vector.shape[0]}"
    return state_vector