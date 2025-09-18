import glob
import os
import sys
import time
import random
import carla
import numpy as np

def setup_carla():
    """ CARLA 시뮬레이터 환경 설정 및 연결 """
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()
    return client, world, blueprint_library

collision_detected = False
collision_actor_id = None
_detected_ts = 0.0          
_ignore_logged = False       

def handle_collision(event):
    global collision_detected, collision_actor_id, _detected_ts, _ignore_logged
    other_actor = event.other_actor
    other_type = other_actor.type_id

    ignore_keywords = ["static.", "road", "terrain"]
    if any(keyword in other_type for keyword in ignore_keywords):
        print(f"⚪ 무시된 충돌(정적/지형): {other_type} (ID={other_actor.id})")
        return

    print(f"🚨 차량 충돌 감지(후보)! Actor={other_type}, ID={other_actor.id}")
    collision_detected = True
    collision_actor_id = other_actor.id
    _detected_ts = time.time()  
    _ignore_logged = False       

def check_collision(nearest_vehicles, timeout_sec=1.0):

    global collision_detected, collision_actor_id, _ignore_logged

    if not collision_detected:
        return False

    if time.time() - _detected_ts > timeout_sec:
        print(f"⏱ 후보 만료: ID={collision_actor_id} (timeout={timeout_sec}s)")
        collision_detected = False
        collision_actor_id = None
        _ignore_logged = False
        return False

    nearest_ids = [v.id for v in nearest_vehicles]
    if collision_actor_id in nearest_ids:
        # 소비 후 리셋
        collision_detected = False
        collision_actor_id = None
        _ignore_logged = False
        return True
    else:
        # 같은 내용의 무시 로그 반복 방지
        if not _ignore_logged:           
            _ignore_logged = True
        return False

def reset_collision_flags():
    global collision_detected, collision_actor_id, _detected_ts, _ignore_logged
    collision_detected = False
    collision_actor_id = None
    _detected_ts = 0.0
    _ignore_logged = False

def check_goal_reached(ego_vehicle, goal_location, threshold=8.0):

    ego_location = ego_vehicle.get_location()
    distance = np.sqrt((ego_location.x - goal_location.x) ** 2 +
                       (ego_location.y - goal_location.y) ** 2)
    
    if distance < threshold:
        print(f"🏁 목표 도달! 현재 위치: {ego_location}, 목표 위치: {goal_location}")
        return True  
    return False  

# ========== 특정 지점 근처 확인 함수 ==========
def check_near_specific_points(ego_vehicle, world, termianted_point, threshold=3.0):

    ego_location = ego_vehicle.get_location()
    spawn_points = world.get_map().get_spawn_points()

    for point_idx in termianted_point:
        if point_idx >= len(spawn_points):
            continue  # 인덱스가 스폰 포인트 범위를 벗어나면 스킵
        target_location = spawn_points[point_idx].location
        distance = np.sqrt((ego_location.x - target_location.x) ** 2 +
                           (ego_location.y - target_location.y) ** 2)
        if distance < threshold:
            print(f"📍 {point_idx}번 지점 근처 도달! 거리: {distance:.2f}m")
            return True
    return False

def draw_global_route(world, global_route):
    """ 글로벌 플래너 경로를 초록색으로 시각화 """
    for i in range(len(global_route)):
        world.debug.draw_point(
            carla.Location(global_route[i][0], global_route[i][1], 1),
            size=0.1,
            life_time=0.5,  
            color=carla.Color(0, 255, 0)  # 초록색 (글로벌 경로)
        )

def draw_candidate_paths(world, candidate_paths):
    colors = [
        carla.Color(255, 165, 0),   # 주황 (기본 경로 - 따뜻하고 눈에 잘 띔)
        carla.Color(0, 255, 255),   # 청록 (좌측 차선 변경 경로 - 시원한 느낌)
        carla.Color(0, 128, 0),     # 진초록 (우측 차선 변경 경로 - 안정적이고 분명함)
    ]

    for idx, path in enumerate(candidate_paths):
        color = colors[idx % len(colors)]  # 후보 경로별 색상 지정

        for i in range(len(path)):
            point = path[i]
            if isinstance(point, tuple):  
                x, y = point
                location = carla.Location(x, y, 1.0)
            elif isinstance(point, carla.Location):  
                location = point
            else:
                continue

            world.debug.draw_point(
                location,
                size=0.1,
                life_time=0.15,
                color=color
            )

def draw_best_path(world, path):
    for waypoint in path:
        if isinstance(waypoint, carla.Location):
            location = waypoint
        else:
            location = carla.Location(waypoint[0], waypoint[1], 1)

        world.debug.draw_point(
            location,
            size=0.1,
            life_time=0.15,
            color=carla.Color(255, 0, 0)
        )

def draw_surrounding_paths(world, surrounding_paths):
    color = carla.Color(128, 0, 128)
    for vehicle_id, path in surrounding_paths.items():
        if not isinstance(path, list) or len(path) < 1:
            continue  
        for i in range(len(path)):
            world.debug.draw_point(
                carla.Location(path[i][0], path[i][1], 1),
                size=0.1,
                life_time=0.15,  
                color=color  
            )

def check_fixed_angle_front_vehicle(vehicle, world, max_distance=15.0, angle_width=10):
    vehicle_location = vehicle.get_location()
    vehicle_yaw_deg = vehicle.get_transform().rotation.yaw

    closest_distance = max_distance
    found = False
    found_id = None
    found_location = None

    vehicles = world.get_actors().filter('vehicle.*')

    for other_vehicle in vehicles:
        if other_vehicle.id == vehicle.id:
            continue

        if not other_vehicle.is_alive:
            continue

        try:
            other_location = other_vehicle.get_location()
        except RuntimeError:
            continue

        if other_location.z < -5 or other_location.z > 10:
            world.debug.draw_string(
                other_location + carla.Location(z=2.0),
                text=f"Z-filtered ID:{other_vehicle.id}",
                life_time=0.3,
                color=carla.Color(r=255, g=0, b=0)
            )
            continue

        extent = other_vehicle.bounding_box.extent
        if extent.x < 0.5 or extent.y < 0.5:
            world.debug.draw_string(
                other_location + carla.Location(z=2.0),
                text=f"Size-filtered ID:{other_vehicle.id}",
                life_time=0.2,
                color=carla.Color(r=100, g=100, b=255)
            )
            continue

        dx = other_location.x - vehicle_location.x
        dy = other_location.y - vehicle_location.y
        distance = np.sqrt(dx**2 + dy**2)
        if distance > max_distance:
            continue

        direction_angle = np.degrees(np.arctan2(dy, dx))
        angle_diff = (direction_angle - vehicle_yaw_deg + 360) % 360
        if angle_diff > 180:
            angle_diff -= 360

        if abs(angle_diff) <= angle_width:
            if distance < closest_distance:
                closest_distance = distance
                found_id = other_vehicle.id
                found_location = other_location
                found = True

    if found and found_id is not None:
        world.debug.draw_string(
            found_location + carla.Location(z=2.5),
            text=f"Front ID:{found_id} dist={closest_distance:.1f}m",
            life_time=1.0,
            color=carla.Color(r=0, g=0, b=255),
            persistent_lines=False
        )

    return found, closest_distance if found else None


def is_valid_obstacle(actor):
    if not isinstance(actor, carla.Vehicle):
        return False

    try:
        type_id = actor.type_id.lower()
        blocked_keywords = ["ghost", "dummy", "static", "unknown", "prop"]
        if any(keyword in type_id for keyword in blocked_keywords):
            return False

        location = actor.get_location()
        if location.z < -1.0 or location.z > 5.0: 
            return False

        # 너무 작은 장애물 제외 (모델링 오류 방지)
        if hasattr(actor, "bounding_box"):
            extent = actor.bounding_box.extent
            if extent.x < 0.5 or extent.y < 0.5:  
                return False

        # 특정 맵 위치 필터링 (회전교차로 등)
        if 65 < location.x < 72 and -1 < location.y < 6:
            return False

    except Exception as e:
        return False

    return True


def spawn_traffic(world, blueprint_library, num_vehicles, traffic_manager=None):

    if traffic_manager is None:
        traffic_manager = carla.Client('localhost', 2000).get_trafficmanager()
    
    traffic_manager.set_random_device_seed(0)
    
    vehicle_blueprints = blueprint_library.filter('vehicle.*')
    selected_blueprints = [x for x in vehicle_blueprints if x.id.endswith(
        ('model3', 'mustang', 'crown', 'prius', 'etron', 'a2', 'tt', 'grandtourer', 
         'impala', 'c3', 'charger_2020', 'mkz_2017', 'mkz_2020', 'coupe', 'coupe_2020', 
         'cooper_s', 'cooper_s_2021', 'micra', 'leon'))
    ]

    all_spawn_points = world.get_map().get_spawn_points()

    exclude_indices = {58, 143}
    spawn_points = [pt for idx, pt in enumerate(all_spawn_points) if idx not in exclude_indices]

    vehicles = []
    random.shuffle(spawn_points)


    num_vehicles = min(num_vehicles, len(spawn_points))
    distance_to_leading_vehicle = [np.random.uniform(6.0, 20.0) for _ in range(num_vehicles)]
    vehicle_speed_diff = [np.random.uniform(-10.0, 10.0) for _ in range(num_vehicles)]
    vehicle_lane_offset = [np.random.uniform(-0.1, 0.1) for _ in range(num_vehicles)]
    impatient_lane_change = [np.random.uniform(5.0, 30.0) for _ in range(num_vehicles)]

    for i in range(num_vehicles):
        vehicle_bp = random.choice(selected_blueprints)
        if vehicle_bp.has_attribute('color'):
            color = random.choice(vehicle_bp.get_attribute('color').recommended_values)
            vehicle_bp.set_attribute('color', color)
        if vehicle_bp.has_attribute('driver_id'):
            driver_id = random.choice(vehicle_bp.get_attribute('driver_id').recommended_values)
            vehicle_bp.set_attribute('driver_id', driver_id)
        vehicle_bp.set_attribute('role_name', 'autopilot')


        vehicle = world.try_spawn_actor(vehicle_bp, spawn_points[i])
        if vehicle:
            vehicle.set_autopilot(True, traffic_manager.get_port())
            traffic_manager.distance_to_leading_vehicle(vehicle, distance_to_leading_vehicle[i])
            traffic_manager.vehicle_percentage_speed_difference(vehicle, vehicle_speed_diff[i])
            traffic_manager.vehicle_lane_offset(vehicle, vehicle_lane_offset[i])

            traffic_manager.ignore_lights_percentage(vehicle, 100)
            traffic_manager.ignore_signs_percentage(vehicle, 0)
            traffic_manager.ignore_vehicles_percentage(vehicle, 20)

            vehicles.append(vehicle)

    return vehicles


def spawn_ego_vehicle(world, blueprint_library, start_point):
    global collision_detected
    collision_detected = False

    ego_vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
    spawn_point = world.get_map().get_spawn_points()[start_point]
    ego_vehicle = world.try_spawn_actor(ego_vehicle_bp, spawn_point)

    if ego_vehicle:
        collision_sensor_bp = blueprint_library.find('sensor.other.collision')
        collision_sensor = world.try_spawn_actor(collision_sensor_bp, carla.Transform(), attach_to=ego_vehicle)
        collision_sensor.listen(lambda event: handle_collision(event))
        return ego_vehicle, collision_sensor
    return None, None

def set_birdseye_view(world):
    spectator = world.get_spectator()
    location = carla.Location(x=-0.460435, y=-18.993290, z=110)
    rotation = carla.Rotation(pitch=-90, yaw=0)
    spectator.set_transform(carla.Transform(location, rotation))


def check_middle_target_reached(ego_vehicle, world, middle_points, visited_flags, threshold=3.0):

    ego_location = ego_vehicle.get_location()
    spawn_points = world.get_map().get_spawn_points()

    for idx, point_idx in enumerate(middle_points):
        if visited_flags[idx]:
            continue
        if point_idx >= len(spawn_points):
            continue

        target_location = spawn_points[point_idx].location
        distance = np.linalg.norm([
            ego_location.x - target_location.x,
            ego_location.y - target_location.y
        ])

        if distance < threshold:
            visited_flags[idx] = True
            return True, visited_flags

    return False, visited_flags

