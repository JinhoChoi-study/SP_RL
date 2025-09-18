import time
import traceback
import carla
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from collections import deque
import random
from hybrid_planner import HybridPlanner
from routepredictor import RoutePredictor
from safetypotential import SafetyPotential
from hybrid_controller import HybridVehicleController
import setup
import re
import glob
from train_utils import get_state_vector  
from collections import deque
from sac import Actor, Critic, Alpha, ReplayBuffer, optimize_sac
from safetypotential import draw_debug_boxes
# ========== 설정값 ==========
num_step = 700
num_episode = 2000
gamma = 0.99
batch_size = 256
replay_buffer_size = 200000
safety_threshold = 3
action_safety_threshold = 7
min_buffer_size = 20000
update_frequency = 5
update_iterations = 1
route_combinations = [{"start_point":97, "goal_point": 40, "terminated_point": [238, 39, 32]}]
model_path = "sac_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = 49
action_dim = 1

# 네트워크 초기화
actor = Actor(state_dim=input_dim, action_dim=1).to(device)
critic1 = Critic(input_dim, 1).to(device)
critic2 = Critic(input_dim, 1).to(device)
target_critic1 = Critic(input_dim, 1).to(device)
target_critic2 = Critic(input_dim, 1).to(device)
target_critic1.load_state_dict(critic1.state_dict())
target_critic2.load_state_dict(critic2.state_dict())

alpha_module = Alpha().to(device)

# 옵티마이저
actor_optimizer = torch.optim.Adam(actor.parameters(), lr=2e-4)
critic1_optimizer = torch.optim.Adam(critic1.parameters(), lr=2e-4)
critic2_optimizer = torch.optim.Adam(critic2.parameters(), lr=2e-4)
alpha_optimizer = torch.optim.Adam([alpha_module.log_alpha], lr=1e-4)

# 버퍼
replay_buffer = ReplayBuffer(replay_buffer_size)

def load_latest_model():
    latest_model = "sac_model_latest.pth"
    if os.path.exists(latest_model):
        try:
            checkpoint = torch.load(latest_model)
            actor.load_state_dict(checkpoint['actor_state_dict'])
            critic1.load_state_dict(checkpoint['critic1_state_dict'])
            critic2.load_state_dict(checkpoint['critic2_state_dict'])
            target_critic1.load_state_dict(checkpoint['target_critic1_state_dict'])
            target_critic2.load_state_dict(checkpoint['target_critic2_state_dict'])
            alpha_module.load_state_dict(checkpoint['alpha_state_dict'])

            if 'replay_buffer' in checkpoint:
                replay_buffer.buffer = checkpoint['replay_buffer']
                print(f"✅ Replay buffer 복원됨 (크기: {len(replay_buffer)})")

            episode_num = checkpoint.get('episode', 0)
            print(f"✅ Loaded latest model from {latest_model}, starting from episode {episode_num}")
            return episode_num
        except Exception as e:
            print(f"⚠️ 최신 모델 로딩 실패: {e}")

    model_files = glob.glob("sac_model_episode_*.pth")

    episode_file_pairs = []
    for f in model_files:
        match = re.search(r"sac_model_episode_(\d+)\.pth", f)
        if match:
            episode = int(match.group(1))
            episode_file_pairs.append((episode, f))

    if not episode_file_pairs:
        print("🟡 모델 파일이 존재하지 않습니다. 에피소드 0부터 시작합니다.")
        return 0

    episode_file_pairs.sort(reverse=True)
    best_episode, best_file = episode_file_pairs[0]

    try:
        checkpoint = torch.load(best_file)

        actor.load_state_dict(checkpoint['actor_state_dict'])
        critic1.load_state_dict(checkpoint['critic1_state_dict'])
        critic2.load_state_dict(checkpoint['critic2_state_dict'])
        target_critic1.load_state_dict(checkpoint['target_critic1_state_dict'])
        target_critic2.load_state_dict(checkpoint['target_critic2_state_dict'])
        alpha_module.load_state_dict(checkpoint['alpha_state_dict'])

        if 'replay_buffer' in checkpoint:
            replay_buffer.buffer = checkpoint['replay_buffer']
            print(f"✅ Replay buffer 복원됨 (크기: {len(replay_buffer)})")

        episode_num = checkpoint.get('episode', best_episode)
        print(f"✅ fallback: {best_file} 로드됨, 에피소드 {episode_num}부터 시작합니다.")
        return episode_num
    except Exception as e:
        print(f"❌ {best_file} 모델 로딩 중 오류 발생: {e}")
        return 0


def run_experiment():
    client, world, blueprint_library = setup.setup_carla()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.1  
    settings.max_substeps = 10         
    settings.max_substep_delta_time = 0.01  
    world.apply_settings(settings)
    world.tick()
    setup.set_birdseye_view(world)
    vehicles = []
    route_predictor = RoutePredictor(world, world.get_map())
    global epsilon
    start_episode = load_latest_model()
    for episode in range(start_episode, num_episode):
        num_vehicles = random.randint(180, 190)
        middle_points = [131, 42]  
        middle_flags = [False for _ in middle_points]           
        route_idx = episode % len(route_combinations)
        selected_route = route_combinations[route_idx]
        start_point = selected_route["start_point"]
        goal_point = selected_route["goal_point"]
        terminated_point = selected_route["terminated_point"]
        setup.collision_detected = False

        try:
            ego_vehicle, collision_sensor = setup.spawn_ego_vehicle(world, blueprint_library, start_point)
            if ego_vehicle is None:
                time.sleep(1)
                continue

            vehicles = setup.spawn_traffic(world, blueprint_library, num_vehicles=num_vehicles)
            if not vehicles:
                continue

            hybrid_controller = HybridVehicleController(
                ego_vehicle,
                wheel_base=2.875,
                max_steering=0.8)
            
            hybrid_planner = HybridPlanner(world, route_path="global_route_97_to_40.npy")
            safety_potential = SafetyPotential()
            spawn_points = world.get_map().get_spawn_points()
            goal_location = spawn_points[goal_point].location
            start_location = spawn_points[start_point].location

        except Exception as e:
            print(f"🚨 초기 설정 중 오류 발생: {str(e)}")
            traceback.print_exc()
            if vehicles:
                client.apply_batch([carla.command.DestroyActor(x) for x in vehicles])
            if ego_vehicle:
                ego_vehicle.destroy()
            if collision_sensor:
                collision_sensor.destroy()
            continue

        total_reward = 0.0
        sp_penalty_total = 0.0
        speed_reward_total = 0.0
        middle_reward_total = 0.0
        acc_penalty_total = 0.0
        low_speed_penalty_total = 0.0
        done = False  

        try:
            for step in range(num_step):
                if step < 10:
                    control = carla.VehicleControl()
                    control.brake = 1.0
                    control.throttle = 0.0
                    ego_vehicle.apply_control(control)
                    world.tick()
                    continue

               #======================== ready for action, state vector=============================================
                nearest_vehicles = route_predictor.get_fixed_nearest_vehicles(ego_vehicle)
                surrounding_trajectories, _ = route_predictor.get_trajectories_and_next_waypoints(nearest_vehicles)
                ego_path = hybrid_planner.generate_target_path(ego_vehicle)

                if not ego_path:
                    continue 

                if len(ego_path) > 4:
                    target_point = ego_path[4]
                else:
                    target_point = ego_path[-1]

                pp_control, _ = hybrid_controller._lat_controller.run_step(ego_vehicle, path=ego_path)        
                state_vector = get_state_vector(
                    ego_vehicle, target_point, nearest_vehicles, goal_location, start_location, world.get_map())

                control = carla.VehicleControl()

                #================================= action =====================================================

                state_tensor = torch.from_numpy(state_vector).float().unsqueeze(0).to(device)
                with torch.no_grad():
                    action_tensor, _ = actor.sample(state_tensor)
                action = action_tensor.cpu().squeeze().item()

                steering = pp_control.steer

                if action > 0:
                    throttle = action
                    brake = 0.0
                else:
                    throttle = 0.0
                    brake = -action

                control = carla.VehicleControl(throttle=throttle, brake=brake, steer=steering)
                ego_vehicle.apply_control(control)

                world.tick()

                #================= reward ================================
                new_ego_path = hybrid_planner.generate_target_path(ego_vehicle)
                if not new_ego_path:
                    continue 

                new_surrounding_trajectories, _ = route_predictor.get_trajectories_and_next_waypoints(nearest_vehicles)
    
                safety_score,_ ,_  = safety_potential.compute_potential(new_ego_path, new_surrounding_trajectories)     
                sp_penalty = -0.005* safety_score
                sp_penalty_total += sp_penalty    

                velocity = ego_vehicle.get_velocity()
                speed = np.linalg.norm([velocity.x, velocity.y])
                target_speed = 8.0
                speed_diff = speed - target_speed

                if speed <= (target_speed + 2) and safety_score < safety_threshold:
                    speed_reward = max(0.0, 1.0 - abs(speed_diff) / target_speed) * 0.02  
                else:
                    if speed > (target_speed + 2):  
                        excess_speed = speed - target_speed
                        speed_reward = -0.2 * (excess_speed / target_speed) 
                    else:
                        speed_reward = 0.0  

                reward = sp_penalty + speed_reward
                speed_reward_total += speed_reward  

                if not done and speed < 0.5 and action < 0.1 and safety_score < 0.01:
                    low_speed_penalty = -0.015
                    reward += low_speed_penalty
                    low_speed_penalty_total += low_speed_penalty

                if not done and safety_score > action_safety_threshold :
                    acc_penalty = -0.005 * safety_score * action
                    reward += acc_penalty
                    acc_penalty_total += acc_penalty

                if not done and setup.check_collision(nearest_vehicles):
                    reward += -10.0  
                    done = True
                    collision = True
                    print(f"💥 충돌 발생 → -10점 및 종료")

                elif not done and setup.check_goal_reached(ego_vehicle, goal_location):
                    reward += 5.0
                    done = True
                    goal_reached = True
                    print(f"🏁 목표 도달 → +5점 및 종료")
                
                reached, middle_flags = setup.check_middle_target_reached(
                    ego_vehicle, world, middle_points, middle_flags)
                if reached:
                    reward += 1.0
                    middle_reward_total += 1.0
                    print(f"✅ 중간 지점 도달 → +1점")

                # ✅ 리워드 누적
                total_reward += reward

        #======================== s', replay buffer push, upadate ===================================

                if len(new_ego_path) > 4:
                    new_target_point = new_ego_path[4] 
                else:
                    new_target_point = new_ego_path[-1]          

                new_state = get_state_vector(
                    ego_vehicle, new_target_point, nearest_vehicles, goal_location, start_location, world.get_map())

                replay_buffer.push(state_vector, [action], reward, new_state, done)

                if step % update_frequency == 0:
                    _, _, _, _, _, _ = optimize_sac(
                        actor, critic1, critic2, target_critic1, target_critic2,
                        actor_optimizer, critic1_optimizer, critic2_optimizer,
                        alpha_module, alpha_optimizer,
                        replay_buffer, min_buffer_size,
                        gamma=gamma, tau=0.001, target_entropy=-1.0, device=device
                    )

                if done:
                    break

        except Exception as e:
            print(f"❌ 스텝 실행 중 오류: {str(e)}")
            traceback.print_exc()

        if ego_vehicle:
            ego_vehicle.destroy()
        if collision_sensor:
            collision_sensor.destroy()
        if vehicles:
            client.apply_batch([carla.command.DestroyActor(x) for x in vehicles])

        checkpoint_data = {
            'episode': episode + 1,  
            'actor_state_dict': actor.state_dict(),
            'critic1_state_dict': critic1.state_dict(),
            'critic2_state_dict': critic2.state_dict(),
            'target_critic1_state_dict': target_critic1.state_dict(),
            'target_critic2_state_dict': target_critic2.state_dict(),
            'alpha_state_dict': alpha_module.state_dict(),
            'replay_buffer': replay_buffer.buffer
        }

        if (episode + 1) % 50 == 0:
            torch.save(checkpoint_data, f"sac_model_episode_{episode+1}.pth")

        torch.save(checkpoint_data, "sac_model_latest.pth")

    settings = world.get_settings()
    settings.synchronous_mode = False
    world.apply_settings(settings)
    print("✅ 모든 에피소드 완료")

if __name__ == "__main__":
    run_experiment()

