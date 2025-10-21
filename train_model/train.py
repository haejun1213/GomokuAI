# train.py
import numpy as np
import random
from collections import deque
from gomoku_env import GomokuEnv
from agent import DDDQNPERAgent
import tensorflow as tf

# TensorBoard 로깅 설정
log_dir = "logs/fit/"
summary_writer = tf.summary.create_file_writer(log_dir)

# 하이퍼파라미터
BOARD_SIZE = 9  # 초기 학습을 위해 작은 보드에서 시작 (9x9)
NUM_EPISODES = 5000
MAX_STEPS_PER_EPISODE = BOARD_SIZE * BOARD_SIZE
REPLAY_START_SIZE = 10000
TARGET_UPDATE_FREQ = 1000  # steps
SOFT_UPDATE_TAU = 1e-3
BATCH_SIZE = 64

# Epsilon-greedy 파라미터
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY_STEPS = 200000

def main():
    env = GomokuEnv(board_size=BOARD_SIZE)
    state_shape = (BOARD_SIZE, BOARD_SIZE, 2)
    num_actions = BOARD_SIZE * BOARD_SIZE
    
    agent = DDDQNPERAgent(
        state_shape=state_shape,
        num_actions=num_actions,
        board_size=BOARD_SIZE,
        batch_size=BATCH_SIZE
    )

    total_steps = 0
    epsilon = EPSILON_START

    for episode in range(NUM_EPISODES):
        state = env.reset()
        episode_reward = 0
        
        # 에이전트가 스스로와 대결하도록 설정
        players = {1: agent, -1: agent}
        
        for step in range(MAX_STEPS_PER_EPISODE):
            current_player_id = env.current_player
            current_player_agent = players[current_player_id]
            
            # Epsilon-greedy 행동 선택
            legal_actions = env.get_legal_actions()
            if not legal_actions: break # 둘 곳이 없으면 종료
            
            action = current_player_agent.act(state, epsilon, legal_actions)
            
            next_state, reward, done, _ = env.step(action)
            
            # 리플레이 버퍼에 저장
            # 보상이 현재 플레이어 기준이므로, 이전 플레이어의 경험으로 저장
            # (이전 상태, 행동, 보상, 현재 상태, 종료 여부)
            agent.replay_buffer.add(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            total_steps += 1
            
            # Epsilon 값 감소
            epsilon = max(EPSILON_END, EPSILON_START - (EPSILON_START - EPSILON_END) * (total_steps / EPSILON_DECAY_STEPS))

            # 학습 (리플레이 버퍼가 충분히 쌓였을 때)
            if total_steps > REPLAY_START_SIZE:
                loss = agent.replay()
                if total_steps % 100 == 0:
                    with summary_writer.as_default():
                        tf.summary.scalar('loss', loss, step=total_steps)

                # 타겟 네트워크 업데이트 (Soft update)
                if total_steps % TARGET_UPDATE_FREQ == 0:
                     agent.update_target_network(tau=SOFT_UPDATE_TAU)

            if done:
                break
        
        print(f"Episode: {episode}, Total Steps: {total_steps}, Reward: {episode_reward}, Epsilon: {epsilon:.4f}")
        with summary_writer.as_default():
            tf.summary.scalar('episode_reward', episode_reward, step=episode)
            tf.summary.scalar('epsilon', epsilon, step=episode)

        # 모델 저장
        if episode % 100 == 0:
            agent.save_model(f"models/gomoku_agent_episode_{episode}.h5")

if __name__ == "__main__":
    main()