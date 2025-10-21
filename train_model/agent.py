# agent.py
import numpy as np
import random
import tensorflow as tf
from model import create_dueling_dqn_model
from replay_buffer import PrioritizedReplayBuffer

class DDDQNPERAgent:
    def __init__(self, state_shape, num_actions, board_size=15, gamma=0.99, learning_rate=1e-4, buffer_size=100000, batch_size=64):
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.board_size = board_size
        self.gamma = gamma
        self.batch_size = batch_size

        # 온라인 네트워크와 타겟 네트워크 생성
        self.online_network = create_dueling_dqn_model(state_shape, num_actions)
        self.target_network = create_dueling_dqn_model(state_shape, num_actions)
        self.update_target_network()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.replay_buffer = PrioritizedReplayBuffer(buffer_size)

    def action_to_coords(self, action_index):
        return action_index // self.board_size, action_index % self.board_size

    def coords_to_action(self, coords):
        # coords는 (row, col) 튜플 또는 리스트 형태여야 합니다.
        # 정확히 row * board_size + col 형태의 단일 인덱스를 반환하도록 수정합니다.
        row, col = coords
        return row * self.board_size + col

    def act(self, state, epsilon, legal_actions):
        """Epsilon-greedy 정책에 따라 행동을 선택합니다."""
        if np.random.rand() <= epsilon:
            return random.choice(legal_actions)
        else:
            q_values = self.online_network.predict(np.expand_dims(state, axis=0), verbose=0) # Add verbose=0 to suppress output

            # 합법적인 수 중에서만 Q-값이 가장 높은 수를 선택
            legal_action_indices = [self.coords_to_action(coords) for coords in legal_actions]

            # 불법적인 수의 Q-값을 매우 낮게 설정
            # mask의 두 번째 차원에 인덱싱하여 Q-값을 조정합니다.
            mask = np.ones_like(q_values) * -np.inf
            
            # Corrected line: apply mask to the second dimension (actions)
            mask[0, legal_action_indices] = 0 
            
            q_values += mask

            action_index = np.argmax(q_values)
            return self.action_to_coords(action_index)

    def replay(self):
        """리플레이 버퍼에서 샘플링하여 네트워크를 학습시킵니다."""
        if self.replay_buffer.tree.n_entries < self.batch_size:
            return 0.0 # 학습할 데이터가 충분하지 않음

        # PER에서 미니배치 샘플링
        mini_batch, tree_indices, is_weights = self.replay_buffer.sample(self.batch_size)

        # 수정된 부분: mini_batch에서 각 요소를 개별적으로 추출하여 NumPy 배열로 변환
        # zip(*mini_batch)를 사용하여 튜플 리스트를 튜플 묶음으로 언팩합니다.
        states_list, actions_list, rewards_list, next_states_list, dones_list = zip(*mini_batch)
        
        states = np.array(states_list)
        # action은 (row, col) 튜플이므로, coords_to_action을 적용하여 1차원 인덱스로 변환합니다.
        actions = np.array([self.coords_to_action(action_coords) for action_coords in actions_list])
        rewards = np.array(rewards_list, dtype=np.float32) # 명시적 타입 지정
        next_states = np.array(next_states_list)
        dones = np.array(dones_list, dtype=np.float32)

        # Double DQN 업데이트 로직
        # 1. 온라인 네트워크로 다음 상태에서 최적의 행동 선택
        next_q_values_online = self.online_network.predict(next_states)
        best_next_actions = np.argmax(next_q_values_online, axis=1)

        # 2. 타겟 네트워크로 그 행동의 가치 평가
        next_q_values_target = self.target_network.predict(next_states)
        target_q_for_best_actions = next_q_values_target[np.arange(self.batch_size), best_next_actions]

        # TD 타겟 계산
        target_q = rewards + self.gamma * target_q_for_best_actions * (1 - dones)
        
        with tf.GradientTape() as tape:
            # 현재 상태의 Q-값 예측
            q_values = self.online_network(states)
            action_indices = tf.stack([tf.range(self.batch_size, dtype=tf.int32), tf.cast(actions, dtype=tf.int32)], axis=1)
            predicted_q = tf.gather_nd(q_values, action_indices)
            
            # 손실 계산 (IS 가중치 적용)
            td_errors = target_q - predicted_q
            loss = tf.reduce_mean(is_weights * tf.square(td_errors))

        # 그래디언트 계산 및 가중치 업데이트
        gradients = tape.gradient(loss, self.online_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.online_network.trainable_variables))
        
        # PER의 우선순위 업데이트
        self.replay_buffer.update_priorities(tree_indices, td_errors.numpy())
        
        return loss.numpy()

    def update_target_network(self, tau=1.0):
        """타겟 네트워크 가중치를 온라인 네트워크 가중치로 업데이트합니다."""
        if tau == 1.0: # Hard update
            self.target_network.set_weights(self.online_network.get_weights())
        else: # Soft update
            online_weights = self.online_network.get_weights()
            target_weights = self.target_network.get_weights()
            new_weights = [tau * o + (1 - tau) * t for o, t in zip(online_weights, target_weights)]
            self.target_network.set_weights(new_weights)

    def save_model(self, path):
        self.online_network.save_weights(path)

    def load_model(self, path):
        self.online_network.load_weights(path)