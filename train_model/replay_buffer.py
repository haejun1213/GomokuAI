# replay_buffer.py
import numpy as np
import random

class SumTree:
    """
    SumTree 자료구조. 효율적인 우선순위 샘플링을 위해 사용됩니다.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0
        self.n_entries = 0

    def add(self, priority, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_idx, priority)

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, tree_idx, priority):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx!= 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get(self, s):
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            if left_child_idx >= len(self.tree): # 변경: len(self.tree) 대신 self.capacity - 1 (리프 노드 시작 인덱스)를 기준으로 해야 하지만,
                                                  # 현재 트리의 크기를 고려하면 이 조건도 작동할 수 있음.
                                                  # 하지만 더 정확한 트리의 리프 노드 범위 체크가 필요할 수 있습니다.
                leaf_idx = parent_idx
                break
            else:
                # 여기서 s와 self.tree[left_child_idx]는 스칼라 값이어야 합니다.
                # 이 값이 배열이 되는 경우는 total_priority 문제 때문입니다.
                if s <= self.tree[left_child_idx]:
                    parent_idx = left_child_idx
                else:
                    s -= self.tree[left_child_idx]
                    parent_idx = right_child_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_priority(self):
        # 수정: 전체 우선순위의 합은 트리의 루트 노드 (인덱스 0)에 저장됩니다.
        return self.tree[0]

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment_per_sampling=0.001):
        self.tree = SumTree(capacity)
        self.alpha = alpha  # 우선순위 강도
        self.beta = beta    # 중요도 샘플링 가중치 강도
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.epsilon = 0.01 # 우선순위가 0이 되는 것을 방지

    def add(self, state, action, reward, next_state, done):
        # 1. 리플레이 버퍼가 비어있는지 확인합니다.
        if self.tree.n_entries == 0:
            # 버퍼가 비어있다면, 초기 max_priority를 1.0으로 설정합니다.
            # 이렇게 하면 np.max()를 빈 배열에 대해 호출하려는 시도를 방지합니다.
            max_priority = 1.0
        else:
            # 버퍼에 이미 경험이 있다면, 현재 리프 노드들 중에서 최대 우선순위를 계산합니다.
            max_priority = np.max(self.tree.tree[self.tree.capacity - 1 : self.tree.capacity - 1 + self.tree.n_entries])
            
            # 만약 어떤 이유로든 계산된 max_priority가 0이라면 1.0으로 설정하여
            # 우선순위가 0이 되는 것을 방지하고, 모든 경험이 학습될 기회를 가지도록 합니다.
            if max_priority == 0:
                max_priority = 1.0

        # 계산된 max_priority를 사용하여 SumTree에 경험을 추가합니다.
        self.tree.add(max_priority, (state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = []
        idxs = []
        # self.tree.total_priority는 이제 스칼라 값입니다.
        segment = self.tree.total_priority / batch_size
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b) # a, b가 스칼라 값이므로 random.uniform이 정상 작동합니다.
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        # 샘플링 확률 계산 시 total_priority도 스칼라 값이 되어야 합니다.
        # is_weights 계산도 정상적으로 이루어집니다.
        sampling_probabilities = np.array(priorities) / self.tree.total_priority
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max() # 정규화

        return batch, idxs, is_weights

    def update_priorities(self, tree_indices, td_errors):
        # td_errors는 NumPy 배열이므로 np.abs를 사용합니다.
        # self.epsilon을 더하여 우선순위가 0이 되는 것을 방지합니다.
        priorities = np.power(np.abs(td_errors) + self.epsilon, self.alpha)
        for idx, p in zip(tree_indices, priorities):
            self.tree.update(idx, p)