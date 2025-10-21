# gomoku_env.py
import numpy as np

class GomokuEnv:
    def __init__(self, board_size=15):
        self.board_size = board_size
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.current_player = 1  # 1: 흑, -1: 백

    def reset(self):
        """환경을 초기 상태로 리셋합니다."""
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        self.current_player = 1
        return self._get_state()

    def _get_state(self):
        """
        신경망 입력에 적합한 형태로 상태를 반환합니다.
        채널 0: 현재 플레이어의 돌
        채널 1: 상대 플레이어의 돌
        """
        state = np.zeros((self.board_size, self.board_size, 2), dtype=np.float32)
        state[:, :, 0] = (self.board == self.current_player).astype(np.float32)
        state[:, :, 1] = (self.board == -self.current_player).astype(np.float32)
        return state

    def get_legal_actions(self):
        """착수 가능한 모든 위치의 리스트를 반환합니다."""
        return list(zip(*np.where(self.board == 0)))

    def step(self, action):
        """
        행동을 실행하고 (다음 상태, 보상, 종료 여부, 정보)를 반환합니다.
        action: (row, col) 튜플
        """
        row, col = action
        if self.board[row, col]!= 0:
            # 비합법적인 수에 대한 페널티
            return self._get_state(), -10, True, {"error": "Invalid move"}

        self.board[row, col] = self.current_player
        
        # 승리 확인
        if self.check_win(self.current_player):
            reward = 1.0
            done = True
        # 무승부 확인 (바둑판이 꽉 참)
        elif len(self.get_legal_actions()) == 0:
            reward = 0.0
            done = True
        # 게임 진행 중
        else:
            reward = 0.0
            done = False

        # 플레이어 턴 전환
        self.current_player *= -1
        
        return self._get_state(), reward, done, {}

    def check_win(self, player):
        """특정 플레이어의 승리 여부를 확인합니다."""
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.board[r, c] == player:
                    # 가로, 세로, 대각선 2방향 확인
                    if c + 4 < self.board_size and all(self.board[r, c+i] == player for i in range(5)):
                        return True
                    if r + 4 < self.board_size and all(self.board[r+i, c] == player for i in range(5)):
                        return True
                    if r + 4 < self.board_size and c + 4 < self.board_size and all(self.board[r+i, c+i] == player for i in range(5)):
                        return True
                    if r + 4 < self.board_size and c - 4 >= 0 and all(self.board[r+i, c-i] == player for i in range(5)):
                        return True
        return False