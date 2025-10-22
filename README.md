# 오목 AI와 얼굴 인식

이 프로젝트는 우선순위 경험 재현(PER)을 사용하는 Dueling Double Deep Q-Network(DDDQN) 기반의 오목 AI입니다. AI 및 얼굴 인식 로그인을 위한 API를 제공하는 Flask 서버를 포함합니다.

> **참고:** 웹 기반 오목 게임 서비스와 사용자 인터페이스는 별도의 [Gomoku 프로젝트](https://github.com/haejun1213/Gomoku)에서 구현되어 있습니다.  
> 이 저장소는 AI 로직과 얼굴 인식 API 제공에 집중합니다.


## 프로젝트 구조

- `ai_server.py`: 오목 예측 및 얼굴 인식을 위한 REST API를 제공하는 Flask 서버입니다.
- `train_model/`: AI 모델 학습을 위한 코드가 포함된 디렉터리입니다.
  - `train.py`: 오목 AI 에이전트를 학습시키는 메인 스크립트입니다.
  - `agent.py`: PER을 사용하는 DDDQN 에이전트를 구현합니다.
  - `gomoku_env.py`: 오목 게임 환경을 정의합니다.
  - `model.py`: Dueling DQN 신경망 아키텍처를 정의합니다.
  - `replay_buffer.py`: 우선순위 재현 버퍼를 구현합니다.
- `model/best.h5`: 서버에서 사용하는 학습된 모델입니다.
- `.gitignore`: 버전 관리에서 무시할 파일 및 디렉터리를 지정합니다.

## 설치

먼저, 저장소를 복제합니다. 그런 다음 필요한 Python 라이브러리를 설치합니다.

```bash
pip install flask flask_cors numpy tensorflow deepface pillow
```

## 사용법

### AI 서버 실행

AI 서버를 시작하려면 다음 명령을 실행합니다.

```bash
python ai_server.py
```

서버는 `http://0.0.0.0:5000`에서 시작됩니다.

### 모델 학습

새 모델을 학습시키려면 `train_model` 디렉터리로 이동하여 `train.py` 스크립트를 실행합니다.

```bash
cd train_model
python train.py
```

학습된 모델은 `train_model/models` 디렉터리에 저장됩니다.

## API 엔드포인트

`ai_server.py`는 다음 엔드포인트를 제공합니다.

### 오목

- **POST /predict**
  - 현재 바둑판 상태와 AI의 플레이어 번호가 포함된 JSON 페이로드를 받습니다.
  - AI의 다음 수를 반환합니다.
  - **요청 본문:**
    ```json
    {
      "board": [[0, 0, ...], ...],
      "aiPlayer": 1
    }
    ```
  - **성공 응답:**
    ```json
    {
      "x": 7,
      "y": 7
    }
    ```

### 얼굴 인식

- **POST /encode-face**
  - 이미지 데이터 URL을 받아 얼굴 인코딩을 반환합니다.
  - **요청 본문:**
    ```json
    {
      "imageDataUrl": "data:image/jpeg;base64,..."
    }
    ```
  - **성공 응답:**
    ```json
    {
      "success": true,
      "encoding": [...]
    }
    ```

- **POST /verify-face**
  - 알려진 얼굴 인코딩을 새 이미지와 비교하여 신원을 확인합니다.
  - **요청 본문:**
    ```json
    {
      "knownEncoding": [...],
      "targetImageDataUrl": "data:image/jpeg;base64,..."
    }
    ```
  - **성공 응답:**
    ```json
    {
      "verified": true,
      "distance": 0.34
    }
    ```
