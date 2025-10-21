
from flask import Flask, request, jsonify
from flask_cors import CORS  # CORS 라이브러리 임포트
import numpy as np
import tensorflow as tf
from deepface import DeepFace # dlib 대신 DeepFace 사용
import base64
import io
from PIL import Image
import os

app = Flask(__name__)
CORS(app) # 모든 경로에 대해 CORS 허용

# --- 기존 오목 AI 모델 로드 ---
MODEL_PATH = './model/best.h5'
omok_model = None
try:
    if os.path.exists(MODEL_PATH):
        omok_model = tf.keras.models.load_model(MODEL_PATH)
        print(f"✅ 오목 AI 모델 로딩 성공: {MODEL_PATH}")
    else:
        print(f"❌ 오목 AI 모델 파일({MODEL_PATH})을 찾을 수 없습니다.")
except Exception as e:
    print(f"❌ 오목 AI 모델 로딩 중 오류 발생: {e}")

# --- 기존 오목 AI 관련 함수들 (수정 없음) ---
def preprocess_board(board, aiPlayer):
    board_np = np.array(board)
    channel_player = (board_np == aiPlayer).astype(np.float32)
    channel_opponent = ((board_np != 0) & (board_np != aiPlayer)).astype(np.float32)
    input_data = np.stack([channel_player, channel_opponent], axis=-1)
    return np.expand_dims(input_data, axis=0)

@app.route('/predict', methods=['POST'])
def predict():
    if omok_model is None:
        return jsonify({'error': '오목 AI 모델이 로드되지 않았습니다.'}), 500
    try:
        data = request.get_json()
        board, aiPlayer = data.get('board'), data.get('aiPlayer')
        input_data = preprocess_board(board, aiPlayer)
        policy_pred, value_pred = omok_model.predict(input_data, verbose=0)
        policy_pred = policy_pred[0]
        board_np = np.array(board)
        policy_mask = board_np.flatten() == 0
        masked_policy = policy_pred * policy_mask
        move_idx = np.argmax(masked_policy)
        y, x = divmod(int(move_idx), 15)
        return jsonify({'x': x, 'y': y})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ====================================================================
# ★★★ 얼굴 인식 로그인 기능 추가 ★★★
# ====================================================================

# Base64 Data URL을 이미지(numpy 배열)로 변환하는 함수
def data_url_to_image(data_url):
    try:
        header, encoded = data_url.split(",", 1)
        image_data = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        return np.array(image)
    except Exception as e:
        print(f"이미지 데이터 변환 오류: {e}")
        return None

# 1. 얼굴 등록 시 '인코딩'을 생성하는 API
@app.route('/encode-face', methods=['POST'])
def encode_face():
    data = request.get_json()
    if not data or 'imageDataUrl' not in data:
        return jsonify({'success': False, 'error': '이미지 데이터가 없습니다.'}), 400
    try:
        image = data_url_to_image(data.get('imageDataUrl'))
        if image is None: return jsonify({'success': False, 'error': '잘못된 이미지 데이터 형식입니다.'})
        
        # ★★★ 더 강력한 얼굴 탐지기(mtcnn)로 변경 ★★★
        embedding_objs = DeepFace.represent(img_path=image, model_name="VGG-Face", detector_backend='mtcnn', enforce_detection=True)
        embedding = embedding_objs[0]['embedding']
        
        return jsonify({'success': True, 'encoding': embedding})
    except ValueError:
        return jsonify({'success': False, 'error': '이미지에서 얼굴을 명확히 인식할 수 없습니다. 더 밝은 곳에서 정면을 응시해주세요.'})
    except Exception as e:
        return jsonify({'success': False, 'error': f'인코딩 오류: {str(e)}'}), 500

# 2. 로그인 시 두 얼굴을 '비교'하는 API
@app.route('/verify-face', methods=['POST'])
def verify_face():
    data = request.get_json()
    if not data or 'knownEncoding' not in data or 'targetImageDataUrl' not in data:
         return jsonify({'error': '필요한 데이터가 없습니다.'}), 400
    try:
        known_encoding = np.array(data.get('knownEncoding'))
        target_image = data_url_to_image(data.get('targetImageDataUrl'))
        if target_image is None:
            return jsonify({'verified': False, 'error': '웹캠 이미지 처리 중 오류 발생'})

        # 1. 새로 찍은 웹캠 이미지에서 얼굴 인코딩을 추출
        # ★★★ 더 강력한 얼굴 탐지기(mtcnn)로 변경 ★★★
        target_embedding_objs = DeepFace.represent(
            img_path=target_image, 
            model_name="VGG-Face", 
            detector_backend='mtcnn',
            enforce_detection=True # 얼굴을 반드시 찾아야 함
        )
        target_encoding = target_embedding_objs[0]['embedding']
            
        # 2. 두 인코딩 사이의 거리를 계산
        distance = np.linalg.norm(known_encoding - np.array(target_encoding))
        
        # 3. 임계값은 유연하게 유지
        threshold = 0.6
        is_match = distance <= threshold
            
        print(f"얼굴 비교 결과: 거리 = {distance:.4f}, 임계값 = {threshold}, 일치 여부 = {is_match}")
        
        return jsonify({'verified': bool(is_match), 'distance': distance})
    except ValueError:
        return jsonify({'verified': False, 'error': '현재 웹캠 화면에서 얼굴을 인식할 수 없습니다.'})
    except Exception as e:
        return jsonify({'verified': False, 'error': f'얼굴 비교 중 오류 발생: {str(e)}'}), 500


if __name__ == '__main__':
    print("🚀 Python AI 서버를 시작합니다. (오목 AI: /predict, 얼굴인식: /encode-face, /verify-face)")
    app.run(host='0.0.0.0', port=5000, debug=False)