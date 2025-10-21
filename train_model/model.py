# model.py
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Lambda
from tensorflow.keras.models import Model

def create_dueling_dqn_model(input_shape, num_actions):
    """Dueling DQN 모델을 생성합니다."""
    inputs = Input(shape=input_shape)

    # 컨볼루션 레이어: 공간적 특징 추출
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Flatten()(x)

    # Value Stream
    value_stream = Dense(512, activation='relu')(x)
    value = Dense(1, name='value')(value_stream)

    # Advantage Stream
    advantage_stream = Dense(512, activation='relu')(x)
    advantage = Dense(num_actions, name='advantage')(advantage_stream)

    # 스트림 결합
    def aggregate_streams(streams):
        v, a = streams
        # Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
        return v + (a - tf.reduce_mean(a, axis=1, keepdims=True))

    q_values = Lambda(aggregate_streams, name='q_values')([value, advantage])

    model = Model(inputs=inputs, outputs=q_values)
    return model