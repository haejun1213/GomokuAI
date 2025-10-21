# Gomoku AI with Face Recognition

This project is a Gomoku (Five in a Row) AI that uses a Dueling Double Deep Q-Network (DDDQN) with Prioritized Experience Replay (PER). It includes a Flask server to provide an API for the AI and for face recognition login.

## Project Structure

- `ai_server.py`: A Flask server that provides REST APIs for Gomoku predictions and face recognition.
- `train_model/`: Directory containing the code for training the AI model.
  - `train.py`: The main script to train the Gomoku AI agent.
  - `agent.py`: Implements the DDDQN with PER agent.
  - `gomoku_env.py`: Defines the Gomoku game environment.
  - `model.py`: Defines the Dueling DQN neural network architecture.
  - `replay_buffer.py`: Implements the Prioritized Replay Buffer.
- `model/best.h5`: The trained model used by the server.
- `.gitignore`: Specifies which files and directories to ignore in version control.

## Installation

First, clone the repository. Then, install the required Python libraries.

```bash
pip install flask flask_cors numpy tensorflow deepface pillow
```

## Usage

### Running the AI Server

To start the AI server, run the following command:

```bash
python ai_server.py
```

The server will start on `http://0.0.0.0:5000`.

### Training the Model

To train a new model, navigate to the `train_model` directory and run the `train.py` script:

```bash
cd train_model
python train.py
```

The trained models will be saved in the `train_model/models` directory.

## API Endpoints

The `ai_server.py` provides the following endpoints:

### Gomoku

- **POST /predict**
  - Takes a JSON payload with the current board state and the AI's player number.
  - Returns the AI's next move.
  - **Request Body:**
    ```json
    {
      "board": [[0, 0, ...], ...],
      "aiPlayer": 1
    }
    ```
  - **Success Response:**
    ```json
    {
      "x": 7,
      "y": 7
    }
    ```

### Face Recognition

- **POST /encode-face**
  - Takes an image data URL and returns a facial encoding.
  - **Request Body:**
    ```json
    {
      "imageDataUrl": "data:image/jpeg;base64,..."
    }
    ```
  - **Success Response:**
    ```json
    {
      "success": true,
      "encoding": [...]
    }
    ```

- **POST /verify-face**
  - Compares a known facial encoding with a new image to verify identity.
  - **Request Body:**
    ```json
    {
      "knownEncoding": [...],
      "targetImageDataUrl": "data:image/jpeg;base64,..."
    }
    ```
  - **Success Response:**
    ```json
    {
      "verified": true,
      "distance": 0.34
    }
    ```
