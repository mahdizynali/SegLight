import numpy as np

# Input dimensions
INPUT_WIDTH = 320
INPUT_HEIGHT = 240

# Output dimensions
OUTPUT_WIDTH = 384
OUTPUT_HEIGHT = 240

NUMBER_OF_CLASSES = 4

COLOR_MAP = {
    'background': [np.array([0.0, 0.0, 0.0], dtype=np.float32),],
    'grass': [np.array([0.0, 255.0, 0.0], dtype=np.float32),],
    'line': [np.array([255.0, 255.0, 255.0], dtype=np.float32),],
    'ball':[np.array([0.0, 0.0, 255.0], dtype=np.float32),],
}

MAX_MODELS_TO_KEEP = 5

DEBUG_ENABLED = False