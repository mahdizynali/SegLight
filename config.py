import os
import cv2
import json
import glob
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanIoU, Mean
from tensorflow.keras.losses import SparseCategoricalCrossentropy

BASE_DIR = os.path.dirname(__file__)

IMAGES_PATH = BASE_DIR + "/dataset/images/"
LABELS_PATH = BASE_DIR + "/dataset/labels/"

# Input dimensions
INPUT_WIDTH = 320
INPUT_HEIGHT = 240

# Output dimensions
OUTPUT_WIDTH = 384
OUTPUT_HEIGHT = 240

NUMBER_OF_CLASSES = 4
BATCH_SIZE = 32
EPOCH_NUMBER = 100
LEARNING_RATE = 0.001

# read datasheet
with open(BASE_DIR + "/dataset/label_classes.json", "r") as js:
    file = json.load(js)
    js.close()

# import color map
COLOR_MAP = {}
for item in file:
    name = item["name"]
    color = item["png_index"]
    COLOR_MAP[name] = color

# COLOR_MAP = {
#     'background': [np.array([0.0, 0.0, 0.0], dtype=np.float32),],
#     'field': [np.array([0.0, 255.0, 0.0], dtype=np.float32),],
#     'line': [np.array([255.0, 255.0, 255.0], dtype=np.float32),],
#     'ball':[np.array([0.0, 0.0, 255.0], dtype=np.float32),],
# }

MAX_MODELS_TO_KEEP = 5

DEBUG_ENABLED = False