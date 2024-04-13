import os
import cv2
import json
import glob
import random
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import MeanIoU, Mean
from tensorflow.keras.losses import SparseCategoricalCrossentropy, CategoricalCrossentropy

BASE_DIR = os.path.dirname(__file__)

IMAGES_PATH = BASE_DIR + "/dataset/images/"
LABELS_PATH = BASE_DIR + "/dataset/labels/"

# Input dimensions
INPUT_WIDTH = 320
INPUT_HEIGHT = 240

# Output dimensions
OUTPUT_WIDTH = 320
OUTPUT_HEIGHT = 240

NUMBER_OF_CLASSES = 3
BATCH_SIZE = 64
EPOCH_NUMBER = 200
LEARNING_RATE = 0.001

# read datasheet
with open(BASE_DIR + "/dataset/label_classes.json", "r") as js:
    file = json.load(js)
    js.close()

# # import color map
# COLOR_MAP = {}
# for item in file:
#     name = item["name"]
#     color = item["png_index"]
#     COLOR_MAP[name] = color

# i have consider ball and line as one class
COLOR_MAP = { #bgr
    'background': [0.0, 0.0, 0.0],
    'field': [0.0, 255.0, 0.0],
    'line': [255.0, 255.0, 255.0]
}

DEBUG_ENABLED = False
