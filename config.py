import numpy as np
import tensorflow as tf
import json

# Input dimensions
INPUT_WIDTH = 320
INPUT_HEIGHT = 240

# Output dimensions
OUTPUT_WIDTH = 384
OUTPUT_HEIGHT = 240

NUMBER_OF_CLASSES = 4

with open("/home/mahdi/Desktop/hslSegment/SegLight/dataset/label_classes.json", "r") as js:
    file = json.load(js)
    js.close()

COLOR_MAP = {}
for item in file :
    if item["name"] == "field" :
        COLOR_MAP["filed"] = item["png_index"]
    elif item["name"] == "line" :
        COLOR_MAP["line"] = item["png_index"]
    elif item["name"] == "ball" :
        COLOR_MAP["ball"] = item["png_index"]
    elif item["name"] == "background" :
        COLOR_MAP["background"] = item["png_index"]

# COLOR_MAP = {
#     'background': [np.array([0.0, 0.0, 0.0], dtype=np.float32),],
#     'field': [np.array([0.0, 255.0, 0.0], dtype=np.float32),],
#     'line': [np.array([255.0, 255.0, 255.0], dtype=np.float32),],
#     'ball':[np.array([0.0, 0.0, 255.0], dtype=np.float32),],
# }

MAX_MODELS_TO_KEEP = 5

DEBUG_ENABLED = False