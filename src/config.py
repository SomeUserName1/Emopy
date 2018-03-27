# coding=utf-8
# Directory Setup
SHAPE_PREDICTOR_PATH = "../shape_predictor_68_face_landmarks.dat"
DATA_SET_DIR = "E:/DataSets/"
DATA_OUT_DIR = "E:/DataSets/EmoPyData"
MODEL_OUT_DIR = "E:/models/EmoPy"

# Session Setup
NETWORK_TYPE = "vgg-net"  # mi for multi input or si for single input
SESSION = 'train'  # this value should be 'init_data', 'train' or 'predict'
IMG_SIZE = (48, 48)

# Training & Testing Setup
LEARNING_RATE = 1e-4
BATCH_SIZE = 32  # Batch sized used for training.
STEPS_PER_EPOCH = 512
EPOCHS = 5
AUGMENTATION = True

# Prediction Setup
PREDICTION_DATA = '../models/0.png'
PREDICTION_TYPE = 'image'
