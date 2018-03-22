# coding=utf-8
# Directory Setup
SHAPE_PREDICTOR_PATH = "../shape_predictor_68_face_landmarks.dat"
DATA_SET_DIR = "C:/Users/Fabi/DataSets/"
DATA_OUT_DIR = "C:/Users/Fabi/DataSets/EmoPyData"
MODEL_OUT_DIR = "../models"

# Session Setup
NETWORK_TYPE = "caps"  # mi for multi input or si for single input
SESSION = 'train'  # this value should be 'init_data', 'train' or 'predict'
IMG_SIZE = (64, 64)

# Training & Testing Setup
LEARNING_RATE = 0.0001
BATCH_SIZE = 32  # Batch sized used for training.
STEPS_PER_EPOCH = 500
EPOCHS = 15
AUGMENTATION = False

# Prediction Setup
PREDICTION_IMAGE = '0.png'
PREDICTION_VIDEO = "C:/Users/Fabi/DataSets/75Emotions.mp4"
"""Test type either image,video or webcam"""
PREDICTION_TYPE = 'image'
