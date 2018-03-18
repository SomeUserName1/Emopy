# coding=utf-8
# Directory Setup
SHAPE_PREDICTOR_PATH = "../shape_predictor_68_face_landmarks.dat"
DATA_SET_DIR = "C:/Users/Fabi/DataSets/"
DATA_OUT_DIR = "C:/Users/Fabi/DataSets/EmoPyData"
MODEL_OUT_DIR = "../models"

# Session Setup
NETWORK_TYPE = "imagenn"  # mi for multi input or si for single input
SESSION = 'train'  # this value should be 'init_data', 'train' or 'predict'
IMG_SIZE = (64, 64)

# Training & Testing Setup
LEARNING_RATE = 1e-4
BATCH_SIZE = 64  # Batch sized used for traing.
STEPS_PER_EPOCH = 32000
EPOCHS = 50
AUGMENTATION = True

# Prediction Setup
PREDICTION_IMAGE = '0.png'
PREDICTION_VIDEO = "C:/Users/Fabi/DataSets/75Emotions.mp4"
"""Test type either image,video or webcam"""
PREDICTION_TYPE = 'image'
