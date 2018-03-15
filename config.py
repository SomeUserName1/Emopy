# coding=utf-8
# global
SESSION = 'test'  # this value should be either 'test' or 'train'
VERBOSE = True
IMG_SIZE = (64, 64)
SHAPE_PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"

# train
BATCH_SIZE = 32  # Batch sized used for traing.
EPOCHS = 10
LEARNING_RATE = 1e-3
PATH2SAVE_MODELS = "models"
DATA_SET_DIR = "C:/Users/Fabi/DataSets/CK/EmoPyData"
LOG_DIR = "logs"
STEPS_PER_EPOCH = 640
NETWORK_TYPE = "mi"  # mi for multi input or si for single input
AUGMENTATION = True

# test
MODEL_PATH = "models/minn/minn-0"  # model name used for testing
TEST_IMAGE = '0.png'
TEST_VIDEO = "C:/Users/Fabi/DataSets/75Emotions.mp4"
"""Test type either image,video or webcam"""
TEST_TYPE = 'image'
