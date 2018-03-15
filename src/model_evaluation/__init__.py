import os

import cv2
from keras.models import model_from_json


def load_model(path):
    """

    Args:
        path:

    Returns:

    """
    with open(path + ".json") as json_file:
        model = model_from_json(json_file.read())
        model.load_weights(path + ".h5")
        return model


def show_sequence_images(path):
    """

    Args:
        path:
    """
    for img_file in os.listdir(path):
        img = cv2.imread(os.path.join(path, img_file))
        cv2.imshow("Sequence", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def test_sequences():
    """

    """
    # TODO ck-split info gathering
    # dataset_path = "dataset/ck-split/test"
    # for emotion_dir in os.listdir(dataset_path):
    raise NotImplementedError
