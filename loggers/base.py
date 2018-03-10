from __future__ import print_function

import os
import sys
import time

import numpy as np

from train_config import PATH2SAVE_MODELS, DATA_SET_DIR, BATCH_SIZE, EPOCHS, LEARNING_RATE


class EmopyLogger(object):
    """
    """
    def __init__(self, output_files=[sys.stdout]):
        """

        Args:
            output_files:
        """
        self.output_files = output_files

    def log(self, string):
        """

        Args:
            string:
        """
        for f in self.output_files:
            if f == sys.stdout:
                print(string)
            elif type(f) == str:
                with open(f, "a+") as out_file:
                    out_file.write(string + "\n")

    def add_log_file(self, log_file):
        """

        Args:
            log_file:
        """
        self.output_files.append(log_file)

    def log_model(self, models_local_folder, score):
        """

        Args:
            models_local_folder:
            score:
        """
        model_number = np.fromfile(os.path.join(PATH2SAVE_MODELS, models_local_folder, "model_number.txt"), dtype=int)
        model_file_name = models_local_folder + "-" + str(model_number[0] - 1)

        self.log("**************************************")
        self.log("Trained model " + model_file_name + ".json")
        self.log(time.strftime("%A %B %d,%Y %I:%M%p"))
        self.log("Dataset dir: " + DATA_SET_DIR)
        self.log("Parameters")
        self.log("_______________________________________")
        self.log("Batch-Size    : " + str(BATCH_SIZE))
        self.log("Epoches       : " + str(EPOCHS))
        self.log("Learning rate : " + str(LEARNING_RATE))
        self.log("_______________________________________")
        self.log("Loss          : " + str(score[0]))
        self.log("Accuracy      : " + str(score[1]))
        self.log("**************************************")
