import numpy as np

from preprocess.ImagePreprocessor import Preprocessor


class CapsPreprocessor(Preprocessor):
    def flow(self):
        """
            returns the next training batch
        """
        assert self.called, "Preprocessor should be called with path of dataset first to use flow method."
        while True:
            indexes = self.generate_indexes(True)
            for i in range(0, len(indexes) - self.batch_size, self.batch_size):
                current_indexes = indexes[i:i + self.batch_size]
                current_paths = self.train_image_paths[current_indexes]
                current_emotions = self.train_image_emotions[current_indexes]
                current_images = self.get_images(current_paths, self.augmentation).reshape(-1, self.input_shape[0],
                                                                                           self.input_shape[1],
                                                                                           self.input_shape[2])
                current_images = self.feature_extractor.extract(current_images)
                current_emotions = np.eye(self.classifier.get_num_class())[current_emotions]
                yield [current_images, current_emotions], [current_emotions, current_images]
