from abc import abstractmethod, ABCMeta


class AbstractPreprocessor(object, metaclass=ABCMeta):
    @abstractmethod
    def load_dataset(self, path):
        """Load dataset with given path

        parameters
        ----------
        path    : str
            path to directory containing training and test directory.
        """
        pass

    @abstractmethod
    def __call__(self, path):
        """
        Pre-process given path

        Args:
            path: str
                path to directory containing training and test directory.

        """
        pass

    @abstractmethod
    def generate_indexes(self, random):
        """

        Args:
            random: If True use pseudo-randomization

        Returns:
            returns an array of indexes for the training data
        """
        pass

    @abstractmethod
    def flow(self):
        """
            returns the next training batch
        """
        pass

    @abstractmethod
    def sanitize(self, image):
        """

        Args:
            image:

        Returns:

        """
        pass

    @abstractmethod
    def get_images(self, paths, augmentation):
        """

        Args:
            paths:
            augmentation:

        Returns:

        """
        pass

    @abstractmethod
    def get_faces(self, frame, detector):
        """

        Args:
            frame:
            detector:

        Returns:

        """
        pass

    @abstractmethod
    def load_sequencial_dataset(self, path, max_sequence_length):
        """

        Args:
            path:
            max_sequence_length:

        Returns:

        """
        pass
