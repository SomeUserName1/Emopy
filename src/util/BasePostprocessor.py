import cv2


class PostProcessor(object):
    """
    """

    def __init__(self, classifier):
        """

        Args:
            classifier:
        """
        self.classifier = classifier

    def arg_max(self, array):
        """

        Args:
            array:

        Returns:

        """
        max_value = array[0]
        index = 0
        for i, el in enumerate(array):
            if el > max_value:
                index = i
                max_value = el
        return index

    def __call__(self, image, rectangles, predictions):
        """

        Args:
            image:
            rectangles:
            predictions:
        """
        emotions = []
        for i in range(len(predictions)):
            emotions.append(self.classifier.get_string(self.arg_max(predictions[i])))
        self.overlay(image, rectangles, emotions)

    def overlay(self, frame, rectangles, text, color=(48, 12, 160)):
        """
        Draw rectangles and text over image

        Args:
            frame (Mat): Image
            rectangles (list): Coordinates of rectangles to draw
            text (list): List of emotions to write
            color (tuple): Box and text color

        Returns:
            list: Most dominant emotion of each face
        """

        for i, rectangle in enumerate(rectangles):
            cv2.rectangle(frame, (rectangle.left(), rectangle.top()), (rectangle.right(), rectangle.bottom()), color)
            cv2.putText(frame, text[i], (rectangle.left() + 10, rectangle.top() + 10), cv2.FONT_HERSHEY_DUPLEX, 0.4,
                        color)
        return frame
