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

    @staticmethod
    def arg_max(array):
        """

        Args:
            array:

        Returns:

        """
        index = [[0]]
        for j, img in enumerate(array):
            max_value = img[0]
            for i, el in enumerate(img):
                print(el)
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
        img_emotions = []
        for i, img in enumerate(predictions):
            for j in range(0, self.classifier.get_num_class()):
                img_emotions.append((self.classifier.get_string(j), ' = ', f'{img[i][j]:.3f}'))
        emotions.append(img_emotions)
        self.overlay(image, rectangles, emotions)

    @staticmethod
    def overlay(frame, rectangles, text, color=(48, 12, 160)):
        """
        Draw rectangles and text over image

        Args:
            frame (Mat): Image
            rectangles (list): Coordinates of rectangles to draw
            text (list): List of emotions to write
            color (tuple): Box and text color
        """
        j = 1
        for i, rectangle in enumerate(rectangles):
            cv2.rectangle(frame, (rectangle.left(), rectangle.top()), (rectangle.right(), rectangle.bottom()), color)
            for mTuple in text[i]:
                cv2.putText(frame, "".join(mTuple), (rectangle.right() + 10, rectangle.top() + j * 12),
                            cv2.FONT_HERSHEY_DUPLEX, 0.2, color)
                j += 1
        return frame
