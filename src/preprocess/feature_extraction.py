from __future__ import print_function
import numpy as np, os, cv2, dlib

from config import IMG_SIZE


class FeatureExtractor(object):
    """Base class for Feature extactors
    """

    def __init__(self):
        """

        """
        pass

    def extract(self, images):
        """

        Args:
            images:
        """
        raise NotImplementedError('abstract method extract should be implemented!')


class ImageFeatureExtractor(FeatureExtractor):
    """
    """

    def __init__(self, **kwargs):
        """

        Args:
            kwargs:
        """
        FeatureExtractor.__init__(self, **kwargs)

    def extract(self, images):
        """

        Args:
            images:

        Returns:

        """
        new_images = np.array(images).astype(float) / 255
        return new_images


class DlibFeatureExtractor(FeatureExtractor):
    """
    """

    def __init__(self, predictor, **kwargs):
        """

        Args:
            predictor:
            kwargs:
        """
        FeatureExtractor.__init__(self, **kwargs)
        self.predictor = predictor

    def get_dlib_points(self, image):
        """

        Args:
            image:

        Returns:

        """
        face = dlib.rectangle(0, 0, image.shape[1] - 1, image.shape[0] - 1)
        img = image.reshape(IMG_SIZE[0], IMG_SIZE[1])
        shapes = self.predictor(img, face)
        parts = shapes.parts()
        output = np.zeros((68, 2))
        for i, point in enumerate(parts):
            output[i] = [point.x, point.y]
        output = np.array(output).reshape((1, 68, 2))
        return output

    def to_dlib_points(self, images):
        """

        Args:
            images:

        Returns:

        """
        output = np.zeros((len(images), 1, 68, 2))
        centroids = np.zeros((len(images), 2))
        for i in range(len(images)):
            dlib_points = self.get_dlib_points(images[i])[0]
            centroid = np.mean(dlib_points, axis=0)
            centroids[i] = centroid
            output[i][0] = dlib_points
        return output, centroids

    def get_distances_angles(self, all_dlib_points, centroids):
        """

        Args:
            all_dlib_points:
            centroids:

        Returns:

        """
        all_distances = np.zeros((len(all_dlib_points), 1, 68, 1))
        all_angles = np.zeros((len(all_dlib_points), 1, 68, 1))
        for i in range(len(all_dlib_points)):
            dists = np.linalg.norm(centroids[i] - all_dlib_points[i][0], axis=1)
            angles = self.get_angles(all_dlib_points[i][0], centroids[i])
            all_distances[i][0] = dists.reshape(1, 68, 1)
            all_angles[i][0] = angles.reshape(1, 68, 1)
        return all_distances, all_angles

    def angle_between(self, p1, p2):
        """

        Args:
            p1:
            p2:

        Returns:

        """
        ang1 = np.arctan2(*p1[::-1])
        ang2 = np.arctan2(*p2[::-1])
        return (ang1 - ang2) % (2 * np.pi)

    def get_angles(self, dlib_points, centroid):
        """

        Args:
            dlib_points:
            centroid:

        Returns:

        """
        output = np.zeros((68))
        for i in range(68):
            angle = self.angle_between(dlib_points[i], centroid)
            output[i] = angle
        return output

    def extract(self, images):
        """

        Args:
            images:

        Returns:

        """

        dlib_points, centroids = self.to_dlib_points(images)

        distances, angles = self.get_distances_angles(dlib_points, centroids)

        IMAGE_CENTER = np.array(IMG_SIZE) / 2
        IMG_WIDTH = IMG_SIZE[1]
        # normalize
        dlib_points = (dlib_points - IMAGE_CENTER) / IMG_WIDTH
        dlib_points = dlib_points.reshape((-1, 1, 68, 2))

        distances /= 50.0;
        distances = distances.reshape(-1, 1, 68, 1)

        angles /= (2 * np.pi)
        angles = angles.reshape(-1, 1, 68, 1)
        images = images.astype(np.float32) / 255

        return images, dlib_points, distances, angles


##### face extractor (init and main from data_collect) where is that called?
# call from main:
# collect_faces("/home/mtk/iCog/projects/emopy/test-videos/1.mp4", "/home/mtk/dataset/from_videos")

detector = dlib.get_frontal_face_detector()


def save_face(frame, face, file_path):
    """

    Args:
        frame:
        face:
        file_path:
    """
    face_image = frame[
                 max(face.top() - 100, 0):min(face.bottom() + 100, frame.shape[0]),
                 max(face.left() - 100, 0):min(face.right() + 100, frame.shape[1])

                 ]
    if face_image.shape[0] > 100:
        cv2.imwrite(file_path, face_image)


def collect_faces(video_path, output_path):
    """

    Args:
        video_path:
        output_path:
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Unable to open video", video_path)
        exit(0)
    print("Processing video")
    number_of_faces = 0
    saved_faces = 0
    while cap.isOpened():
        ret, frame = cap.read()
        faces = detector(frame)
        for face in faces:
            number_of_faces += 1
            if number_of_faces % 5 == 0:  # TODO mod 5 ??? wtf; maybe 5 faces = 1 batch
                saved_faces += 1
                save_face(frame, face, os.path.join(output_path, str(saved_faces) + ".png"))
                print("collected ", saved_faces, "faces")