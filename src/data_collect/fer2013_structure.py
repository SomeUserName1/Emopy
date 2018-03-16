import pandas as pd
import numpy as np
import random, scipy.misc, os, cv2
from PIL import Image
import sys
import matplotlib.pyplot as plt
import matplotlib.image as image




# fer2013 dataset:
# Training       28709
# PrivateTest     3589
# PublicTest      3589

# emotion labels from FER2013:
emotion_in = {'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3,
              'Sad': 4, 'Surprise': 5, 'Neutral': 6}
emotions_out = ['anger', 'neutral', 'disgust', 'fear', 'happy', 'sad', 'surprise']


def reconstruct_img(pix_str, size=(48,48)):
    pix_arr = np.fromstring(pix_str, dtype=int, sep=' ')
    im = Image.fromarray(pix_arr.reshape(size))
    return im


def make_dirs(path):
    train = os.path.join(path + 'train')
    test = os.path.join(path + 'test')
    for path in [train, test]:
        if not os.path.exists(path):
            os.mkdir(path)
        for emotion in emotions_out:
            if not os.path.exists(train + emotion):
                os.mkdir(path + emotion)


def map_emotions(emo_in):
    if emo_in == 'Angry':
        emo_out = emotions_out[0]

    elif emo_in == 'Disgust':
        emo_out = emotions_out[2]

    elif emo_in == 'Fear':
        emo_out = emotions_out[3]

    elif emo_in == 'Happy':
        emo_out = emotions_out[4]

    elif emo_in == 'Sad':
        emo_out = emotions_out[5]

    elif emo_in == 'Surprise':
        emo_out = emotions_out[6]

    elif emo_in == 'Neutral':
        emo_out = emotions_out[1]
    else:
        emo_out = 'undef'
    return emo_out


def main(csv_path='~/DataSets/fer2013/fer2013.csv', out='C:/Users/Fabi/DataSets/EmoPyData'):
    data = pd.read_csv(csv_path)
    train = data[data.Usage == 'Training']
    train.name = 'train'
    test = pd.concat([data[data.Usage == 'PrivateTest'], data[data.Usage == 'PublicTest']], ignore_index=True)
    test.name = 'test'
    make_dirs(out)
    for d_set in [train, test]:
        for emotion in emotion_in:
            em_data = d_set[d_set.emotion == emotion_in[emotion]]
            counter = 0
            for im_arr in em_data.pixels:
                img = reconstruct_img(im_arr)
                emo_out = map_emotions(emotion)
                image.imsave(out + '/' + d_set.name + '/' + emo_out + '/' + '%s.png' % counter, img)
                counter = counter +1

if __name__ == '__main__':
    main()
