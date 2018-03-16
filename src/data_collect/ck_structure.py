# coding=utf-8
import argparse
import glob
import os
import random
import shutil

CK_DIR = "C:/Users/Fabi/DataSets/CK/images"
TRAIN_OUT_DIR = "C:/Users/Fabi/DataSets/EmoPyData/train"
TEST_OUT_DIR = "C:/Users/Fabi/DataSets/EmoPyData/test"


def create_folder_structure():
    """
    Create the folder structure needed by the preprocessor
    """
    if not os.path.exists(TRAIN_OUT_DIR):
        os.mkdir(TRAIN_OUT_DIR)

    for sdir in os.listdir(CK_DIR):
        spath = os.path.join(CK_DIR, sdir)
        for ddir in os.listdir(spath):
            dpath = os.path.join(spath, ddir)
            if os.path.isdir(dpath):
                os.chdir(dpath)
            else:
                print("not a dir:", dpath)
            emotion_txt = glob.glob('*emotion*')
            if len(emotion_txt) == 1:
                add_emotion(os.path.join(dpath, emotion_txt[0]))
            elif len(emotion_txt) > 1:
                print(emotion_txt)
    test()


def add_emotion(path):
    """
        Copies the image in the correct folder according to the label

    :param path: Path of the emotion.txt file to be appended to the data set
    """
    emotion_txt = os.open(path, os.O_RDONLY)
    emotion = int(os.read(emotion_txt, 4))

    if emotion == 1:
        label = 'anger'
    elif emotion == 2:
        label = 'neutral'
    elif emotion == 3:
        label = 'disgust'
    elif emotion == 4:
        label = 'fear'
    elif emotion == 5:
        label = 'happy'
    elif emotion == 6:
        label = 'sad'
    elif emotion == 7:
        label = 'surprise'
    else:
        label = "undef"

    parts = path.split("_")
    img_path = '.'.join(("_".join(parts[:3]), 'png'))
    img_name = os.path.basename(img_path)
    dst_path = os.path.join(TRAIN_OUT_DIR, label)

    if not os.path.exists(dst_path):
        os.mkdir(dst_path)

    shutil.copy(img_path, os.path.join(dst_path, img_name))


def test():
    """
        choose and copy test 7 images
    """
    above = os.path.join(TRAIN_OUT_DIR, '..')
    os.chdir(above)
    if not os.path.exists("test"):
        os.mkdir("test")

    for dir in os.listdir(TRAIN_OUT_DIR):
        cur_dir = os.path.join(TRAIN_OUT_DIR, dir)
        list_curr_dir = os.listdir(cur_dir)
        random.seed()
        rand_num = random.randint(0, len(list_curr_dir) - 1)
        rand_img = list_curr_dir[rand_num]
        rand_img_path = os.path.join(cur_dir, rand_img)
        dst_path = os.path.join("test", dir)
        if not os.path.exists(dst_path):
            os.mkdir(dst_path)

        shutil.move(rand_img_path, os.path.join(dst_path, os.path.basename(rand_img_path)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ck", default=CK_DIR, type=str)
    parser.add_argument("--out", default=TRAIN_OUT_DIR, type=str)

    args = parser.parse_args()
    create_folder_structure()
