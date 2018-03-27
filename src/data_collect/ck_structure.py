# coding=utf-8
import argparse
import glob
import os
import random
import shutil

emotions_out = ['anger', 'neutral', 'disgust', 'fear', 'happy', 'sad', 'surprise']


def make_dirs(root_path):
    if not os.path.exists(root_path):
        os.mkdir(root_path)
    train = root_path + '/train'
    test = root_path + '/test'
    for path in [train, test]:
        if not os.path.exists(path):
            os.mkdir(path)
        for emotion in emotions_out:
            if not os.path.exists(path + '/' + emotion):
                os.mkdir(path + '/' + emotion)


def create_folder_structure(ck_dir, out_dir):
    """
    Create the folder structure needed by the preprocessor
    """
    make_dirs(out_dir)
    train_out_dir = out_dir + '/train'
    if not os.path.exists(train_out_dir):
        os.mkdir(train_out_dir)

    for sdir in os.listdir(ck_dir):
        spath = os.path.join(ck_dir, sdir)
        for ddir in os.listdir(spath):
            dpath = os.path.join(spath, ddir)
            if os.path.isdir(dpath):
                os.chdir(dpath)
            else:
                print("not a dir:", dpath)
            emotion_txt = glob.glob('*emotion*')
            if len(emotion_txt) == 1:
                add_emotion(os.path.join(dpath, emotion_txt[0]), train_out_dir)
            elif len(emotion_txt) > 1:
                print(emotion_txt)
    test(train_out_dir)


def add_emotion(path, train_out_dir):
    """
        Copies the image in the correct folder according to the label

    :param train_out_dir:
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
    dst_path = os.path.join(train_out_dir, label)

    if not os.path.exists(dst_path):
        os.mkdir(dst_path)

    shutil.copy(img_path, os.path.join(dst_path, img_name))


def test(train_out_dir):
    """
        choose and copy test 7 images
    """
    above = os.path.join(train_out_dir, '..')
    os.chdir(above)
    if not os.path.exists("test"):
        os.mkdir("test")

    for sdir in os.listdir(train_out_dir):
        cur_dir = os.path.join(train_out_dir, sdir)
        list_curr_dir = os.listdir(cur_dir)
        random.seed()
        rand_num = random.randint(0, len(list_curr_dir) - 1)
        rand_img = list_curr_dir[rand_num]
        rand_img_path = os.path.join(cur_dir, rand_img)
        dst_path = os.path.join("test", sdir)
        if not os.path.exists(dst_path):
            os.mkdir(dst_path)

        shutil.move(rand_img_path, os.path.join(dst_path, os.path.basename(rand_img_path)))


def main(data_set_dir, data_out_dir):
    ck_in_dir = data_set_dir + '/CK/images'

    create_folder_structure(ck_in_dir, data_out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_set_dir", type=str)
    parser.add_argument("--data_out_dir", type=str)

    args = parser.parse_args()
    main(args.data_set_dir, args.data_out_dir)
