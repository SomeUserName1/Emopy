import argparse
import os

from config import SESSION
from runners import start_train_program, run_test
from train_config import DATA_SET_DIR, EPOCHS, LEARNING_RATE, STEPS_PER_EPOCH, BATCH_SIZE, AUGMENTATION


def main():
    """

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_set_dir", default=DATA_SET_DIR, type=str)
    parser.add_argument("--train", default=(SESSION == "train"), type=bool)
    parser.add_argument("--epochs", default=EPOCHS, type=int)
    parser.add_argument("--batch_size", default=BATCH_SIZE, type=int)
    parser.add_argument("--lr", default=LEARNING_RATE, type=float)
    parser.add_argument("--steps", default=STEPS_PER_EPOCH, type=int)
    parser.add_argument("--augmentation", default=AUGMENTATION, type=bool)

    args = parser.parse_args()
    if not os.path.exists(args.data_set_dir):
        print("Data set path given does not exists")
        exit(0)

    if args.train == (SESSION == "train"):
        start_train_program(dataset_dir=args.data_set_dir, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                            steps=args.steps, augmentation=AUGMENTATION)
    else:
        run_test()


if __name__ == "__main__":
    main()
