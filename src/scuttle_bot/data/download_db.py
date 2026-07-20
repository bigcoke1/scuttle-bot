import gdown
import os


if __name__ == "__main__":
    file_id = '1oTwXYj5U7fA0mZkokmiZFSuXy0xN7osU'
    url = f'https://drive.google.com/uc?id={file_id}'
    DB_PATH = 'src/scuttle_bot/cache/ml_dataset.db'

    if not os.path.exists(DB_PATH):
        gdown.download(url, DB_PATH, quiet=False)
    else:
        print('DB already exists')