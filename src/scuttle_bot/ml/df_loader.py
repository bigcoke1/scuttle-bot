from scuttle_bot.data.dataset import Dataset
import pandas as pd


class Dataframe_Loader():
    def __init__(self, dataset: Dataset):
        self.df = dataset.retrieve_dataset()
        pass

if __name__ == "__main__":
    dataset = Dataset(db_path="src/scuttle_bot/cache/ml_dataset.db")
    loader = Dataframe_Loader(dataset)
    print(loader.df.head())