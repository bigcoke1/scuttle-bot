def main():
    from scuttle_bot.data.dataset import Dataset
    from sklearn.model_selection import train_test_split

    dataset = Dataset(db_path="src/scuttle_bot/cache/ml_dataset.db")
    df = dataset.retrieve_dataset()
    print(df.head())

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

    print(f"Training set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    print(f"Test set size: {len(test_df)}")

    

if __name__ == "__main__":    
    main()