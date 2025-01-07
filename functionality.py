import pandas as pd
import os
import opendatasets as od
import pandas as pd


def extraction(
    dataset_url: str = "https://www.kaggle.com/datasets/rmisra/news-category-dataset/data",
    API_kaggle: str = "kaggle.json",
    data_file: str = "News_Category_Dataset_v3.json",
) -> None:
    """
    Data loading from Kaggle source

    Parameters:
    dataset_url - URL of the Kaggle data
    API_kaggle - Kaggle API token'
    data_file - name of downloaded file
    """
    os.environ["KAGGLE_CONFIG_DIR"] = API_kaggle
    try:
        od.download(dataset_url)
        for root, dirs, files in os.walk(os.getcwd()):
            if data_file in files:
                print("File downloaded to a:", os.path.join(root, data_file))
                break
    except:
        print("Error during loading.")


def loading(file_path: str) -> pd.DataFrame:
    """
    Load file into dataframe

    Parameters:
    file_path:str - .csv or .json format

    Returns:
    dataframe - data in dataframe format if load is successful, else - error notification
    """

    if file_path[-4:] == "json":
        dataframe = pd.read_json(file_path, lines=True)
    elif file_path[-3:] == "csv":
        dataframe = pd.read_csv(file_path, index_col=False)
    else:
        return "Invalid file format. .csv or .json expected"
    return dataframe


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Data Transform step to ensure appropriate data format, absence of nan and duplicated values

    Parameters:
    df - pd.Dataframe with our data

    Returns:
    df - formatted version of our data
    """

    # NAN values
    if (df.isna().sum() <= len(df) * 0.03).any():
        df.dropna(inplace=True)
        print(
            "The number of NaN values is less than or equal to 3%.\nNan values are dropped successfully."
        )
    else:
        df.fillna(0, inplace=True)
        print(
            "The number of NaN values is more than 3%.\nNan values are filled with 0 successfully."
        )

    # Duplicated values
    if df.duplicated().any():
        print(f"The number of duplicated rows is {df.duplicated().sum()/len(df):.3%}.")
        df.drop_duplicates(inplace=True)
        print("Duplicated values are dropped successfully.")

    # Datetime check
    if "date" in df:
        df["date"] = pd.to_datetime(df["date"], errors="raise")
        print('Column "date" converted to datetime.')

    return df
