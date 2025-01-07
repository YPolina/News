import pandas as pd
import os
import opendatasets as od
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import string
import contractions
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import scipy.cluster.hierarchy as sch
from collections import Counter


nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


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
    file_path:str - .csv/.json/.parquet formats

    Returns:
    dataframe - data in dataframe format if load is successful, else - error notification
    """

    if file_path[-4:] == "json":
        dataframe = pd.read_json(file_path, lines=True)
    elif file_path[-3:] == "csv":
        dataframe = pd.read_csv(file_path, index_col=False)
    elif file_path[-7:] == 'parquet':
        dataframe = pd.read_parquet(file_path)
    else:
        return "Invalid file format. .csv/.json/.parquet expected"
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

def file_saving(df: pd.DataFrame, save_directory: str) -> None:
    """
    Data saving in parquet format

    Parameters:
    df - pd.Dataframe to save
    save_directory - path for saving
    """
    df.to_parquet(save_directory)

def preprocess_text(text:str) -> str:

    """
    Preprocessing text into tokens with cleaning

    Parameters:
    text - text in string format to preprocess
    
    Returns:
    preprocessed_text: str - text after preprocessing
    """
    # Expand contractions
    text = contractions.fix(text)

    # Remove punctuation and normalize quotes
    text = re.sub(r'[“”‘’\'"`]', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))


    # Tokenize and lowercase
    tokens = word_tokenize(text.lower())
    # Remove stopwords, punctuation, and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and word not in string.punctuation]
    preprocessed_text = " ".join(tokens)

    return preprocessed_text


def tf_idf_vectors(text: str):
    """
    tf-idf embeddings

    Parameters:
    text - text for df_idf
    
    Returns:
    category_vectors: np.array - vectors of words
    """

    vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(text)

    # Convert to array for clustering
    category_vectors = tfidf_matrix.toarray()

    return category_vectors

class EDA:
    """
    Class for EDA task

    """
    def __init__(self):
        """
        Class initialization
        """
        self.data = None

    def plot_class_distribution(self, df, category_col:str ='category'):

        """
        Plot of class distribution

        Parameters:
        df - pd.Dataframe for EDA
        category_col - column with news categories
        """

        plt.figure(figsize=(12, 6))
        category_counts = df[category_col].value_counts()
        sns.barplot(x=category_counts.index, y=category_counts.values, palette='viridis', hue = category_counts)
        plt.xticks(rotation=90)
        plt.title('Category Distribution')
        plt.xlabel('Category')
        plt.ylabel('Number of Articles')
        plt.show()

    def categories_clusters(self, df: pd.DataFrame):
        
        """
        Plot of categories clusters

        Parameters:
        df - pd.Dataframe for EDA
        """
        self.data = df
        self.data['processed_text'] = (self.data['headline'] + " " + self.data['short_description']).apply(preprocess_text)

        #Data group by category
        category_data = self.data.groupby('category')['processed_text'].apply(' '.join).reset_index()
        #TF-idf vectors for category_data
        category_vectors = tf_idf_vectors(category_data['processed_text'])

        #Similarity between different categories
        similarity_matrix = cosine_similarity(category_vectors)
        similarity_df = pd.DataFrame(similarity_matrix, index=category_data['category'], columns=category_data['category'])

        dendrogram = sch.dendrogram(sch.linkage(category_vectors, method='ward'), labels=category_data['category'].values)
        plt.title('Dendrogram for Category Merging')
        plt.xlabel('Categories')
        plt.ylabel('Euclidean Distance')
        plt.xticks(rotation=90)
        plt.show()


        




