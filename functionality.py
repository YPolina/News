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


def tf_idf(text: str, max_features: int = 100):
    """
    tf-idf embeddings

    Parameters:
    text:str - text for df_idf
    max_features:int
    
    Returns:
    tf_idf_vectors: np.array - vectors of words
    tfidf_df - 
    """

    tfidf = TfidfVectorizer(max_features=max_features, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(text)

    tf_idf_vectors = tfidf_matrix.toarray()
    feature_names = tfidf.get_feature_names_out()
    tfidf_df = pd.DataFrame(tf_idf_vectors, columns=feature_names)

    return tf_idf_vectors, tfidf_df

def map_category_to_group(category_groups: dict, category:str) -> str:

        """
        Map category to the bigger one

        Parameters:
        category_groups:dict - categories as keys, subcategories as values
        category: str - mapped category

        Returns:
        groupped category for mapped category

        """

        for group, subcategories in category_groups.items():
            if category in subcategories:
                return group

class EDA:
    """
    Class for EDA task

    """
    def __init__(self, data:pd.DataFrame):
        """
        Class initialization
        data - pd.Dataframe for EDA
        """
        self.data = data

    def plot_class_distribution(self, category_col:str ='category'):

        """
        Plot of class distribution

        Parameters:
        category_col - column with news categories
        """

        plt.figure(figsize=(12, 6))
        category_counts = self.data[category_col].value_counts()
        sns.barplot(x=category_counts.index, y=category_counts.values, palette='viridis', hue = category_counts)
        plt.xticks(rotation=90)
        plt.title('Category Distribution')
        plt.xlabel('Category')
        plt.ylabel('Number of Articles')
        plt.show()

    def categories_clusters(self):
        
        """
        Plot of categories clusters

        Parameters:
        df - pd.Dataframe for EDA
        """
        self.data['processed_text'] = (self.data['headline'] + " " + self.data['short_description']).apply(preprocess_text)

        #Data group by category
        category_data = self.data.groupby('category')['processed_text'].apply(' '.join).reset_index()
        #TF-idf vectors for category_data
        category_vectors, _ = tf_idf(category_data['processed_text'], max_features=5000)

        #Similarity between different categories
        similarity_matrix = cosine_similarity(category_vectors)
        similarity_df = pd.DataFrame(similarity_matrix, index=category_data['category'], columns=category_data['category'])

        dendrogram = sch.dendrogram(sch.linkage(category_vectors, method='ward'), labels=category_data['category'].values)
        plt.title('Dendrogram for Category Merging')
        plt.xlabel('Categories')
        plt.ylabel('Euclidean Distance')
        plt.xticks(rotation=90)
        plt.show()

    def most_common_words_per_big_category(self, category_groups: dict):
        """
        Top 10 words per aggregated category

        Parameters:
        category_groups:dict - categories as keys, subcategories as values
        """
        
        self.data['category_group'] = self.data['category'].apply(
        lambda category: map_category_to_group(category_groups, category)
    )

        for group in category_groups.keys():
            group_text = " ".join(self.data[self.data['category_group'] == group]['processed_text'])
            group_word_counts = Counter(group_text.split()).most_common(10)
            print(f"Most Common Words in {group}:", group_word_counts)

            # Plot the most common words
            words, counts = zip(*group_word_counts)
            plt.figure(figsize=(10, 5))
            plt.bar(words, counts, color='skyblue')
            plt.title(f"Top 10 Words in {group}")
            plt.xlabel("Words")
            plt.ylabel("Frequency")
            plt.xticks(rotation=45)
            plt.show()

    def words_tfidf_per_big_category(self, category_groups: dict):
        """
        Top 10 words per aggregated category

        Parameters:
        category_groups:dict - categories as keys, subcategories as values
        """

        _, tf_idf_df = tf_idf(self.data['processed_text'], max_features=100)

        tfidf_by_category = pd.concat([self.data['category_group'], tf_idf_df], axis=1).groupby('category_group').mean()

        # Display top words for each group
        top_words_per_group = {}
        for group in tfidf_by_category.index:
            top_words = tfidf_by_category.loc[group].sort_values(ascending=False).head(10)
            top_words_per_group[group] = top_words

        top_words_df = pd.DataFrame(top_words_per_group).T

        # Plot the data as a heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            top_words_df, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={"label": "TF-IDF Score"}
        )
        plt.title("Top TF-IDF Words per Aggregated Category", fontsize=16)
        plt.ylabel("Category Group", fontsize=12)
        plt.xlabel("Words", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.show()

    def analyze_length_outliers(self):
        """
        Length outlier analysis in short description and headline

        """

        # Calculate length of headlines and descriptions
        self.data['headline_length'] = self.data['headline'].apply(len)
        self.data['description_length'] = self.data['short_description'].apply(len)


        # Plot distributions
        plt.figure(figsize = (10,6))

        plt.subplot(1,2,1)
        sns.boxplot(self.data['headline_length'], color = 'skyblue')
        plt.title('Headline Length Plot Distribution')
        plt.xlabel('Length (characters)')

        plt.subplot(1,2,2)
        sns.boxplot(self.data['description_length'], color = 'lightgreen')
        plt.title('Short Description Plot Distribution')
        plt.xlabel('Length (characters)')

        plt.tight_layout()
        plt.show()

        # Thresholds for outliers
        short_headline_threshold = 5
        long_headline_threshold = 110
        short_description_threshold = 10
        long_description_threshold = 300

        # Identify long headlines/descriptions
        short_headlines = self.data[self.data['headline_length'] <= short_headline_threshold]
        long_headlines = self.data[self.data['headline_length'] > long_headline_threshold]
        short_descriptions = self.data[self.data['description_length'] <= short_description_threshold]
        long_descriptions = self.data[self.data['description_length'] > long_description_threshold]

        print(f"Number of short headlines (<{short_headline_threshold} chars): {len(short_headlines)}")
        print(f"Number of long headlines (>{long_headline_threshold} chars): {len(long_headlines)}")
        print(f"Number of short descriptions (<={short_description_threshold} chars): {len(short_descriptions)}")
        print(f"Number of long descriptions (>{long_description_threshold} chars): {len(long_descriptions)}")

        return short_headlines, long_headlines, short_descriptions, long_descriptions



        




