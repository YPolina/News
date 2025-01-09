import pandas as pd
import os
import opendatasets as od
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
import emoji
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import string
import contractions
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import scipy.cluster.hierarchy as sch
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter



nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


class ETL:
    """
    Class for ETL

    """

    def __init__(
        self,
        dataset_url: str = "https://www.kaggle.com/datasets/rmisra/news-category-dataset/data",
        API_kaggle: str = "kaggle.json",
        data_file: str = "News_Category_Dataset_v3.json",
        save_directory: str = "./data/etl_data.parquet",
    ):
        """
        Class initialization

        Parameters:
        dataset_url - URL of the Kaggle data
        API_kaggle - Kaggle API token'
        data_file - name of downloaded file
        save_directory - path for saving
        """
        self.dataset_url = dataset_url
        self.API_kaggle = API_kaggle
        self.data_file = data_file
        self.save_directory = save_directory
        self.data = None

    def extraction(self) -> None:
        """
        Data loading from Kaggle source
        Downloads the dataset using the Kaggle API

        """
        os.environ["KAGGLE_CONFIG_DIR"] = self.API_kaggle
        try:
            od.download(self.dataset_url)

            # Check if file exists after download
            for root, dirs, files in os.walk(os.getcwd()):
                if self.data_file in files:
                    print(f"File downloaded: {os.path.join(root, self.data_file)}")
                    break
        except Exception as e:
            print(f"Error during loading: {e}")

    def loading(
        self, file_path: str = "news-category-dataset\\News_Category_Dataset_v3.json"
    ) -> pd.DataFrame:
        """
        Load file into dataframe (supports .csv, .json, .parquet formats).

        Parameters:
        file_path: str - Path to the file (e.g., .csv/.json/.parquet)

        Returns:
        dataframe: pd.DataFrame - Loaded data in a dataframe format
        """
        if file_path.endswith(".json"):
            dataframe = pd.read_json(file_path, lines=True)
        elif file_path.endswith(".csv"):
            dataframe = pd.read_csv(file_path, index_col=False)
        elif file_path.endswith(".parquet"):
            dataframe = pd.read_parquet(file_path)
        else:
            raise ValueError("Invalid file format. Expected .csv, .json, or .parquet.")
        self.data = dataframe

        return self.data

    def transform(self) -> pd.DataFrame:
        """
        Data Transform step to ensure appropriate data format, absence of NaN, and duplicated values

        Returns:
        self.data: pd.DataFrame - Transformed data
        """
        # Handle NaN values
        if (self.data.isna().sum() <= len(self.data) * 0.03).any():
            self.data.dropna(inplace=True)
            print("Dropped NaN values (less than or equal to 3%).")
        else:
            self.data.fillna(0, inplace=True)
            print("Filled NaN values (more than 3%) with 0.")

        # Handle duplicated rows
        if self.data.duplicated().any():
            print(
                f"Duplicated rows: {self.data.duplicated().sum() / len(self.data):.3%}. Dropping duplicates."
            )
            self.data.drop_duplicates(inplace=True)

        # Convert 'date' column to datetime if it exists
        if "date" in self.data:
            self.data["date"] = pd.to_datetime(self.data["date"], errors="raise")
            print('Converted "date" column to datetime format.')

        return self.data

    def file_saving(self, df: pd.DataFrame) -> None:
        """
        Save the dataframe to a parquet file.

        Parameters:
        df: pd.DataFrame - The dataframe to save
        """
        df.to_parquet(self.save_directory)
        print(f"Data saved to: {self.save_directory}")

    def run_etl(self) -> None:
        """
        Run the entire ETL process: extraction, loading, transforming, and saving the data.
        """
        # Step 1: Data Extraction
        self.extraction()

        # Step 2: Data Loading
        df = self.loading()

        # Step 3: Data Transformation
        df = self.transform()

        # Step 4: Save the Transformed Data
        self.file_saving(df)

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

    def analyze_date_trends(self):

        """
        Time trend analysis of the overall data and category groups 

        """

        self.data['month_year'] = self.data['date'].dt.to_period('M')
        trends = self.data.groupby('month_year').size()

        plt.figure(figsize = (12,6))
        trends.plot(kind = 'line', color = 'skyblue', marker = 'o')
        plt.title('Articles Volume over time')
        plt.xlabel('Month-Year')
        plt.ylabel('Number of Articles')
        plt.grid(True)
        plt.show()

        #Trends by category_group
        category_trends = self.data.groupby(['month_year', 'category_group']).size().unstack(fill_value=0)
        
        plt.figure(figsize = (14,8))
        category_trends.plot(figsize = (14,8), kind = 'line', marker='o', colormap='tab10')
        plt.title('Articles Volume over time by Category Group')
        plt.xlabel('Month-Year')
        plt.ylabel('Number of Articles')
        plt.legend(title = 'Category Group', bbox_to_anchor = (1.05, 1), loc = 'upper left')
        plt.tight_layout()
        plt.grid(True)
        plt.show()

class Data_Preparation:
    """
    Class for data cleaning(based on EDA) and text preprocessing
    """

    def __init__(self, data):
        """
        Class initialization

        Parameters:
        data: pd.DataFrame - data for preprocessing
        """
        self.data = data

    def text_preprocessing(self, text):
        """
        Text Preprocessing

        """
        text = contractions.fix(text)

        # Remove punctuation and normalize quotes
        text = re.sub(r'[“”‘’\'"`]', "", text)
        text = text.translate(str.maketrans("", "", string.punctuation))

        text = emoji.replace_emoji(text, replace="")

        # Tokenize and lowercase
        tokens = word_tokenize(text.lower())

        # Stop words list
        stop_words = set(stopwords.words("english"))
        custom_stop_words = [
            "day",
            "life",
            "new",
            "people",
            "like",
            "make",
            "year",
            "world",
            "woman",
            "time",
            "say",
            "said",
        ]
        stop_words.update(custom_stop_words)

        # Remove stopwords, punctuation, and lemmatize
        tokens = [
            lemmatizer.lemmatize(word)
            for word in tokens
            if word not in stop_words and word not in string.punctuation
        ]

        # Bigram detection (To handle "Donald Trump", "video games")
        corpus = [tokens]

        bigram_model = Phrases(corpus, min_count=1, threshold=2)
        bigram_phraser = Phraser(bigram_model)

        bigram_tokens = bigram_phraser[tokens]

        preprocessed_text = " ".join(bigram_tokens)

        return preprocessed_text

    def data_transformation(self):
        """
        Data transformation based on ETL, EDA and text_preprocessing

        """

        # Replcement of redundant categories
        replacements = {
            "WORLDPOST": "WORLD NEWS",
            "THE WORLDPOST": "WORLD NEWS",
            "COMEDY": "ENTERTAINMENT",
            "PARENTS": "PARENTING",
            "HEALTHY LIVING": "WELLNESS",
            "GREEN": "ENVIRONMENT",
            "ARTS & CULTURE": "CULTURE & ARTS",
            "ARTS": "CULTURE & ARTS",
            "TASTE": "FOOD & DRINK",
            "STYLE": "STYLE & BEAUTY",
        }
        self.data.replace(replacements, inplace=True)

        self.data["headline"] = self.data["headline"].apply(self.text_preprocessing)
        self.data["short_description"] = self.data["short_description"].apply(
            self.text_preprocessing
        )

        # Remove length outliers
        self.data["headline_length"] = self.data["headline"].apply(len)
        self.data["description_length"] = self.data["short_description"].apply(len)

        # Thresholds for outliers
        short_headline_threshold = 5
        long_headline_threshold = 110
        short_description_threshold = 10
        long_description_threshold = 300

        # Identify long headlines/descriptions
        short_headlines = self.data[
            self.data["headline_length"] <= short_headline_threshold
        ]
        long_headlines = self.data[
            self.data["headline_length"] > long_headline_threshold
        ]
        short_descriptions = self.data[
            self.data["description_length"] <= short_description_threshold
        ]
        long_descriptions = self.data[
            self.data["description_length"] > long_description_threshold
        ]

        # Drop length outliers
        self.data = self.data[
            ~self.data["headline"].isin(short_headlines)
            & ~self.data["short_description"].isin(short_descriptions)
        ].reset_index(drop=True)
        self.data = self.data[
            ~self.data["headline"].isin(long_headlines)
            | ~self.data["short_description"].isin(long_descriptions)
        ].reset_index(drop=True)

        self.data["processed_text"] = (
            self.data["headline"] + " " + self.data["short_description"]
        )
        
        # Added data for date trends
        self.data["month_year"] = self.data["date"].dt.to_period("M")

        self.data.to_parquet('./data/preprocessed_data.parquet')

        return self.data

    
    


        




