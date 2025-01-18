import os
import gc
import re
import nltk
import torch
import emoji
import string
import neptune
import logging
import numpy as np
import contractions
import pandas as pd
from time import time
import seaborn as sns
from tqdm import tqdm
import torch.nn as nn
import opendatasets as od
import torch.nn.functional as F
from collections import Counter
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from gensim.models import Phrases
import scipy.cluster.hierarchy as sch
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from neptune_pytorch import NeptuneLogger
from gensim.models.phrases import Phraser
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_class_weight
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup


nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("stopwords")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def preprocess_text(text: str) -> str:
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
    text = re.sub(r'[“”‘’\'"`]', "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Tokenize and lowercase
    tokens = word_tokenize(text.lower())
    # Remove stopwords, punctuation, and lemmatize
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and word not in string.punctuation
    ]
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
    tfidf_df: pd.DataFrame - Dataframe with vectors of words and there names
    """

    tfidf = TfidfVectorizer(max_features=max_features, stop_words="english")
    tfidf_matrix = tfidf.fit_transform(text)

    tf_idf_vectors = tfidf_matrix.toarray()
    feature_names = tfidf.get_feature_names_out()
    tfidf_df = pd.DataFrame(tf_idf_vectors, columns=feature_names)

    return tf_idf_vectors, tfidf_df


def map_category_to_group(category_groups: dict, category: str) -> str:
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

    def __init__(self, data: pd.DataFrame):
        """
        Class initialization
        data - pd.Dataframe for EDA
        """
        self.data = data

    def plot_class_distribution(self, category_col: str = "category"):
        """
        Plot of class distribution

        Parameters:
        category_col - column with news categories
        """

        plt.figure(figsize=(12, 6))
        category_counts = self.data[category_col].value_counts()
        sns.barplot(
            x=category_counts.index,
            y=category_counts.values,
            palette="viridis",
            hue=category_counts,
        )
        plt.xticks(rotation=90)
        plt.title("Category Distribution")
        plt.xlabel("Category")
        plt.ylabel("Number of Articles")
        plt.show()

    def categories_clusters(self):
        """
        Plot of categories clusters

        Parameters:
        df - pd.Dataframe for EDA
        """
        self.data["processed_text"] = (
            self.data["headline"] + " " + self.data["short_description"]
        ).apply(preprocess_text)

        # Data group by category
        category_data = (
            self.data.groupby("category")["processed_text"]
            .apply(" ".join)
            .reset_index()
        )
        # TF-idf vectors for category_data
        category_vectors, _ = tf_idf(category_data["processed_text"], max_features=5000)

        # Similarity between different categories
        similarity_matrix = cosine_similarity(category_vectors)
        similarity_df = pd.DataFrame(
            similarity_matrix,
            index=category_data["category"],
            columns=category_data["category"],
        )

        dendrogram = sch.dendrogram(
            sch.linkage(category_vectors, method="ward"),
            labels=category_data["category"].values,
        )
        plt.title("Dendrogram for Category Merging")
        plt.xlabel("Categories")
        plt.ylabel("Euclidean Distance")
        plt.xticks(rotation=90)
        plt.show()

    def most_common_words_per_big_category(self, category_groups: dict):
        """
        Top 10 words per aggregated category

        Parameters:
        category_groups:dict - categories as keys, subcategories as values
        """

        self.data["category_group"] = self.data["category"].apply(
            lambda category: map_category_to_group(category_groups, category)
        )

        for group in category_groups.keys():
            group_text = " ".join(
                self.data[self.data["category_group"] == group]["processed_text"]
            )
            group_word_counts = Counter(group_text.split()).most_common(10)
            print(f"Most Common Words in {group}:", group_word_counts)

            # Plot the most common words
            words, counts = zip(*group_word_counts)
            plt.figure(figsize=(10, 5))
            plt.bar(words, counts, color="skyblue")
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

        _, tf_idf_df = tf_idf(self.data["processed_text"], max_features=100)

        tfidf_by_category = (
            pd.concat([self.data["category_group"], tf_idf_df], axis=1)
            .groupby("category_group")
            .mean()
        )

        # Display top words for each group
        top_words_per_group = {}
        for group in tfidf_by_category.index:
            top_words = (
                tfidf_by_category.loc[group].sort_values(ascending=False).head(10)
            )
            top_words_per_group[group] = top_words

        top_words_df = pd.DataFrame(top_words_per_group).T

        # Plot the data as a heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            top_words_df,
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
            cbar_kws={"label": "TF-IDF Score"},
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
        self.data["headline_length"] = self.data["headline"].apply(len)
        self.data["description_length"] = self.data["short_description"].apply(len)

        # Plot distributions
        plt.figure(figsize=(10, 6))

        plt.subplot(1, 2, 1)
        sns.boxplot(self.data["headline_length"], color="skyblue")
        plt.title("Headline Length Plot Distribution")
        plt.xlabel("Length (characters)")

        plt.subplot(1, 2, 2)
        sns.boxplot(self.data["description_length"], color="lightgreen")
        plt.title("Short Description Plot Distribution")
        plt.xlabel("Length (characters)")

        plt.tight_layout()
        plt.show()

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

        print(
            f"Number of short headlines (<{short_headline_threshold} chars): {len(short_headlines)}"
        )
        print(
            f"Number of long headlines (>{long_headline_threshold} chars): {len(long_headlines)}"
        )
        print(
            f"Number of short descriptions (<={short_description_threshold} chars): {len(short_descriptions)}"
        )
        print(
            f"Number of long descriptions (>{long_description_threshold} chars): {len(long_descriptions)}"
        )

        return short_headlines, long_headlines, short_descriptions, long_descriptions

    def analyze_date_trends(self):
        """
        Time trend analysis of the overall data and category groups

        """

        self.data["month_year"] = self.data["date"].dt.to_period("M")
        trends = self.data.groupby("month_year").size()

        plt.figure(figsize=(12, 6))
        trends.plot(kind="line", color="skyblue", marker="o")
        plt.title("Articles Volume over time")
        plt.xlabel("Month-Year")
        plt.ylabel("Number of Articles")
        plt.grid(True)
        plt.show()

        # Trends by category_group
        category_trends = (
            self.data.groupby(["month_year", "category_group"])
            .size()
            .unstack(fill_value=0)
        )

        plt.figure(figsize=(14, 8))
        category_trends.plot(figsize=(14, 8), kind="line", marker="o", colormap="tab10")
        plt.title("Articles Volume over time by Category Group")
        plt.xlabel("Month-Year")
        plt.ylabel("Number of Articles")
        plt.legend(title="Category Group", bbox_to_anchor=(1.05, 1), loc="upper left")
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
            "QUEER VOICES": "GROUPS VOICES",
            "LATINO VOICES": "GROUPS VOICES",
            "BLACK VOICES": "GROUPS VOICES",
            "SCIENCE": "SCIENCE & TECH",
            "TECH": "SCIENCE & TECH",
            "MONEY": "BUSINESS & FINANCES",
            "COLLEGE": "EDUCATION",
            "FIFTY": "MISCELLANEOUS",
            "GOOD NEWS": "MISCELLANEOUS",
        }
        self.data.replace(replacements, inplace=True)
        label_encoder = LabelEncoder()
        self.data["category_encoded"] = label_encoder.fit_transform(
            self.data["category"]
        )

        # encoder = LabelEncoder()
        # encoder.fit_transform(self.data['category'])

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
        self.data = self.data[~self.data["headline"].isin(long_headlines)].reset_index(
            drop=True
        )

        self.data = self.data[
            ~self.data["short_description"].isin(long_descriptions)
        ].reset_index(drop=True)

        self.data["processed_text"] = (
            self.data["headline"] + " " + self.data["short_description"]
        )

        # Added data for date trends
        self.data["month_year"] = self.data["date"].dt.to_period("M")

        self.data.to_parquet("./data/preprocessed_data.parquet")

        return self.data


class CustomTextDataset(Dataset):
    """
    Class Dataset

    """

    def __init__(self, data, labels, tokenizer, max_length):
        """
        Class initialization

        """
        self.data = data
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """
        returns the length of the labels

        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        get a sample and construct the dataset

        """
        text = self.data.iloc[idx]
        label = self.labels.iloc[idx]

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=True,
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        label = torch.tensor(label, dtype=torch.long)

        return input_ids, attention_mask, label


def sampler(data: pd.DataFrame, target: str = "category_encoded"):
    """
    Weighted Sampling

    Parameters:
    data: pd.DataFrame - data that need to be
    target: str - name of column with labels

    Returns:
    sampler - class instance that samples elements from [0,..,len(weights)-1] with given probabilities

    """

    category_counts = Counter(data[target])
    num_samples = len(data)
    class_weights = {
        label: num_samples / count for label, count in category_counts.items()
    }
    weights = data[target].map(class_weights).values

    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    return sampler

def extract_embeddings(bert_model, dataloader, device):

    bert_model.eval()
    bert_model.to(device) 

    embeddings = []
    labels = []

    with torch.no_grad():  
        for input_ids, attention_mask, batch_labels in dataloader:
            
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            batch_labels = batch_labels.to(device)
            
            
            outputs = bert_model(input_ids, attention_mask=attention_mask)
            
            cls_embeddings = outputs.last_hidden_state[:, 0, :]

            # Append to lists
            embeddings.append(cls_embeddings.cpu()) 
            labels.append(batch_labels.cpu())
    
    
    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels, dim=0)

    return embeddings, labels


def train_epoch(
    model,
    bert_model,
    dataloader,
    optimizer,
    criterion,
    device,
    use_bert_embeddings=True,
):
    """
    Training for 1 epoch

    Parameters:
    model - model that will be trained
    bert_model - pre-trained model for embeddings
    dataloader - pre-defined dataloader to train data in batches
    optimizer - optimizer for parameters update
    criterion - Loss function
    device - the device type responsible to load a tensor into memory
    use_bert_embeddings - wether to use BERT embeddings

    Returns:
    avg_loss - average loss per epoch
    f1 - F1-score

    """

    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for input_ids, attention_mask, labels in tqdm(dataloader, desc="Training Epoch"):
        optimizer.zero_grad()

        # Get the input tensors and labels from the batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        if use_bert_embeddings:
            with torch.no_grad():
                outputs = bert_model(input_ids, attention_mask=attention_mask)
                token_embeddings = outputs.last_hidden_state
            model_outputs = model(token_embeddings)
        else:
            model_outputs = model(
                input_ids=input_ids, attention_mask=attention_mask
            )

        # Compute the loss
        loss = criterion(model_outputs, labels)
        total_loss += loss.item()

        # Backward pass
        loss.backward()
        optimizer.step()

        # Get predictions
        preds = torch.argmax(model_outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average="macro")
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, f1, accuracy


@torch.no_grad()
def eval(
    model,
    bert_model,
    dataloader,
    optimizer,
    criterion,
    device,
    use_bert_embeddings=True,
):
    """
    Training for 1 epoch

    Parameters:
    model - model that will be trained
    bert_model - pre-trained model for embeddings
    dataloader - pre-defined dataloader to train data in batches
    optimizer - optimizer for parameters update
    criterion - Loss function
    device - the device type responsible to load a tensor into memory
    use_bert_embeddings - wether to use BERT embeddings

    Returns:
    avg_loss - average loss per epoch
    f1 - F1-score

    """
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    for input_ids, attention_mask, labels in tqdm(dataloader, desc="Evaluation"):

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        # Forward pass: Get model outputs (logits)
        if use_bert_embeddings:
            outputs = bert_model(input_ids, attention_mask=attention_mask)
            token_embeddings = outputs.last_hidden_state
            model_outputs = model(token_embeddings)
        else:
            model_outputs = model(
                input_ids=input_ids, attention_mask=attention_mask
            )


        # Compute the loss
        loss = criterion(model_outputs, labels)
        total_loss += loss.item()

        # Get predictions
        preds = torch.argmax(model_outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, average="macro")
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, f1, accuracy


def train_model_with_callbacks(
    model,
    bert_model,
    train_loader,
    val_loader,
    epochs,
    optimizer,
    criterion,
    device,
    callbacks,
    use_bert_embeddings=True,
):
    """
    Model Training

    Parameters:

    Parameters:
    model - model that will be trained
    bert_model - pre-trained model for embeddings
    train_loader&val_loader - pre-defined dataloader to process data in batches
    epochs - the total number of iterations
    optimizer - optimizer for parameters update
    criterion - Loss function
    device - the device type responsible to load a tensor into memory
    callbacks - callback system
    use_bert_embeddings - wether to use BERT embeddings

    """
    training_logs = []

    model.to(device)
    bert_model.to(device)

    for callback in callbacks:
        callback.on_train_begin()

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):

        for callback in callbacks:
            callback.on_epoch_begin(epoch)

        logs = {"epoch": epoch + 1, "model": model}

        print(f"Epoch {epoch + 1}/{epochs}")

        train_loss, train_f1, train_acc = train_epoch(
            model, bert_model, train_loader, optimizer, criterion, device, use_bert_embeddings
        )
        logs["train_loss"] = train_loss
        logs["train_f1"] = train_f1
        logs["train_accuracy"] = train_acc

        print(f"Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}")

        if (epoch + 1) % 2 == 0:

            val_loss, val_f1, val_accuracy = eval(
                model, bert_model, val_loader, optimizer, criterion, device, use_bert_embeddings
            )
            logs["val_loss"] = val_loss
            logs["val_f1"] = val_f1
            logs["val_accuracy"] = val_accuracy
            print(f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}, Val accuracy: {val_accuracy:.4f}")

            for callback in callbacks:
                callback.on_epoch_end(epoch, logs)

            if any(getattr(callback, "stop_training", False) for callback in callbacks):
                break

        torch.cuda.empty_cache()
        gc.collect()

    for callback in callbacks:
        callback.on_train_end()

    return model





class Callback:
    """
    Base Callback Class

    """

    def __init__(self, log_interval: int = 10):
        """
        Class initialization

        Parameters:
        log_interval - he number of training iterations before logging

        """
        self.log_interval = log_interval
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    def on_epoch_begin(self, epoch: int):
        self.epoch_start_time = time()
        logging.info(f"Epoch {epoch+1} starting.")

    def on_epoch_end(self, epoch: int, logs=None):
        elapsed_time = time() - self.epoch_start_time
        logging.info(f"Epoch {epoch + 1} finished in {elapsed_time:.2f} seconds.")
        logs["epoch_time"] = elapsed_time
        training_logs.append(logs)

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass


class EarlyStoppingCallBack(Callback):

    def __init__(self, patience: int = 3):
        self.patience = patience
        self.best_loss = float("inf")
        self.patience_counter = 0
        self.stop_training = False

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get("val_loss")
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                print("Early stopping triggered!")
                self.stop_training = True


class ModelCheckpointCallback(Callback):

    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path
        self.best_loss = float("inf")

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get("val_loss")
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            torch.save(logs.get("model").state_dict(), self.checkpoint_path)
            print(f"Model saved at epoch {epoch + 1}")


class LoggingCallBack(Callback):

    def __init__(self, run):
        self.run = run

    def on_epoch_end(self, epoch, logs=None):
        for key, value in logs.items():
            if key != "model":
                self.run[f"{key}"].log(value)


class RNNClassifier(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RNNClassifier, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTM, self).__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out


class AttentionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(AttentionLSTM, self).__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.attention = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        out = self.fc(context_vector)
        return out


class BERT(nn.Module):

    def __init__(self, bert, num_classes = 27, dropout_rate=0.3):
        super(BERT, self).__init__()
        
        self.bert = bert 

        self.dropout = nn.Dropout(p = dropout_rate)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)


    def forward(self, input_ids, attention_mask):
        
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.fc(x)
        return logits

    def unfreeze(self,start_layer,end_layer):
        def children(m):
            return m if isinstance(m, (list, tuple)) else list(m.children())
        def set_trainable_attr(m, b):
            m.trainable = b
            for p in m.parameters():
                p.requires_grad = b
        def apply_leaf(m, f):
            c = children(m)
            if isinstance(m, nn.Module):
                f(m)
            if len(c) > 0:
                for l in c:
                    apply_leaf(l, f)
        def set_trainable(l, b):
            apply_leaf(l, lambda m: set_trainable_attr(m, b))
        
        set_trainable(self.bert, False)
        set_trainable(model.bert.encoder.layer[23], True)
        for i in range(start_layer, end_layer+1):
            set_trainable(self.bert.encoder.layer[i], True)

class BERT_bigger(nn.Module):

    def __init__(self, bert, num_classes = 27, dropout_rate=0.3):
        super(BERT_bigger, self).__init__()
        
        self.bert = bert 

        self.fc1 = nn.Linear(self.bert.config.hidden_size, 256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout_rate)

        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout_rate)

        self.fc3 = nn.Linear(128, num_classes)


    def forward(self, input_ids, attention_mask):
        
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.fc1(pooled_output)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        logits = self.fc3(x)
        return logits

    def unfreeze(self,start_layer,end_layer):
        def children(m):
            return m if isinstance(m, (list, tuple)) else list(m.children())
        def set_trainable_attr(m, b):
            m.trainable = b
            for p in m.parameters():
                p.requires_grad = b
        def apply_leaf(m, f):
            c = children(m)
            if isinstance(m, nn.Module):
                f(m)
            if len(c) > 0:
                for l in c:
                    apply_leaf(l, f)
        def set_trainable(l, b):
            apply_leaf(l, lambda m: set_trainable_attr(m, b))

        set_trainable(model.bert.encoder.layer[23], True)
        for i in range(start_layer, end_layer+1):
            set_trainable(self.bert.encoder.layer[i], True)

def optuna_objective(trial, train_loader, val_loader):

    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bert_identifiers = ['embedding', 'encoder', 'pooler']
    no_weight_decay_identifiers = ['bias', 'LayerNorm.weight']

    BETAS = (0.9, 0.999)
    EPS = 1e-8
    EPOCHS = 1
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    model = BERT(bert_model, num_classes=27, dropout_rate=dropout_rate).to(device)


    grouped_model_parameters = [

        {'params': [param for name, param in model.named_parameters()
                    if any(identifier in name for identifier in bert_identifiers) and
                    not any(identifier_ in name for identifier_ in no_weight_decay_identifiers)],
         'lr': learning_rate,
         'betas': BETAS,
         'weight_decay': weight_decay,
         'eps': EPS},

        {'params': [param for name, param in model.named_parameters()
                    if any(identifier in name for identifier in bert_identifiers) and
                    any(identifier_ in name for identifier_ in no_weight_decay_identifiers)],
         'lr': learning_rate,
         'betas': BETAS,
         'weight_decay': 0.0,
         'eps': EPS},

        {'params': [param for name, param in model.named_parameters()
                    if not any(identifier in name for identifier in bert_identifiers)],
         'lr': learning_rate,
         'betas': BETAS,
         'weight_decay': 0.0,
         'eps': EPS}
    ]

    optimizer = torch.optim.AdamW(grouped_model_parameters)

    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )


    criterion = nn.CrossEntropyLoss().to(device)

    run = neptune.init_run(
        project="yatskopolina1/News",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiNTYzMDJkMi00YThkLTRiYWYtOGU5ZC02MGFiOGEzNjkzYTIifQ==",
    )

    callbacks = [
        ModelCheckpointCallback(checkpoint_path="model/best_model.pth"),
        EarlyStoppingCallBack(patience=3),
        LoggingCallBack(run=run)
    ]

    trained_model = train_model_with_callbacks(
        model=model,
        bert_model=bert_model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        callbacks=callbacks,
        use_bert_embeddings=False
    )

    val_loss, val_f1, val_accuracy = eval(
        trained_model, bert_model, val_loader, optimizer, criterion, device, use_bert_embeddings=False
    )
    

    return val_loss

