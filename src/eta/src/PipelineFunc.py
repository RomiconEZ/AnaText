import os

import numpy as np
import pandas as pd

from .ApproxClustNum import approximate_n_clusters
from .BasicFunc import (
    detect_language,
    embeddings_for_df,
    preprocess_text,
    read_dataset,
    rename_and_check_columns,
)
from .Classifier import text_classifier, train_classifier
from .KeywordDetect import get_top_words_for_clusters
from .SCCLBert import (
    initial_operation,
    recalculate_model,
    reduce_dimension_truncated_svd,
    split_cluster,
    union_clusters,
)


def read_dataset_pipeline(filename: str, verbose: bool = False) -> pd.DataFrame:
    """
    Reading the dataset in the main pipeline with exception handling.
    """
    try:
        df = read_dataset(filename, verbose)
        return df
    except Exception as e:
        print("An error occurred at the stage of reading the file:", e)


def preprocess_df_pipeline(df: pd.DataFrame, verbose: bool = False):
    """
    Preprocessing the entire dataset in the main pipeline with exception handling.
    """
    try:
        df.iloc[:, 0] = df.iloc[:, 0].astype(str).map(preprocess_text)
        if verbose:
            print("Preprocessed texts")
    except Exception as e:
        print("An error occurred at the text preprocessing stage:", e)


def save_df_with_emb_pipeline(df, filename, verbose: bool = False):
    try:
        name_without_extension = os.path.splitext(filename)[0]
        new_file_name = name_without_extension + "_with_emb" + ".csv"
        df.to_csv(new_file_name, index=False)
        if verbose:
            print("Saved a dataframe with embeddings")
    except Exception as e:
        print("An error occurred at the stage of saving a dataframe with embeddings:", e)


def embeddings_for_df_pipeline(
    df,
    verbose: bool = False,
):
    """
    Embeddings for the entire dataset in the main pipeline with exception handling
    """
    try:
        batch_size = int(os.getenv("BATCH_SIZE"))
        device_name = os.getenv("DEVICE_NAME")
        chunk_size = int(os.getenv("CHUNK_SIZE"))
        max_length = int(os.getenv("MAX_LENGTH"))
        num_workers = int(os.getenv("NUM_WORKERS"))
        pkl_filename_list = os.getenv("PKL_FILENAME_LIST") or None
        max_token_seq_length = int(os.getenv("MAX_TOKEN_SEQ_LENGTH"))

        if verbose:
            print("Device:", device_name)

        KeyBERTmodel, data, mean_embedding, tokenizer, model = embeddings_for_df(
            df,
            pkl_filename_list,
            chunk_size,
            max_length,
            device_name,
            batch_size,
            num_workers,
            max_token_seq_length,
        )
        if verbose:
            print("Counted the embeddings for the texts")
        return KeyBERTmodel, data, mean_embedding, tokenizer, model
    except Exception as e:
        print("An error occurred during the embedding calculation stage:", e)


def lang_of_df_pipeline(df, verbose: bool = False):
    """
    Language definitions for a dataframe in the main pipeline with exception handling
    """
    try:
        lang = detect_language(df["text"][0])
        if lang is None:
            if verbose:
                print("Undefined language, English will be used by default")
        if verbose:
            print("Determined the language of the text")
        return lang
    except Exception as e:
        print("An error occurred at the language detection stage:", e)


def load_preprocess_embeddings_pipeline(filename=None, verbose: bool = False):
    """
    Function for loading, preprocessing, calculating document embeddings
    in the main pipeline with exception handling
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    input:
        filename - file name for reading data
    output:
        df - dataframe with text, embeddings columns
        KeyBERTmodel - initialized keyBERT model
        lang - text language (ru or eng)
        data - special dataframe for the operation of the SCCL model
        mean_embedding - embedding of texts (at this stage they are transmitted separately, you can get them from df)
        tokenizer
        model - SCCLBert
    """
    # Reading data
    df = read_dataset_pipeline(filename, verbose)
    # Renaming the columns
    rename_and_check_columns(df, verbose)
    # Text preprocessing
    preprocess_df_pipeline(df, verbose)
    # Defining the language of the texts
    lang = lang_of_df_pipeline(df, verbose)
    # Counting embeddings for texts
    KeyBERTmodel, data, mean_embedding, tokenizer, model = embeddings_for_df_pipeline(df, verbose)
    # Saving the dataframe
    save_df_with_emb_pipeline(df, filename, verbose)

    return df, KeyBERTmodel, lang, data, mean_embedding, tokenizer, model


def approximate_n_clusters_pipeline(df, verbose: bool = False):
    """
    Function for calculating the approximate number of clusters
    in the main pipeline with exception handling
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    input:
        df - dataframe with embeddings column (other columns are possible)
    output:
        num_clusters - number of clusters
    """
    try:
        embeddings_array = np.array([*df["embeddings"]])
        num_clusters = approximate_n_clusters(embeddings_array)
        if verbose:
            print("Number of clusters:", num_clusters)
        return num_clusters
    except Exception as e:
        print(
            "An error occurred at the stage of calculating the approximate number of clusters:",
            e,
        )
        return None


def clustering_pipeline(df, num_clusters, embeddings, verbose: bool = False):
    """
    Data clustering in the main pipeline with exception handling
    """
    try:
        # Compute cluster labels and cluster centers
        data = df.copy()
        data, cluster_centers_1, radiuses, cluster_model_1 = initial_operation(
            num_clusters, data, embeddings=embeddings
        )
        # Compute 2d cluster centers
        cluster_centers_2d_1, reduce_model_1 = reduce_dimension_truncated_svd(cluster_centers_1, n_components=2)
        labels = data["current_class"].to_list()

        if df.shape[0] != len(labels):
            raise Exception("Incorrect size of the list of labels for data")

        df["label"] = labels
        if verbose:
            print("Clustered data")
        return (
            df,
            data,
            cluster_centers_1,
            radiuses,
            cluster_model_1,
            cluster_centers_2d_1,
            reduce_model_1,
        )

    except Exception as e:
        print("An error occurred at the clustering stage:", e)


def top_words_pipeline(df, KeyBERTmodel, lang):
    """
    Definition of the top words for clusters
    ...
    Parameters
    ----------
        df - dataframe with text, embeddings, label columns
        KeyBERTmodel - initialized keyBERT model
        lang - text language (ru or eng)
    Returns
        dict: claster_label: [top_words]
    """
    try:
        top_word_dict = get_top_words_for_clusters(df, KeyBERTmodel, lang)
        return top_word_dict
    except Exception as e:
        print("An error occurred at the keyword search stage:", e)


def classifier_model_pipeline(new_texts, data, embeddings_1, model, tokenizer):
    """
    Classifying the list of new tests using a trained random forest model and
    a pre-trained language model.
    """
    try:
        random_forest = train_classifier(data, embeddings_1)
        labels_for_texts = text_classifier(
            new_texts,
            random_forest,
            model,
            tokenizer,
        )
        return labels_for_texts
    except Exception as e:
        print("An error occurred during the classification stage:", e)
        return None


def split_cluster_pipeline(cluster_num, divisor, data, reduce_model, embeddings):
    """
    Splitting cluster with number {cluster_num} into new clusters in the amount of {divisor}.
    """

    try:
        # Counting the number of documents in the cluster
        count = data[data["current_class"] == cluster_num].shape[0]
        # If there is only one element in the cluster, then it cannot be divided
        if count == 1:
            print("You cannot split a cluster containing a single element")
            return data, None, None
        # Splitting the cluster
        data, new_cluster_centers_2d_1, new_radiuses_1 = split_cluster(
            cluster_num=cluster_num,
            divisor=divisor,
            data=data,
            reduce_model=reduce_model,
            embeddings=embeddings,
        )
        return data, new_cluster_centers_2d_1, new_radiuses_1
    except Exception as e:
        print("An error occurred at the cluster separation stage:", e)
        return data, None, None


def union_clusters_pipeline(cl_list, data, reduce_model, embeddings):
    """
    Merging clusters with numbers from cl_list.
    """
    try:
        data, new_cluster_centers_2d_2, new_radiuses_2 = union_clusters(
            cl_list=cl_list,
            data=data,
            reduce_model=reduce_model,
            embeddings=embeddings,
        )
        return data, new_cluster_centers_2d_2, new_radiuses_2
    except Exception as e:
        print("An error occurred at the cluster merging stage:", e)
        return data, None, None


def recalculate_model_pipeline(data, embeddings, tokenizer, model):
    """
    Recalculation of the model based on new clusters obtained during the union and splitting of clusters.
    Recalculation is possible only on cuda.
    """
    try:
        batch_size = int(os.getenv("BATCH_SIZE"))
        device_name = os.getenv("DEVICE_NAME")
        lr = float(os.getenv("LR"))
        max_length = int(os.getenv("MAX_LENGTH"))
        num_workers = int(os.getenv("NUM_WORKERS"))
        augmenter_pkl_name = os.getenv("AUGMENTER_PKL_NAME") or None
        lr_scale = int(os.getenv("LR_SCALE"))
        temperature = float(os.getenv("TEMPERATURE"))
        max_iter_1 = int(os.getenv("MAX_ITER_1"))
        eta = int(os.getenv("ETA"))

        # Recalculating the model
        (
            data,
            cluster_centers_2,
            radiuses_2,
            cluster_model_2,
            embeddings_2,
        ) = recalculate_model(
            data,
            embeddings,
            tokenizer,
            model,
            batch_size,
            device_name,
            lr,
            max_length,
            num_workers,
            augmenter_pkl_name,
            lr_scale,
            temperature,
            max_iter_1,
            eta,
        )

        return data, cluster_centers_2, radiuses_2, cluster_model_2, embeddings_2
    except Exception as e:
        print("An error occurred at the recalculation stage of the model:", e)
        return data, None, None, None, embeddings
