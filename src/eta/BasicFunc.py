import json
import pickle
import re
from pathlib import Path

import keybert
import numpy as np
import pandas as pd
import torch
from langdetect import detect
from sentence_transformers import SentenceTransformer

from .KeywordDetect import CustomEmbedder
from .SCCLBert import SCCLBert, evaluate_embeddings, token_loader


def read_dataset(filename: str, verbose: bool = False) -> pd.DataFrame:
    """
    Conversion of data from the presented formats to pandas dataframe.
    """
    dataset_extension = filename.split(".")[-1]
    df = None
    if dataset_extension == "json":
        # Uploading a JSON file
        try:
            if filename is not None:
                with open(filename) as f:
                    json_data = json.load(f)
        except FileNotFoundError:
            print("File not found")
        except json.JSONDecodeError:
            print("JSON decoding error")
        except Exception as e:
            print("There was an error:", e)
        # Reading a JSON file and converting to a dataframe
        try:
            if json_data is not None:
                # Loading data from JSON and creating a DataFrame
                df = pd.read_json(json_data)
        except Exception as e:
            print("Error while reading JSON file:", e)

    elif dataset_extension == "xlsx":
        # Reading excel file
        try:
            df = pd.read_excel(filename)
        except Exception as e:
            print("Error while reading excel file:", e)
    elif dataset_extension == "parquet":
        # Reading parquet file
        try:
            df = pd.read_parquet(filename, engine="pyarrow")
        except Exception as e:
            print("Error while reading parquet file:", e)
    elif dataset_extension == "csv":
        # Reading parquet file
        try:
            df = pd.read_csv(filename)
        except Exception as e:
            print("Error while reading csv file:", e)
    elif dataset_extension == "npy":
        # Reading npy file
        try:
            df = pd.DataFrame(np.load(filename))
        except Exception as e:
            print("An error occurred while reading npy file:", e)
    else:
        raise Exception("Unknown file format. Available formats: [csc,json,xlsx,parquet,npy]")
    if verbose:
        print("Read and download the data")
    return df


def split_document(tokens, mask, chunk_size):
    """
    Splitting the document into consecutive parts of tokens to calculate the total embedding
    """
    chunks = []
    chunks_mask = []
    # The number of tokens in the document
    num_of_tokens = tokens.shape[1]
    # Creating a list of token lists of chunk_size length
    for i in range(0, (num_of_tokens // chunk_size) * chunk_size, chunk_size):
        chunk = tokens[0, i: i + chunk_size]

        chunks.append(chunk)

        chunk_mask = mask[0, i: i + chunk_size]

        chunks_mask.append(chunk_mask)
    # If the number of tokens in the document is not a multiple of chunk_size
    if num_of_tokens % chunk_size != 0:
        last_chunk = tokens[0, (num_of_tokens // chunk_size) * chunk_size:]
        # Filling the last chunk to chunk_size length
        last_chunk = torch.nn.functional.pad(
            last_chunk,
            pad=(0, chunk_size - last_chunk.shape[0]),
            mode="constant",
            value=0,
        )
        chunks.append(last_chunk)
        # Creating a mask for the last chunk
        last_chunk_mask = mask[0, (num_of_tokens // chunk_size) * chunk_size:]
        last_chunk_mask = torch.nn.functional.pad(
            last_chunk_mask,
            pad=(0, chunk_size - last_chunk_mask.shape[0]),
            mode="constant",
            value=0,
        )
        chunks_mask.append(last_chunk_mask)
    return chunks, chunks_mask


def detect_language(text):
    """
    Determining the language of the text (Russian or English) using the langdetect library
    """
    lang = detect(text)
    if lang == "ru":
        return "ru"
    elif lang == "en":
        return "eng"
    else:
        return None


def preprocess_text(doc):
    """
    Text preprocessing: reduction to lowercase, removal of special characters, removal of extra spaces.
    """
    # lower case
    doc = doc.lower()
    # remove special simbols
    for p in ["«", "»", "'", '"', "\n"]:
        doc = doc.replace(p, " ")
    # removing extra spaces
    doc = re.sub(r"\s+", " ", doc).strip()
    return doc


def mean_vector(group):
    """
    Calculation of the average embedding vector by embedding vectors of its parts.
    """
    vectors = group["embeddings"].tolist()
    mean = [sum(i) / len(i) for i in zip(*vectors)]
    return mean


def rename_and_check_columns(df, verbose: bool = False):
    try:
        # Counting the number of columns
        num_cols = df.shape[1]
        if num_cols != 1:
            raise Exception(f"Incorrect number of dataframe columns {num_cols}")
        df.columns = ["text"]
        if verbose:
            print("Renamed the columns")
    except Exception as e:
        print("An error occurred at the stage of renaming columns:", e)


def embeddings_for_df(
    df,
    pkl_filename_list,
    chunk_size,
    max_length,
    device_name,
    batch_size,
    num_workers,
    max_token_seq_length,
):
    """
    Embeddings for the entire dataset
    """
    df_splitted = pd.DataFrame(columns=["text", "index", "embeddings", "input_ids", "attention_mask"])
    # Initialization of model
    if device_name == "cpu" or device_name == "mps" or pkl_filename_list is None:
        sbert = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2")
    else:
        # current_path = Path(__file__).parent
        # augmenter_file_path = current_path.parent / "pkl" / pkl_filename_list[0]
        # with open(augmenter_file_path, "rb") as file:
        #     sbert = pickle.load(file)
        pass

    bert = sbert[0].auto_model
    tokenizer = sbert[0].tokenizer

    # Splitting texts into parts to obtain embeddings for each of them,
    # if its length exceeds the maximum size of the processed sequence.
    for i in range(0, df.shape[0]):
        text = df.loc[i, "text"]
        t = tokenizer(
            text,
            max_length=max_token_seq_length,
            return_tensors="pt",
            truncation=True,
        )
        tokens = t["input_ids"]
        mask = t["attention_mask"]
        num_tokens = t["input_ids"].shape[1]
        if num_tokens <= max_token_seq_length:
            tokens = torch.nn.functional.pad(
                tokens,
                pad=(0, max_token_seq_length - num_tokens),
                mode="constant",
                value=0,
            )
            mask = torch.nn.functional.pad(
                mask,
                pad=(0, max_token_seq_length - num_tokens),
                mode="constant",
                value=0,
            )

            new_row = {
                "text": text,
                "index": i,
                "embeddings": [],
                "input_ids": tokens[0, :],
                "attention_mask": mask[0, :],
            }

            df_splitted = pd.concat([df_splitted, pd.DataFrame([new_row])], ignore_index=True)

        else:
            parts_tokens, parts_mask = split_document(tokens, mask, chunk_size)
            for token, mask in zip(parts_tokens, parts_mask):
                new_row = {
                    "text": "",
                    "index": i,
                    "embeddings": [],
                    "input_ids": token,
                    "attention_mask": mask,
                }

                df_splitted = pd.concat([df_splitted, pd.DataFrame([new_row])], ignore_index=True)

    # Initialize pretrained model
    model = SCCLBert(bert, tokenizer)

    data = df_splitted.copy()

    unshuffled_loader = token_loader(data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    custom_embedder = CustomEmbedder(
        model=model,
        tokenizer=tokenizer,
        device=torch.device(device_name),
        max_length=max_length,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Initialize keyBERT model with custom embeddings
    KeyBERTmodel = keybert.KeyBERT(model=custom_embedder)

    # Evaluate initial embeddings
    embeddings_1 = evaluate_embeddings(
        model,
        tokenizer,
        unshuffled_loader,
        max_length=max_length,
        device=torch.device(device_name),
    )
    # Split each row into a separate list
    split_arr = np.split(embeddings_1, embeddings_1.shape[0])

    # Combine lists into one list to create a vector
    merged_embeddings = [x.tolist()[0] for x in split_arr]

    if df_splitted.shape[0] != len(merged_embeddings):
        raise Exception(
            f"Discrepancy between the size of the dataframe {df_splitted.shape[0]} and the embedding list {len(merged_embeddings)}"
        )
    df_splitted["embeddings"] = merged_embeddings
    # Index grouping
    grouped = df_splitted.groupby(["index"])
    # Calculation of the average for the "embedding" column
    mean_embedding = grouped.apply(mean_vector)
    df["embeddings"] = mean_embedding
    # Convert to np.array for keyBERT
    for i in range(len(df["embeddings"])):
        df["embeddings"][i] = np.array(df["embeddings"][i])

    return KeyBERTmodel, data, mean_embedding, tokenizer, model
