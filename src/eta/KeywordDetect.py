import itertools
from collections import Counter
import numpy as np
import pandas as pd
from keybert.backend import BaseEmbedder
from stop_words import get_stop_words
from eta.SCCLBert import evaluate_embeddings, text_loader


class CustomEmbedder(BaseEmbedder):
    def __init__(self, model, tokenizer, device, max_length, batch_size, num_workers):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers

    def embed(self, documents, **kwargs):
        documents_df = pd.DataFrame({"documents": documents})
        unshuffled_loader = text_loader(
            documents_df,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        # Evaluate initial embeddings
        embeddings = evaluate_embeddings(
            self.model,
            self.tokenizer,
            unshuffled_loader,
            max_length=self.max_length,
            device=self.device,
            loader_type="text",
        )
        return embeddings


def get_keywords_by_class(
    df: pd.DataFrame,
    text_col: str,
    emb_col: str,
    label_col: str,
    model,
    lang,
    no_above=0.6,
):
    """
    A function for finding the top words for each cluster of texts.
    ...
    Parameters
    ----------
        df: pd.DataFrame
            dataframe with columns {text_col}, {emb_col} и {label_col}
        lang: str ({ru} or {eng})
        model
            initialized keyBERT model
        no_above: float
            if a word is a key word in the part of clusters of more than a given value, then this word is excluded
    Returns
    -------
        keywords_by_class: dict ({{class label}: list of top words })
    """

    if lang == "eng" or lang is None:
        stop_words = get_stop_words("en")
    elif lang == "ru":
        stop_words = get_stop_words("ru")
    else:
        stop_words = get_stop_words("en")

    classes = df[label_col].unique()
    keywords_by_class = {}

    for label in classes:
        class_df = df[df[label_col] == label]
        documents = np.array(class_df[text_col])
        doc_embeddings = np.array(class_df[emb_col])
        # texts_keywords = model.extract_keywords(documents, top_n=8, doc_embeddings=doc_embeddings,
        # stop_words=stop_words, use_mmr=True, diversity=0.5)
        texts_keywords = model.extract_keywords(
            documents,
            top_n=6,
            doc_embeddings=doc_embeddings,
            stop_words=stop_words,
            use_maxsum=True,
            nr_candidates=15,
        )
        # one document in a class
        if len(documents) == 1:
            top_keywords = [keyword[0] for keyword in texts_keywords]
        # several documents in class
        else:
            class_keywords = [list(keyword[0] for keyword in text_keywords) for text_keywords in texts_keywords]
            # Take the 30 most common words from the top words for all texts of the same class
            top_keywords = [
                word[0] for word in Counter(list(itertools.chain.from_iterable(class_keywords))).most_common(30)
            ]

        keywords_by_class[label] = top_keywords

    # combine all keywords in one list
    all_keywords = []
    for label in classes:
        all_keywords += keywords_by_class[label]

    # Count keywords occurrences and add frequent ones to filter
    filters = set()
    for keyword in all_keywords:
        if all_keywords.count(keyword) >= (len(classes) * no_above):
            filters.add(keyword)

    # filter out frequent terms and trim keywords to 10 words
    for label in classes:
        keywords = keywords_by_class[label]

        keywords = [word for word in keywords if word not in filters]
        keywords = keywords[:10]

        keywords_by_class[label] = keywords

    return keywords_by_class


def get_top_words_for_clusters(df, model, lang, verbose: bool = False):
    """
    Find the top words.
    ...
    Parameters
    ----------
        df - dataframe with "text", "embeddings", "label" columns
        model - initialized keyBERT model
        lang - text language (ru or eng)
    Returns
    -------
        keywords_by_class: dict ({{class label}: list of top words })
    """
    if ~df.columns.isin(["text", "embeddings", "label"]).all():
        raise Exception("The necessary columns are missing")

    keywords_by_class = get_keywords_by_class(
        df,
        text_col="text",
        emb_col="embeddings",
        label_col="label",
        model=model,
        lang=lang,
    )
    if verbose:
        print("Found the top words")
    return keywords_by_class
