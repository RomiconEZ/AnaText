from pathlib import Path

from .DefineDevice import setup_env
from .KeywordDetect import get_top_words_for_clusters
from .OutputPrintFunc import print_pretty_clusters
from .PipelineFunc import (
    approximate_n_clusters_pipeline,
    clustering_pipeline,
    load_preprocess_embeddings_pipeline,
    split_cluster_pipeline,
    union_clusters_pipeline,
)

# -----------------------------------------------------------------------------


def cluster_documents_with_keywords(filename: str | Path, verbose: bool = False):
    """
    Document clustering with keywords for each cluster
    ...

    Parameters
    ----------
        filename: str
            path to the file
        verbose: bool
            flag for printing

    Returns
    -------
        df: pd.DataFrame
            dataframe with columns {text_col}, {emb_col} и {label_col}
        top_word_dict: dict
            dictionary with keywords for each cluster
        data: pd.DataFrame
            dataframe for internal function operation
        cluster_centers: np.array
            cluster centers in the original dimension
        radiuses: np.array
            relative radii of clusters for two-dimensional representation
        cluster_model
            KMeans with the number of clusters determined at the approximation stage
        cluster_centers_2d: np.array
            cluster centers for 2D Visualization
        reduce_model
            dimension reduction model for visualization
        embeddings: np.array
            embeddings
        tokenizer
        model
            SCCLBert
    """
    try:
        current_path = Path(__file__).parent
        setup_env(current_path)

        # Check that the file name is a string
        if not isinstance(filename, (str, Path)):
            raise ValueError("filename must be a string or a Path object")

        filename = str(filename)

        # Loading, pre-processing, calculation of document embeddings
        (
            df,
            KeyBERTmodel,
            lang,
            data,
            embeddings_1,
            tokenizer,
            model,
        ) = load_preprocess_embeddings_pipeline(filename=filename, verbose=verbose)

        # Cluster number approximation
        num_clusters = approximate_n_clusters_pipeline(df, verbose=verbose)

        # Document clustering
        (
            df,
            data,
            cluster_centers_1,
            radiuses,
            cluster_model_1,
            cluster_centers_2d_1,
            reduce_model_1,
        ) = clustering_pipeline(df, num_clusters, embeddings_1.to_list(), verbose=verbose)

        # Getting keywords for each cluster
        top_word_dict = get_top_words_for_clusters(df, KeyBERTmodel, lang, verbose)

        if verbose:
            print_pretty_clusters(top_word_dict)

        return {
            "df": df,
            "top_word_dict": top_word_dict,
            "data": data,
            "cluster_centers": cluster_centers_1,
            "radiuses": radiuses,
            "cluster_model": cluster_model_1,
            "cluster_centers_2d": cluster_centers_2d_1,
            "reduce_model": reduce_model_1,
            "embeddings": embeddings_1,
            "tokenizer": tokenizer,
            "model": model,
        }

    except Exception as e:
        print("An error occurred at the document clustering:", e)
        return None


def split_cluster(cluster_num, divisor, data, reduce_model, embeddings):
    """
    Splitting cluster with number {cluster_num} into new clusters in the amount of {divisor}.
    ...

    Parameters
    ----------
        cluster_num: int
            number of the cluster
        divisor: int
            amount of clusters
        data: pd.DataFrame
            dataframe with columns "current_class", "embeddings", "class"
        embeddings: list(list)
        reduce_model
            model for reducing embedding dimensions to 2 for visualization

    Returns
    -------
        data: pd.DataFrame
            dataframe with columns "current_class", "embeddings", "class", "new_class", "old_class"
        cluster_centers_2d: list(list)
            cluster centers for 2D Visualization
        radiuses: dict (cluster_num: float)
            relative radii of clusters for two-dimensional representation
    """
    try:
        current_path = Path(__file__).parent
        setup_env(current_path)

        data, new_cluster_centers_2d, new_radiuses = split_cluster_pipeline(
            cluster_num, divisor, data, reduce_model, embeddings
        )
        return {
            "data": data,
            "cluster_centers_2d": new_cluster_centers_2d,
            "radiuses": new_radiuses,
        }
    except Exception as e:
        print("An error occurred at the cluster splitting:", e)
        return None


def union_clusters(cl_list, data, reduce_model, embeddings):
    """
    Splitting cluster with number {cluster_num} into new clusters in the amount of {divisor}.
    ...

    Parameters
    ----------
        cl_list: list
            list of numbers of clusters to merge
        data: pd.DataFrame
            dataframe with columns "current_class", "embeddings", "class"
        embeddings: list(list)
        reduce_model
            model for reducing embedding dimensions to 2 for visualization

    Returns
    -------
        data: pd.DataFrame
            dataframe with columns "current_class", "embeddings", "class", "new_class", "old_class"
        cluster_centers_2d: list(list)
            cluster centers for 2D Visualization
        radiuses: dict (cluster_num: float)
            relative radii of clusters for two-dimensional representation
    """
    try:
        current_path = Path(__file__).parent
        setup_env(current_path)

        data, new_cluster_centers_2d, new_radiuses = union_clusters_pipeline(cl_list, data, reduce_model, embeddings)
        return {
            "data": data,
            "cluster_centers_2d": new_cluster_centers_2d,
            "radiuses": new_radiuses,
        }
    except Exception as e:
        print("An error occurred at the cluster union:", e)
        return None
