from pathlib import Path

import src.eta as eta


def test_split_merge_clusters():
    current_path = Path(__file__).parent
    filename = current_path / "20_newsgroup_text_only_50.csv"

    output_dict = eta.cluster_documents_with_keywords(filename, verbose=True)

    df = output_dict["df"]
    top_word_dict = output_dict["top_word_dict"]
    data = output_dict["data"]
    cluster_centers_1 = output_dict["cluster_centers"]
    radiuses = output_dict["radiuses"]
    cluster_model_1 = output_dict["cluster_model"]
    cluster_centers_2d_1 = output_dict["cluster_centers_2d"]
    reduce_model_1 = output_dict["reduce_model"]
    embeddings_1 = output_dict["embeddings"]
    tokenizer = output_dict["tokenizer"]
    model = output_dict["model"]

    cl_list = [0, 1]
    cluster_num = 0
    divisor = 2

    output_dict_split = eta.split_cluster(cluster_num, divisor, data, reduce_model_1, embeddings_1.to_list())
    data = output_dict_split["data"]
    cluster_centers_2d = output_dict_split["cluster_centers_2d"]
    radiuses = output_dict_split["radiuses"]

    output_dict_union = eta.union_clusters(cl_list, data, reduce_model_1, embeddings_1.to_list())
    data = output_dict_union["data"]
    cluster_centers_2d = output_dict_union["cluster_centers_2d"]
    radiuses = output_dict_union["radiuses"]

    # print("------------------------------------------")
    # print(print_full_columns(data))
    # print("------------------------------------------")
    # print(cluster_centers_2d)
    # print("------------------------------------------")
    # print(radiuses)

    # (
    #     data,
    #     cluster_centers_2,
    #     radiuses_2,
    #     cluster_model_2,
    #     embeddings_2,
    # ) = recalculate_model_pipeline(data, embeddings_1.to_list(), tokenizer, model)
    return
