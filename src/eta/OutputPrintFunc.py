import pandas as pd


def print_pretty_clusters(cluster_dict):
    # Start of the output block
    print("\n" + "-" * 40)
    print("{:^40}".format("Топ Слова для Кластеров"))
    print("-" * 40)

    for cluster, words in cluster_dict.items():
        # Header for each cluster
        print(f"\nКластер №{cluster:d}")
        print("=" * 20)

        # We go through the words in the cluster and display them
        for word in words:
            print(f"- {word:<15}")

    # Completion of the output block
    print("\n" + "-" * 40)


def print_large_dataframe(df, max_rows=10, max_columns=5):
    with pd.option_context("display.max_rows", max_rows, "display.max_columns", max_columns):
        print(df)


def print_full_columns(df, num_rows=5):
    with pd.option_context("display.max_rows", num_rows, "display.max_columns", None):
        print(df)
