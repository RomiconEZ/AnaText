import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.utils import shuffle


def scatter_with_labels(reduced_data, labels, numpoints=None, title="", color_pallete=False):
    plt.figure(figsize=(12, 12))
    df1 = pd.DataFrame()
    df1["y"] = labels
    numcolors = len(np.unique(labels))
    df1["comp-1"] = reduced_data[:, 0]
    df1["comp-2"] = reduced_data[:, 1]
    df1 = shuffle(df1)
    if numpoints is not None:
        df1 = df1[:numpoints]
    colpall = sns.color_palette("husl", numcolors)
    sns.scatterplot(x="comp-1", y="comp-2", hue=df1.y.tolist(), palette=colpall, data=df1).set(title=title)
    if color_pallete:
        return colpall


def plot_loss(losses, loss_title="Loss", figsize=(20, 10)):
    plt.figure(figsize=figsize)
    plt.style.use("fivethirtyeight")
    plt.xlabel("Iterations")
    plt.ylabel(loss_title)
    plt.plot(losses)
