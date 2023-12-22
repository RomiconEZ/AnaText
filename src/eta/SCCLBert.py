import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from torch.nn import Parameter
from torch.utils.data import DataLoader, Dataset


# Divide and Unite clusters
def current_class(x):
    return int(x[-1])


def delete_last_elem(x):
    return x[:-1]


def cluster_radiuses(cluster_labels, data):
    labels_dict = data["current_class"].value_counts().to_dict()
    radiuses = {}
    len_data = len(data)
    for elem in cluster_labels:
        radiuses[elem] = labels_dict[elem] / len_data
    return radiuses


def initial_operation(num_clusters, data, embeddings, initial_centers=None, label_key="class"):
    if num_clusters is None:
        num_clusters = len(data["current_class"].unique())

    cluster_labels, cluster_centers, cluster_model = cluster_embeddings(
        embeddings, num_clusters, initial_centers=initial_centers, return_model=True
    )

    data["class"] = [[label] for label in cluster_labels]
    data["current_class"] = data[label_key].apply(current_class)

    max_label = data["class"].max()[0]
    unique_cluster_labels = list(range(0, max_label + 1))

    radiuses = cluster_radiuses(unique_cluster_labels, data)

    return data, cluster_centers, radiuses, cluster_model


def split_cluster(cluster_num, divisor, data, embeddings, reduce_model, label_key="class"):
    """
    This function splits the cluster with the label cluster_num by divisor number of clusters
    """

    def duplicate_last_element(l_list):
        l_list.append(l_list[-1])
        return l_list

    max_label = data[label_key].apply(lambda x: int(x[-1])).max()
    divdf = data[data[label_key].apply(lambda x: int(x[-1])) == cluster_num]
    df_emb = pd.DataFrame(embeddings)
    cluster_labels, cluster_centers = cluster_embeddings(df_emb.loc[divdf.index].values, divisor, initial_centers=None)
    data.loc[divdf.index, "new_class"] = cluster_labels
    data.loc[divdf.index, "old_class"] = data.loc[divdf.index, label_key]
    data.loc[divdf.index, label_key] = data.loc[divdf.index].apply(
        lambda x: x.old_class + [max_label + 1 + int(x.new_class)], axis=1
    )

    new_cluster_labels = [cluster_num] + list(range(max_label + 1, max_label + divisor))

    data.loc[~data.index.isin(divdf.index), label_key] = data.loc[~data.index.isin(divdf.index), label_key].apply(
        duplicate_last_element
    )

    max_new_label = cluster_labels.max()

    maxl_slice = data[data[label_key].apply(lambda x: int(x[-1])) == max_label + max_new_label + 1]

    def remake_label_list(l_list):
        l_list[-1] = cluster_num
        return l_list

    data.loc[maxl_slice.index, label_key] = maxl_slice[label_key].apply(remake_label_list)
    data["current_class"] = data[label_key].apply(current_class)

    new_centers = recalculate_clusters_centers(data.loc[divdf.index], df_emb.loc[divdf.index])
    new_radiuses = cluster_radiuses(new_cluster_labels, data)
    new_centers = reduce_model.transform(new_centers)
    return data, new_centers, new_radiuses


def undo_last_operation(data, cluster_centers_history):
    if len(cluster_centers_history) > 1:
        data["class"] = data["class"].apply(delete_last_elem)
        del cluster_centers_history[-1]
    return data, cluster_centers_history


def union_clusters(cl_list, data, embeddings, reduce_model, label_key="class"):
    """
    This function unites clusters in list of clusters cl_list
    """

    max_label = data[label_key].apply(lambda x: int(x[-1])).max()
    unite_df = data[data[label_key].apply(lambda x: int(x[-1])).isin(cl_list)]
    df_emb = pd.DataFrame(embeddings)
    cluster_labels = np.zeros(len(df_emb.loc[unite_df.index]))
    data.loc[unite_df.index, "new_class"] = cluster_labels
    data.loc[unite_df.index, "old_class"] = data.loc[unite_df.index, label_key]
    new_max_label = max_label - len(cl_list) + 1
    all_labels = list(range(0, max_label + 1))
    new_labels = list(range(1, new_max_label + 1))
    old_labels = list(set(all_labels) - set(cl_list))
    label_translator = {old_labels[i]: new_labels[i] for i in range(len(old_labels))}

    def remake_label_list_union(l_list):
        l_list.append(label_translator[l_list[-1]])
        return l_list

    data.loc[unite_df.index, label_key] = data.loc[unite_df.index].apply(
        lambda x: x.old_class + [int(x.new_class)], axis=1
    )
    data.loc[~data.index.isin(unite_df.index), label_key] = data.loc[~data.index.isin(unite_df.index), label_key].apply(
        remake_label_list_union
    )
    data["current_class"] = data[label_key].apply(current_class)

    new_center = recalculate_clusters_centers(data.loc[unite_df.index], df_emb.loc[unite_df.index])
    new_radiuses = cluster_radiuses([0], data)
    new_center = reduce_model.transform(new_center)

    return data, new_center, new_radiuses


def recalculate_clusters_centers(data, embeddings, requires_labels=False):
    """This function returns new centers of clusters"""
    df_emb = pd.DataFrame(embeddings)
    labels = []
    cluster_centers = []
    for label in sorted(data["current_class"].unique()):
        labels.append(label)
        cluster_center = df_emb.loc[data[data["current_class"] == label].index].values.mean(axis=0)

        cluster_centers.append(cluster_center)
    if requires_labels:
        return np.array(cluster_centers), labels
    else:
        return np.array(cluster_centers)


def recalculate_model(
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
    ):
    if device_name != "cuda":
        return data, None, None, None, embeddings

    device = torch.device(device_name)

    # Recalculate cluster centers
    cluster_centers_2 = recalculate_clusters_centers(data, embeddings)

    # Setting new cluster centers to model
    model.set_cluster_centers(cluster_centers_2)

    # Initialize optimizer
    optimizer = get_optimizer(model, lr=lr, lr_scale=lr_scale)

    # Initialize loss functions
    cluster_loss = nn.KLDivLoss(reduction="batchmean")

    contrast_loss = PairConLoss(temperature=temperature)

    # Initialize augmenter
    current_path = Path(__file__).parent
    augmenter_file_path = current_path.parent / "pkl" / augmenter_pkl_name

    if device_name == "cuda":
        with open(augmenter_file_path, "rb") as file:
            augmenter = torch.load(file)

    # Train loop
    train_loader = text_loader(data, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    sumloss, losslist, clusterloss = train(
        train_loader,
        model,
        tokenizer,
        augmenter,
        contrast_loss,
        cluster_loss,
        optimizer,
        device,
        max_iter=max_iter_1,
        max_length=max_length,
        eta=eta,
        print_freq=25,
    )

    unshuffled_loader = token_loader(data, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Evaluate embeddings after additional training
    embeddings_2 = evaluate_embeddings(model, tokenizer, unshuffled_loader, max_length=max_length, device=device)

    num_clusters = len(data["current_class"].value_counts())

    # Make new clusterisation by new embeddings and initial cluster centers in trained cluster centers

    data, cluster_centers, radiuses, cluster_model = initial_operation(num_clusters, data, embeddings=embeddings_2)

    # Evaluate embeddigs after second additional training
    embeddings_2 = evaluate_embeddings(model, tokenizer, unshuffled_loader, max_length=128, device=device)

    # Make new clusterisation by new embeddings and initial cluster centers in trained cluster centers

    trained_centers = model.cluster_centers.to("cpu", copy=True).detach().numpy()
    data, cluster_centers_2, radiuses_2, cluster_model_2 = initial_operation(
        embeddings=embeddings_2,
        data=data,
        num_clusters=len(trained_centers),
        initial_centers=trained_centers,
    )
    return data, cluster_centers_2, radiuses_2, cluster_model_2, embeddings_2


# Reducing dimensions
def reduce_dimension_truncated_svd(embeddings, n_components=2):
    truncated_svd = TruncatedSVD(n_components=2, random_state=42)
    reduced_data = truncated_svd.fit_transform(embeddings)
    return reduced_data, truncated_svd


class SCCLBert(nn.Module):
    def __init__(self, bert_model, tokenizer, cluster_centers=None, alpha=1.0):
        super().__init__()

        self.tokenizer = tokenizer
        self.bert = bert_model
        self.emb_size = self.bert.config.hidden_size
        self.alpha = alpha

        # Instance-CL head
        self.contrast_head = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.emb_size, 128),
        )

        # Clustering head
        if cluster_centers:
            initial_cluster_centers = torch.tensor(cluster_centers, dtype=torch.float, requires_grad=True)
            self.cluster_centers = Parameter(initial_cluster_centers)

    def forward(self, input_ids, attention_mask, task_type="virtual"):
        if task_type == "evaluate":
            return self.get_mean_embeddings(input_ids, attention_mask)

        elif task_type == "virtual":
            input_ids_1, input_ids_2 = torch.unbind(input_ids, dim=1)
            attention_mask_1, attention_mask_2 = torch.unbind(attention_mask, dim=1)

            mean_output_1 = self.get_mean_embeddings(input_ids_1, attention_mask_1)
            mean_output_2 = self.get_mean_embeddings(input_ids_2, attention_mask_2)
            return mean_output_1, mean_output_2

        elif task_type == "explicit":
            input_ids_1, input_ids_2, input_ids_3 = torch.unbind(input_ids, dim=1)
            attention_mask_1, attention_mask_2, attention_mask_3 = torch.unbind(attention_mask, dim=1)

            mean_output_1 = self.get_mean_embeddings(input_ids_1, attention_mask_1)
            mean_output_2 = self.get_mean_embeddings(input_ids_2, attention_mask_2)
            mean_output_3 = self.get_mean_embeddings(input_ids_3, attention_mask_3)
            return mean_output_1, mean_output_2, mean_output_3

        else:
            raise Exception("TRANSFORMER ENCODING TYPE ERROR! OPTIONS: [EVALUATE, VIRTUAL, EXPLICIT]")

    def set_cluster_centers(self, cluster_centers):
        initial_cluster_centers = torch.tensor(cluster_centers, dtype=torch.float, requires_grad=True)
        self.cluster_centers = Parameter(initial_cluster_centers)

    def get_mean_embeddings(self, input_ids, attention_mask):
        bert_output = self.bert.forward(input_ids=input_ids, attention_mask=attention_mask)
        attention_mask = attention_mask.unsqueeze(-1)
        mean_output = torch.sum(bert_output[0] * attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
        return mean_output

    def get_cluster_prob(self, embeddings):
        norm_squared = torch.sum((embeddings.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator**power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)

    def local_consistency(self, embd0, embd1, embd2, criterion):
        p0 = self.get_cluster_prob(embd0)
        p1 = self.get_cluster_prob(embd1)
        p2 = self.get_cluster_prob(embd2)

        lds1 = criterion(p1, p0)
        lds2 = criterion(p2, p0)
        return lds1 + lds2

    def contrast_logits(self, embd1, embd2=None):
        feat1 = F.normalize(self.contrast_head(embd1), dim=1)
        if embd2 is not None:
            feat2 = F.normalize(self.contrast_head(embd2), dim=1)
            return feat1, feat2
        else:
            return feat1


# Train Loaders


class TextSet(Dataset):
    def __init__(self, dataframe):
        self.text_list = dataframe.iloc[:, 0].values

    def __len__(self):
        return len(self.text_list)

    def __getitem__(self, idx):
        return {"text": self.text_list[idx]}


def text_loader(dataframe, batch_size, shuffle=True, num_workers=4):
    text_dataset = TextSet(dataframe)
    txt_loader = DataLoader(text_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return txt_loader


# Evaluate embeddings


def batch_to_device(batch, device: torch.device):
    """
    send a pytorch batch to a device (CPU/GPU)
    """
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)
    return batch


def get_batch_token(text, tokenizer, max_length):
    token_feat = tokenizer(
        text,
        max_length=max_length,
        return_tensors="pt",
        padding="longest",
        truncation=True,
    )
    return token_feat


def evaluate_embeddings(
    model,
    tokenizer,
    train_loader,
    max_length,
    device,
    loader_type="token"
    ):

    """
    loader type:
            'token' - token loader
            'text' - text loader
    """
    model.eval()

    model.to(device)

    for i, batch in enumerate(train_loader):
        if loader_type == "text":
            text = batch["text"]
            tokenized_features = get_batch_token(text, tokenizer, max_length)
        elif loader_type == "token":
            tokenized_features = batch
        else:
            raise Exception("UNKNOWN DATASET EXTENTION! AVALIABLE OPTIONS: ['token', 'text']")

        tokenized_features = batch_to_device(tokenized_features, device)

        with torch.no_grad():
            corpus_embeddings = model.get_mean_embeddings(
                input_ids=tokenized_features["input_ids"],
                attention_mask=tokenized_features["attention_mask"],
            )

            if i == 0:
                all_embeddings = corpus_embeddings.cpu().detach().numpy()

            else:
                all_embeddings = np.concatenate((all_embeddings, corpus_embeddings.cpu().detach().numpy()), axis=0)
    return all_embeddings


# Clustering


def cluster_embeddings(embeddings, num_clusters, initial_centers=None, return_model=False):
    clustering_model = KMeans(n_clusters=num_clusters)
    if initial_centers is not None:
        assert len(initial_centers) == num_clusters
        params = {"init": initial_centers, "n_init": 1}
        clustering_model.set_params(**params)
    clustering_model.fit(embeddings)
    cluster_labels = clustering_model.labels_

    cluster_centers = clustering_model.cluster_centers_
    if return_model:
        return cluster_labels, cluster_centers, clustering_model
    else:
        return cluster_labels, cluster_centers


# Augmentation


def augment(texts, augmenter):
    def augment_one_text(text):
        augmented_text = augmenter.augment(text)
        try:
            return augmented_text[0]
        except:
            return ""

    texts = [augment_one_text(text) for text in texts]
    return texts


# Optimizer


def get_optimizer(model, lr=1e-5, lr_scale=1):
    optimizer = torch.optim.Adam(
        [
            {"params": model.bert.parameters()},
            {"params": model.contrast_head.parameters(), "lr": lr * lr_scale},
            {"params": model.cluster_centers, "lr": lr * lr_scale},
        ],
        lr=lr,
    )

    return optimizer


# Loss functions


class DCLoss(nn.Module):
    def __init__(self, lamda=0.5):
        super().__init__()
        self.xentropy = nn.CrossEntropyLoss()
        self.t = lamda

    def forward(self, net, x, x_tf, head):
        """Partition Uncertainty Index

        Arguments:
            x {Tensor} -- [assignment probabilities of original inputs (N x K)]
            y {Tensor} -- [assignment probabilities of perturbed inputs (N x k)]

        Returns:
            [Tensor] -- [Loss value]
        """
        assert x.shape == x_tf.shape, "Inputs are required to have same shape"

        p = x.sum(0).view(-1)
        p /= p.sum()
        loss_ne = np.log(p.size(0)) + (p * p.log()).sum()

        # t = 0.5

        x_norm = F.normalize(x)
        x_tf_norm = F.normalize(x_tf)

        logits = torch.mm(x_norm, x_tf_norm.t()) / self.t
        device = torch.device(os.getenv("DEVICE_NAME"))
        device_name = os.getenv("DEVICE_NAME")

        if device_name == "cpu":
            labels = torch.tensor(range(logits.shape[0])).cpu()
        elif device_name == "cuda":
            labels = torch.tensor(range(logits.shape[0])).cuda()
        elif device_name == "mps":
            labels = torch.tensor(range(logits.shape[0])).to(torch.device("mps"))

        # for c
        x_norm = F.normalize(x, dim=0)
        x_tf_norm = F.normalize(x_tf, dim=0)
        logits_c = torch.mm(x_norm.t(), x_tf_norm) / self.t

        if device_name == "cpu":
            labels_c = torch.tensor(range(logits_c.shape[0])).cpu()
        elif device_name == "cuda":
            labels_c = torch.tensor(range(logits_c.shape[0])).cuda()
        elif device_name == "mps":
            labels_c = torch.tensor(range(logits_c.shape[0])).to(torch.device("mps"))

        loss = torch.nn.CrossEntropyLoss()(logits, labels) + torch.nn.CrossEntropyLoss()(logits_c, labels_c) + loss_ne

        return loss


class PairConLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super().__init__()
        self.temperature = temperature
        self.eps = 1e-08
        # print(f"\n Initializing PairConLoss \n")

    def forward(self, features_1, features_2):
        device = features_1.device
        batch_size = features_1.shape[0]
        features = torch.cat([features_1, features_2], dim=0)
        mask = torch.eye(batch_size, dtype=torch.bool).to(device)
        mask = mask.repeat(2, 2)
        mask = ~mask

        pos = torch.exp(torch.sum(features_1 * features_2, dim=-1) / self.temperature)
        pos = torch.cat([pos, pos], dim=0)
        neg = torch.exp(torch.mm(features, features.t().contiguous()) / self.temperature)
        neg = neg.masked_select(mask).view(2 * batch_size, -1)

        neg_mean = torch.mean(neg)
        pos_n = torch.mean(pos)
        Ng = neg.sum(dim=-1)

        loss_pos = (-torch.log(pos / (Ng + pos))).mean()

        return {
            "loss": loss_pos,
            "pos_mean": pos_n.detach().cpu().numpy(),
            "neg_mean": neg_mean.detach().cpu().numpy(),
            "pos": pos.detach().cpu().numpy(),
            "neg": neg.detach().cpu().numpy(),
        }


def target_distribution(batch: torch.Tensor) -> torch.Tensor:
    weight = (batch**2) / (torch.sum(batch, 0) + 1e-9)
    return (weight.t() / torch.sum(weight, 1)).t()


# Train steps


def prepare_transformer_input(batch, tokenizer, augmenter, max_length):
    text = batch["text"]
    batch_size = len(text)
    # print('batch_size', batch_size)
    text2 = augment(text, augmenter)
    feat1 = {}
    feat2 = {}

    feat = get_batch_token(text + text2, tokenizer, max_length)

    feat1["input_ids"], feat2["input_ids"] = torch.split(feat["input_ids"], batch_size, dim=0)
    feat1["attention_mask"], feat2["attention_mask"] = torch.split(feat["attention_mask"], batch_size, dim=0)

    input_ids = torch.stack((feat1["input_ids"], feat2["input_ids"]), dim=1)
    attention_mask = torch.stack((feat1["attention_mask"], feat2["attention_mask"]), dim=1)
    device_name = os.getenv("DEVICE_NAME")

    if device_name == "cpu":
        return input_ids.cpu(), attention_mask.cpu()
    elif device_name == "cuda":
        return input_ids.cuda(), attention_mask.cuda()
    elif device_name == "mps":
        return input_ids.to(torch.device("mps")), attention_mask.to(torch.device("mps"))


def train_step(model, optimizer, contrast_loss, cluster_loss, input_ids, attention_mask, eta=1):
    embd1, embd2 = model(input_ids, attention_mask, task_type="virtual")

    # Instance-CL loss
    feat1, feat2 = model.contrast_logits(embd1, embd2)
    losses = contrast_loss(feat1, feat2)
    loss = eta * losses["loss"]

    # Clustering loss

    output = model.get_cluster_prob(embd1)
    target = target_distribution(output).detach()

    cluster_los = cluster_loss((output + 1e-08).log(), target) / output.shape[0]
    loss += 0.5 * cluster_los
    losses["cluster_loss"] = cluster_los.item()

    losses["sum_loss"] = loss.item()

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return losses


# Training evaluations


def print_losses(losses=None, global_step=0):
    print(f"[{global_step}]-----")
    if losses is not None:
        for key, val in losses.items():
            if isinstance(val, torch.Tensor):
                try:
                    print(key, val.item())
                except:
                    pass
            elif (isinstance(val, np.ndarray) and val.shape == ()) or isinstance(val, float):
                print(key, val)


def train(
    train_loader,
    model,
    tokenizer,
    augmenter,
    contrast_loss,
    cluster_loss,
    optimizer,
    device: torch.device,
    max_length,
    max_iter=1000,
    eta=1,
    print_freq=0,
    train_loader_iter=None,):
    print(f"\n={max_iter}/{len(train_loader)}=Iterations/Batches")
    sumloss = []
    losslist = []
    clusterloss = []
    model.to(device)
    model.train()
    for i in np.arange(max_iter + 1):
        # print('Iter ', i)
        try:
            batch = next(train_loader_iter)
        except:
            train_loader_iter = iter(train_loader)
            batch = next(train_loader_iter)

        input_ids, attention_mask = prepare_transformer_input(batch, tokenizer, augmenter, max_length=max_length)
        losses = train_step(
            model,
            optimizer,
            contrast_loss,
            cluster_loss,
            input_ids,
            attention_mask,
            eta=eta,
        )
        sumloss.append(losses["sum_loss"])
        losslist.append(losses["loss"].item())
        clusterloss.append(losses["cluster_loss"])
        # print(losses)
        if (print_freq > 0) and ((i % print_freq == 0) or (i == max_iter)):
            print_losses(losses=losses, global_step=i)

    return sumloss, losslist, clusterloss


class TokenSet(Dataset):
    def __init__(self, dataframe):
        self.text_list = dataframe["text"]
        self.input_ids_list = dataframe["input_ids"]
        self.attention_mask_list = dataframe["attention_mask"]

    def __len__(self):
        return len(self.text_list)

    def __getitem__(self, idx):
        return {
            "text": self.text_list[idx],
            "input_ids": self.input_ids_list[idx],
            "attention_mask": self.attention_mask_list[idx],
        }


def token_loader(dataframe, batch_size, shuffle=True, num_workers=0):
    token_dataset = TokenSet(dataframe)

    return DataLoader(token_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
