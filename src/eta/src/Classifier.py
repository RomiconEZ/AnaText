import os
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier


def train_classifier(data, embeddings):
    """
    Сlassifier training.
    """
    y = data["current_class"].values
    # Converting pandas Series of arrays to a two-dimensional NumPy array
    X = np.stack(embeddings.values)
    classifier = RandomForestClassifier()
    classifier.fit(X, y)
    return classifier


def text_classifier(text, classifier, model, tokenizer):
    """
    Predicting a class for a text.
    """
    max_length = int(os.getenv("MAX_LENGTH"))
    device_name = os.getenv("DEVICE_NAME")
    # Tokenizing the text
    t = tokenizer(
        text,
        max_length=max_length,
        return_tensors="pt",
        padding="longest",
        truncation=True,
    )
    device = torch.device(device_name)
    model.to(device)
    # Calculating embedding for text
    emb = model(
        input_ids=t["input_ids"].to(device),
        attention_mask=t["attention_mask"].to(device),
        task_type="evaluate",
    )
    np_emb = emb.detach().cpu().numpy()
    # Predicting a class
    return classifier.predict(np_emb)
