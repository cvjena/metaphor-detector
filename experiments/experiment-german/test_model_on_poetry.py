# 1 model name

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
ls = os.listdir
pj = os.path.join

import sys
import pickle

from SimilarityNN import SimilarityNN
from sklearn.metrics import average_precision_score

def load_data():
    data = {
            "train": {
                "sourcea": [],
                "sourcen": [],
                "label": [],
                },
            "val": {
                "sourcea": [],
                "sourcen": [],
                "label": []
                }
            }
    for fname in os.listdir("./german-pretrain/vp_poetry"):
        fname_full = os.path.join("./german-pretrain/vp_poetry", fname)
        with open(fname_full, "rb") as f:
            d = pickle.load(f)
            if "test" in fname:
                split = "val"
            else:
                split = "train"

            # no metaphor = close together, label 1
            # metaphor = far apart, label -1
            if "nonmet" in fname:
                label = [1]*len(d[0])
            else:
                label = [-1]*len(d[0])

            data[split]["sourcea"]+=d[0]
            data[split]["sourcen"]+=d[1]
            data[split]["label"]+=label

    for split in ["train", "val"]:
        for t in ["sourcea", "sourcen", "label"]:
            data[split][t] = torch.tensor(data[split][t])

    return data


model = torch.load(sys.argv[1])
data = load_data()

def get_scores(model, data):
    resulta = model(data["sourcea"])
    resultn = model(data["sourcen"])
    return (nn.CosineSimilarity()(resulta, resultn)+1)/2

similarities = get_scores(model, data["val"]).detach().numpy()
labels = data["val"]["label"].detach().numpy()
print(f"AP val: {round(average_precision_score(labels, similarities), 2):.2f}")
