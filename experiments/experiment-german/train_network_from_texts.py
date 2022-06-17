from tqdm import tqdm
import random
import pickle
from datetime import datetime
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import average_precision_score

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
ls = os.listdir
pj = os.path.join

from SimilarityNN import SimilarityNN

SEED = 42

VISU = True



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
    for fname in os.listdir("./german-pretrain/vector_pairs"):
        fname_full = os.path.join("./german-pretrain/vector_pairs", fname)
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

def load_text_data():
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
    

    filelist = []
    with open('splits/train.txt', 'r') as f:
        for l in f:
            filelist.append(l.strip())

    random.shuffle(filelist)
    val_border = int(len(filelist)*0.8)

    for fnum, fname in enumerate(filelist):
        if not "ADJ_NOUN" in fname:
            continue
        fname_full = os.path.join("vector_pairs", fname)
        if fnum < val_border:
            split = "train"
        else:
            split = "val"

        with open(fname_full, "rb") as f:
            d = pickle.load(f)

            # no metaphor = close together, label 1
            # metaphor = far apart, label -1
            label = [1]*len(d[0])
            data[split]["sourcea"]+=d[0]
            data[split]["sourcen"]+=d[1]
            data[split]["label"]+=label
            
            d2 = d.copy()
            random.shuffle(d2[0])
            label = [-1]*len(d[0])
            data[split]["sourcea"]+=d2[0]
            data[split]["sourcen"]+=d2[1]
            data[split]["label"]+=label

            d2 = d.copy()
            random.shuffle(d2[0])
            label = [-1]*len(d[0])
            data[split]["sourcea"]+=d2[0]
            data[split]["sourcen"]+=d2[1]
            data[split]["label"]+=label

    print("transform data to tensors")
    for split in ["train", "val"]:
        print(f"shuffle {split}")
        shuffle(data[split]["sourcea"], data[split]["sourcen"], data[split]["label"], random_state=SEED)
        print(f"transform {split}")
        for t in ["sourcea", "sourcen", "label"]:
            data[split][t] = torch.tensor(data[split][t])

    return data


def get_num_batches(dataset, bs):
    return int(len(dataset["label"])/bs)

def get_batch(dataset, bs, num):
    return dataset["sourcea"][num*bs:(num+1)*bs], dataset["sourcen"][num*bs:(num+1)*bs], dataset["label"][num*bs:(num+1)*bs]

def train_model(model, data, batch_size, epochs, max_no_improvement):
    loss_function = nn.CosineEmbeddingLoss()
    optimizer = optim.Adam(model.parameters())



    val_results = eval_model(model, data["val"], batch_size)
    print(f"val loss: {val_results}")
    train_results = eval_model(model, data["train"], batch_size)
    print(f"train loss: {train_results}")


    num_batches = get_num_batches(data["train"], batch_size)

    best_loss = val_results
    best_model = model
    no_improvement_counter = 0
    for epoch in range(epochs):
        train_loss = 0
        divisor = 0
        for b in tqdm(range(num_batches)):
            sourcea, sourcen, label = get_batch(data["train"], batch_size, b)
            model.zero_grad()

            preda = model(sourcea)
            predb = model(sourcen)

            loss = loss_function(preda, predb, label)
            train_loss += float(loss)
            loss.backward()
            divisor += 1.0
            optimizer.step()

        val_results = eval_model(model, data["val"], batch_size)
        train_loss /= divisor
        print(f"epoch {epoch} - train loss: {train_loss} - val loss: {val_results}")
        if val_results < best_loss:
            best_loss = val_results
            best_model = model
            no_improvement_counter = 0
        else:
            no_improvement_counter += 1
        if no_improvement_counter == max_no_improvement:
            break
    return best_model

def eval_model(model, dataset, batch_size):
    num_batches = get_num_batches(dataset, batch_size)
    scores = 0
    loss_function = nn.CosineEmbeddingLoss()
    with torch.no_grad():
        for b in range(num_batches):
            sourcea, sourcen, labels = get_batch(dataset, batch_size, b)
            resulta = model(sourcea)
            resultb = model(sourcen)
            scores += float(loss_function(resulta, resultb, labels))
        return scores/num_batches

def infer_stat(model, data, label):
    resulta = model(data["sourcea"][data["label"] == label])
    resultn = model(data["sourcen"][data["label"] == label])
    similarities = (nn.CosineSimilarity()(resulta, resultn)+1)/2
    print(torch.mean(similarities))

def get_scores(model, data):
    resulta = model(data["sourcea"])
    resultn = model(data["sourcen"])
    return (nn.CosineSimilarity()(resulta, resultn)+1)/2

def main():
    np.random.seed(SEED)
    random.seed(SEED)
    torch.random.manual_seed(SEED)

    batch_size = 128
    max_epochs = 30
    #max_epochs = 1
    max_no_improvement=3
    data = load_text_data()
    print(data["train"]["sourcea"].shape)
    print(data["val"]["sourcea"].shape)

    model = SimilarityNN(300, 300, 1, 100)
    
    model = train_model(model, data, batch_size, max_epochs, max_no_improvement)

    torch.save(model, "models/model_GER_base.pth")

    data = load_data()
    print("")
    print("stats no metaphors, should have high similarity")
    infer_stat(model, data["train"], 1)
    print("stats metaphors, should have low similarity")
    infer_stat(model, data["train"], -1)
    print("stats no metaphors, should have high similarity")
    infer_stat(model, data["val"], 1)
    print("stats metaphors, should have low similarity")
    infer_stat(model, data["val"], -1)
    similarities = get_scores(model, data["train"]).detach().numpy()
    labels = data["train"]["label"].detach().numpy()
    print(f"AP train: {average_precision_score(labels, similarities)}")

    apr = 0
    for i in range(100):
        similarities = np.random.rand(similarities.shape[0])
        apr += average_precision_score(labels, similarities)
    print(f"AP train rand: {apr/100}")

    similarities = get_scores(model, data["val"]).detach().numpy()
    labels = data["val"]["label"].detach().numpy()
    similarities = get_scores(model, data["val"]).detach().numpy()
    labels = data["val"]["label"].detach().numpy()
    print(f"AP val: {average_precision_score(labels, similarities)}")

    apr = 0
    for i in range(100):
        similarities = np.random.rand(similarities.shape[0])
        apr += average_precision_score(labels, similarities)
    print(f"AP val rand: {apr/100}")
    print("")
    if not VISU:
        exit()
    similarities = get_scores(model, data["train"]).detach().numpy()
    labels = data["train"]["label"].detach().numpy()
    precision, recall, threshold = precision_recall_curve(labels, similarities)

    with open("tnft_train.pickle", 'wb') as f:
        pickle.dump([precision, recall], f)


    plt.plot(recall, precision)
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.title("training data")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    #plt.show()
    similarities = get_scores(model, data["val"]).detach().numpy()
    labels = data["val"]["label"].detach().numpy()
    precision, recall, threshold = precision_recall_curve(labels, similarities)
    with open("tnft_test.pickle", 'wb') as f:
        pickle.dump([precision, recall], f)
    plt.plot(recall, precision)
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.title("validation data")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    #plt.show()
    #AP train: 0.60
    #AP train rand: 0.53
    #AP val: 0.70
    #AP val rand: 0.56


if __name__ == "__main__":
    main()
