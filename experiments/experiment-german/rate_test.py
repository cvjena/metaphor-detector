import pickle
import torch
import json
import torch.nn as nn
from tqdm import tqdm

from SimilarityNN import SimilarityNN

import sys
import os
ls = os.listdir
pj = os.path.join

# first rank the candidates

# 1 model
# 2 json output file

## load model
model = torch.load(sys.argv[1])
print(model)

filelist = []
with open('splits/test.txt', 'r') as f:
    for l in f:
        filelist.append(l.strip())

annotations = []

json_object = []

def get_tokens(fname):
    with open(fname, 'rb') as f:
        _tokens = pickle.load(f)
    tokens = [d["token"] for d in _tokens]
    del _tokens
    return tokens


filelist = []
with open('splits/test.txt', 'r') as f:
    for l in f:
        filelist.append(l.strip())


jobj = []

torch.no_grad()

score = []

for fname in tqdm(filelist):
    if not "ADJ_NOUN" in fname:
        continue
    fname_full = os.path.join("vector_pairs", fname)
    fname_cut = fname.replace("_ADJ_NOUN", "")

    tokens = get_tokens(pj("/lvdata/compling/gerdracor/processed",  fname_cut))
    with open(pj("./sentences", fname_cut), "rb") as f:
        sentences = pickle.load(f)
    sentences = [0]+sentences

    with open(pj('vector_pairs', fname), 'rb') as f:
       sources, targets, sents, ids = pickle.load(f)

    for i in range(len(sources)):
        tfirst = sentences[sents[i]-1]
        tlast = sentences[sents[i]]

        source = torch.tensor([sources[i]])
        target = torch.tensor([targets[i]])

        source_embedding = model(source)
        target_embedding = model(target)

        score = float(nn.CosineSimilarity()(source_embedding, target_embedding).detach().numpy())
        score += 1
        score /= 2.0
        score = 1-score

        jelem_id = fname_cut.split(".")[0].strip() + "_" + str(i)


        jelem = {'id': jelem_id,
                'sentence': " ".join(tokens[tfirst:tlast]),
                'word pair': [tokens[ids[i]], tokens[ids[i]+1]],
                'score': score,
                'annotation': ''}

        jobj.append(jelem)


with open(sys.argv[2], 'w') as f:
    f.write(json.dumps(jobj, indent=4, ensure_ascii=False))
