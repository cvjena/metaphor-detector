import pickle
import torch
import json
import torch.nn as nn
from tqdm import tqdm
import sys

from SimilarityNN import SimilarityNN

from cltk.tag.pos import POSTag
from cltk.lemmatize.middle_high_german.backoff import BackoffMHGLemmatizer
from cltk.corpus.middle_high_german.alphabet import normalize_middle_high_german
from cltk.phonology.middle_high_german.transcription import Word

mhg_pos_tagger = POSTag("middle_high_german")
lemmatizer = BackoffMHGLemmatizer()
sys.setrecursionlimit(100000)

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
with open('splits/train.txt', 'r') as f:
    for l in f:
        filelist.append(l.strip())

annotations = []

json_object = []

def get_tokens(fname):
    with open(fname, "rb") as f:
        data = pickle.load(f)
    tokens = []
    tok_norm = []
    pos = []
    for d in data:
        t = d[0]
        tokens.append(d[0])
        pos.append(d[1])
        normalized = str(normalize_middle_high_german(t, to_lower_all=True))
        ascii_word = Word(normalized).ASCII_encoding()
        tok_norm.append(ascii_word)

    del data
    del normalized
    del ascii_word
    del pos

    return tokens, tok_norm


filelist = []
with open('splits/train.txt', 'r') as f:
    for l in f:
        filelist.append(l.strip())


jobj = []

torch.no_grad()

score = []

for fname in tqdm(filelist):
    fname_full = os.path.join("vector_pairs", fname)
    fname_cut = fname.replace("_ADJ_NA", "")

    tokens, tok_norm = get_tokens(pj("./texts/",  fname_cut))

    with open(pj('vector_pairs', fname.replace(".pickle", "_ADJ_NA.pickle")), 'rb') as f:
       sources, targets, sents, ids = pickle.load(f)

    for i in range(len(sources)):
        tfirst = max(0, ids[i]-5)
        tlast = min(len(tokens), ids[i]+7)

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
