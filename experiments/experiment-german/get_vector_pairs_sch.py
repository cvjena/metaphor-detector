import sys
import os
from tqdm import tqdm
import pickle
import numpy as np

pj = os.path.join
ls = os.listdir
ss = []

def process_file(fname, source, target):

    global ss # debug

    with open(pj("./sentences_sch", fname), "rb") as f:
        sentences = [0]+pickle.load(f)
    with open(pj("./pos_sch", fname), "rb") as f:
        pos = pickle.load(f)
    with open(pj("./vectors_sch", fname), "rb") as f:
        vectors = pickle.load(f)
    
    sources = []
    targets = []
    sents = []
    ids = []

    sentences = [0] + sentences


    for nsent in range(1, len(sentences)):
        ibegin = sentences[nsent-1]
        iend = sentences[nsent]
        s = []
        ts = []
        for i in range(ibegin, iend-1):
            if pos[i] == source and pos[i+1] == target:
                if vectors[i] is not None and vectors[i+1] is not None:
                    sources.append(vectors[i])
                    targets.append(vectors[i+1])
                    sents.append(nsent-1)
                    ids.append(i)

    fname_save = pj("./vector_pairs_sch", f"{fname.split('.')[0]}_{source}_{target}.pickle")
    with open(fname_save, "wb") as f:
        pickle.dump([sources, targets, sents, ids], f)

    del sentences
    del pos
    del vectors

def main():
    source = sys.argv[1]
    target = sys.argv[2]

    for fname in tqdm(ls("./vectors_sch")):
        if ".pickle" in fname:
            process_file(fname, source, target)

if __name__ == "__main__":
    main()
