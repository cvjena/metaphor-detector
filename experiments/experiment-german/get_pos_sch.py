import sys
import os
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans

NUM_FEATURES = 300

def load_pos(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)

    pos = []

    for d in tqdm(data):
        p = d["pos"]


        pos.append(p)

    del data
    return pos

def main():
    try:
        text_file = sys.argv[1]
        v_file = sys.argv[2]

    except:
        print("usage: python get_pos.py text_filename vector_filename")
    # read data
    pos = None

    pos = load_pos(text_file)

    print("save pos")
    with open(v_file, "wb") as f:
        pickle.dump(pos, f)

    print("done")


if __name__ == "__main__":
    main()
