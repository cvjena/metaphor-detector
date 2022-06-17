import sys
import os
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans

NUM_FEATURES = 300

def load_vectors(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)

    vectors = []

    for d in tqdm(data):
        v = d["vectors"]

        if v is not None:
            assert(v.size == NUM_FEATURES)

        vectors.append(v)

    del data
    return vectors

def main():
    try:
        text_file = sys.argv[1]
        v_file = sys.argv[2]

    except:
        print("usage: python get_vectors.py text_filename vector_filename")
    # read data
    vectors = None

    vectors = load_vectors(text_file)

    print("save vectors")
    with open(v_file, "wb") as f:
        pickle.dump(vectors, f)

    print("done")
            



if __name__ == "__main__":
    main()
