import sys
import os
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans

def get_sentence_boundaries(filename):
    punct = [".", ":", "?", "!", ";"]
    with open(filename, "rb") as f:
        data = pickle.load(f)
    boundaries = []
    last_split = 0
    soft_breaks = 0
    hard_breaks = 0
    for i, d in enumerate(data):
        if d["token"] in punct:
            boundaries.append(i)
            last_split = i
        elif i-last_split > 20:
            if "\n" in d["token"]:
                boundaries.append(i)
                last_split = i
                soft_breaks += 1
            else:
                if i-last_split > 30:
                    boundaries.append(i)
                    last_split = i
                    hard_breaks += 1
        else:
            for p in punct:
                if p in d["token"]:
                    boundaries.append(i)
                    last_split = i
                    break

    return boundaries, soft_breaks, hard_breaks

def main():
    try:
        text_file = sys.argv[1]
        s_file = sys.argv[2]

    except:
        print("usage: python get_sentence_boundaries.py text_filename sentence_filename")
    # read data
    boundaries = None
    boundaries, sb, hb = get_sentence_boundaries(text_file)

    with open(s_file, "wb") as f:
        pickle.dump(boundaries, f)

    print(f"done: {len(boundaries)} sentences, {sb} sb, {hb} hb")
            



if __name__ == "__main__":
    main() 
