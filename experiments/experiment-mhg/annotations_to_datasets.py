import json
import spacy
import numpy as np
import sys
import pickle

from cltk.tag.pos import POSTag
from cltk.lemmatize.middle_high_german.backoff import BackoffMHGLemmatizer
from cltk.corpus.middle_high_german.alphabet import normalize_middle_high_german
from cltk.phonology.middle_high_german.transcription import Word
import fasttext

mhg_pos_tagger = POSTag("middle_high_german")
lemmatizer = BackoffMHGLemmatizer()
sys.setrecursionlimit(100000)

we_len= 100

fasttext_model_file = "mhd_1000e.bin"
we_model = fasttext.load_model(fasttext_model_file)


# 1 name
# 2: annotation list

dataset_name = sys.argv[1]

annotation_list = sys.argv[2:]

jobj = []

for anfile in annotation_list:
    with open(anfile, 'r') as f:
        jobj += json.load(f)

# source, target, label, flagged
word_pairs = {}

errors = 0
ambiguous = 0
double = 0
unknown = 0
for jelem in jobj:
    if jelem['annotation'] == 'e':
        errors+= 1
        continue
    label = None
    if jelem['annotation'] == 'm':
        label = 1
    if jelem['annotation'] == 'x':
        label = 0
    if label == None:
        unknown += 1
        continue
    wp= jelem ['word pair']
    wps = " ".join(wp)
    if wps in word_pairs:
        double += 1
        if not label == word_pairs[wps][2]:
            ambiguous += 1
            word_pairs[wps][3]=True
        continue

    vectors = []
    for w in wp:
        normalized = str(normalize_middle_high_german(w, to_lower_all=True))
        ascii_word = Word(normalized).ASCII_encoding()
        vectors.append(we_model[ascii_word])
    word_pairs[wps] = [vectors[0], vectors[1], label, False]
    assert(len(word_pairs[wps][1])==we_len)
    assert(len(word_pairs[wps][0])==we_len)
    
print(f"{ambiguous} ambiguous elements")
print(f"{errors} wrong detections")
print(f"{double} present multiple times")
print(f"{unknown} unknown elements")
sources = []
targets = []
labels = []

for wps in word_pairs:
    wp = word_pairs[wps]
    if wp[3]: 
        continue
    sources.append(wp[0])
    targets.append(wp[1])
    labels.append(wp[2])

sources = np.asarray(sources)
targets = np.asarray(targets)
labels = np.asarray(labels)

print(f"{sum(labels == 0)} nonmetaphors")
print(f"{sum(labels == 1)} metaphors")

with open(dataset_name, 'wb') as f:
    pickle.dump([sources, targets, labels], f)
