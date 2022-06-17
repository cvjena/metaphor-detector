import json
import sys
import random

import numpy as np

SEED = 42

np.random.seed(SEED)
random.seed(SEED)

#in: argv[1]
#out: argv[2]
#annotations: argv[3:]


with open(sys.argv[1], 'r') as f:
    jobj = json.load(f)

ann_list = sys.argv[3:]
annotations = {}
for  fn in ann_list:
    with open(fn, 'r') as f:
        jobj_ann = json.load(f)
    for jelem in jobj_ann:
        annotations[jelem['id']]=jelem['annotation']

scores = []
for jelem in jobj:
    scores.append(float(jelem['score']))

scores = np.asarray(scores)
#print(scores)
indices = np.argsort(scores)
#print(indices)
#print(scores[indices])

top_100 = []
bottom_50 = []
random_50 = []


jobj_save = []

idx = len(jobj)-1
for i in range(100):
    jelem = jobj[indices[idx]]
    while jelem['id'] in annotations:
        idx -= 1
        jelem = jobj[indices[idx]]
    jobj_save += [jelem]
    annotations[jelem['id']] = jelem['annotation']

idx = 0
for i in range(50):
    jelem = jobj[indices[idx]]
    while jelem['id'] in annotations:
        idx += 1
        jelem = jobj[indices[idx]]
    jobj_save += [jelem]
    annotations[jelem['id']] = jelem['annotation']

#print(jobj[indices[0]])
#print(jobj[indices[1]])
#print(jobj[indices[2]])
#print(jobj[indices[-1]])
#print(jobj[indices[-2]])
#print(jobj[indices[-3]])

np.random.shuffle(indices)
idx = 0
for i in range(50):
    jelem = jobj[indices[idx]]
    while jelem['id'] in annotations:
        idx += 1
        jelem = jobj[indices[idx]]
    jobj_save += [jelem]
    annotations[jelem['id']] = jelem['annotation']


with open(sys.argv[2], 'w') as f:
    f.write(json.dumps(jobj_save, ensure_ascii=False, indent=4))
