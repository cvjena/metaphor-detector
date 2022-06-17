import json
import sys
import random

import numpy as np

SEED = 42

np.random.seed(SEED)
random.seed(SEED)

#in: argv[1]
#out: argv[2]
#number: argv[3] 


with open(sys.argv[1], 'r') as f:
    jobj = json.load(f)

annotations = {}

scores = []
for jelem in jobj:
    scores.append(float(jelem['score']))

scores = np.asarray(scores)
print(scores)
indices = np.argsort(scores)
print(indices)
print(scores[indices])

top_100 = []
bottom_50 = []
random_50 = []


jobj_save = []

number = int(sys.argv[3])

idx = len(jobj)-1
for i in range(number):
    jelem = jobj[indices[idx]]
    jobj_save += [jelem]
    idx -= 1

print(jobj[indices[0]])
print(jobj[indices[1]])
print(jobj[indices[2]])
print(jobj[indices[-1]])
print(jobj[indices[-2]])
print(jobj[indices[-3]])

with open(sys.argv[2], 'w') as f:
    f.write(json.dumps(jobj_save, ensure_ascii=False, indent=4))
