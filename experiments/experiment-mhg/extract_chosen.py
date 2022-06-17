import json
import sys
import random

import numpy as np

SEED = 42

np.random.seed(SEED)
random.seed(SEED)


with open(sys.argv[1], 'r') as f:
    jobj = json.load(f)


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

top_100_indices = indices[len(indices)-101:]
bottom_50_indices = indices[:50]
random_50_indices = np.random.choice(indices[50:len(indices)-101], 50, replace=False)
#print(top_100_indices)
#print(bottom_50_indices)
#print(random_50_indices)

jobj_save = []

for i in range(100):
    jobj_save.append(jobj[top_100_indices[i]])
for i in range(50):
    jobj_save.append(jobj[bottom_50_indices[i]])
for i in range(50):
    jobj_save.append(jobj[random_50_indices[i]])

with open(sys.argv[2], 'w') as f:
    f.write(json.dumps(jobj_save, ensure_ascii=False, indent=4))
