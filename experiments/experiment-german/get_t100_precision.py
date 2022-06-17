# 1 annotated json file

import json
import sys

with open(sys.argv[1], 'r') as f:
    data = json.load(f)

acc = 0.0
for d in data:
    if d['annotation'] == 'm':
        acc += 1.0

acc /= float(len(data))

print(f"accuracy: {acc:.2f}")
