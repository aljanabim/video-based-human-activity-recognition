# %%
import os
import json
import numpy as np
trials = os.listdir('./trials')
scores = []
for trial in trials:
    with open(f"trials/{trial}/trial.json", 'r') as f:
        data = json.loads(f.read())
        score = data['score']
        if score != None:
            scores.append(score)
        else:
            print("The model that did't finish:", trial)

print(np.argmax(np.array([scores])))
best_model = trials[np.argmax(scores)]
print('best model is in this folder:', best_model)
