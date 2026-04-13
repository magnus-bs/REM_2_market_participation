import numpy as np
import pandas as pd

num_scenarios = 4
hours = 24

data = []

for s in range(1, num_scenarios+1):
    SI = np.random.binomial(1, 0.5, hours)  # 0 or 1
    for h in range(1, hours+1):
        data.append([s, h, SI[h-1]])

df = pd.DataFrame(data, columns=["scenario", "hour", "SI"])
df.to_csv("imbalance_scenarios.csv", index=False)