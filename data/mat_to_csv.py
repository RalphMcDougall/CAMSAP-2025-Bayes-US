import csv
import os
from scipy.io import loadmat 

# Convert the data provided at: https://alumni.media.mit.edu/~ayush/USF-RR/Algorithms.html#us-alg
# with direct link: https://alumni.media.mit.edu/~ayush/USF-RR/_downloads/5fa7ddc47c4e38e1332b5b3555b2f549/8K_code_data.zip
# into a CSV format that can be used here.

print("Running from:", os.getcwd())

DIR_PATH = "data"

data = loadmat(f"{DIR_PATH}/Data_8k.mat")

xs, ys = data["x"], data["y"]

sanitised_data = []

for [x, y] in zip(xs, ys):
    x, y = x[0], y[0]
    sanitised_data.append([x, y])

print(f"Saving to: {DIR_PATH}/data.csv")
with open(f"{DIR_PATH}/data.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["x", "y"])
    w.writerows(sanitised_data)