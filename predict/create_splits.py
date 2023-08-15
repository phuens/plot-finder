import os 
import json

train_dir = "/home/phn501/plot-finder/dataset/range_wise/train"
valid_dir = "/home/phn501/plot-finder/dataset/range_wise/validation"
test_dir  = "/home/phn501/plot-finder/dataset/range_wise/test"


keys = {"test_keys": [], "train_keys":[] }
for file in os.listdir(train_dir): 
    file = file.replace(".csv", "")
    keys["train_keys"].append(str(file))

for file in os.listdir(valid_dir): 
    file = file.replace(".csv", "")
    keys["test_keys"].append(str(file))

with open("/home/phn501/plot-finder/dataset/extracted_feature/train_val.json", "w") as file: 
    json.dump(keys, file)

keys = {"test_keys": []}
for file in os.listdir(test_dir):
    file = file.replace(".csv", "")
    keys["test_keys"].append(str(file))

with open("/home/phn501/plot-finder/dataset/extracted_feature/test.json", "w") as file: 
    json.dump(keys, file)
