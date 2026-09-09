import os
import random

from datasets import load_dataset

dataset = load_dataset("yuvalkirstain/pickapic_v1", num_proc=16)

text_dataset = dataset["train"].select_columns(["caption"])
unique_dataset = text_dataset.unique("caption")
unique_dataset = [s for s in unique_dataset if s.count(" ") >= 5]

random.shuffle(unique_dataset)

test_size = 2048
unique_text_dataset = unique_dataset[:test_size]
train_dataset = unique_dataset[test_size:]

os.makedirs("datasets/pickscore", exist_ok=True)
with open("datasets/pickscore/train.txt", "w", encoding="utf-8") as file:
    for line in train_dataset:
        file.write(line + "\n")

with open("datasets/pickscore/test.txt", "w", encoding="utf-8") as file:
    for line in unique_text_dataset:
        file.write(line + "\n")
