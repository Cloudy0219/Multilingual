import demoji
import re
import torch


def handle_emoji(x):
    x = demoji.replace_with_desc(x)
    return re.sub(r":", " ", x)

def get_data_loader(data, batch_size=16, train=True):
    if train:
        shuffled_data = data.sample(frac=1, random_state=0).reset_index(drop=True)
    else:
        shuffled_data = data
    start = 0
    end = start + batch_size
    data_len = len(shuffled_data)
    while start < data_len:
        sub_data = shuffled_data[start:end]
        start += batch_size
        end = min(start + batch_size, data_len)
        yield sub_data["text"].tolist(), torch.tensor(sub_data["label"].tolist())













