import pandas as pd
import datasets
import transformers
import torch
import random
import numpy as np
import torch.nn as nn
from transformers import AutoTokenizer, XLMRobertaModel
import demoji
import re
from utils.data_utils import handle_emoji
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import time
from transformers import AdamW, get_linear_schedule_with_warmup
from utils.train_utils import train_or_valid, get_model_args
from utils.eval_util import reset_language_score, compute_language_correlation, compute_r
import matplotlib.pyplot as plt
from models.xlm_regressor import MultiLingualModel


# def run():
#     pass

# Set Random Seeds and device
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    dev = "cuda:6"
else:
    dev = "cpu"
DEVICE = torch.device(dev)
print("Device: ", DEVICE)


# Set Up Training
training_args = {
    # Add a linear layer before the final layer
    "add_linear": False,
    #for training on original train set: original
    #for training on translation augmented train set: translated
    # "train_on": "original",
    "train_on": "translate",
    "use_demoji": True,
    "remove_mentions": True,
    "remove_numbers": True,
    "remove_http": True,
    "stratify_split": True,
    "train_zero_shot": True
}

# Load Data
home_path = "/home/bzheng12/SemEval-Intimacy"
if training_args["train_on"] == "original":
    data = pd.read_csv(f"{home_path}/data/twitter_train.csv")
else:
    data = pd.read_csv(f"{home_path}/data/full_translate_all.tsv", sep="\t")

# Strip leading and trailing inverted commas
data["text"] = data["text"].apply(lambda x: x.strip("'"))
if training_args["use_demoji"]:
    # Expand emojis with description using demoji library
    data["text"] = data["text"].apply(lambda x: handle_emoji(x))
if training_args["remove_mentions"]:
    # get rid of mentions @user @whatever
    data["text"] = data["text"].str.replace(r"@[A-Za-z0-9_]+", "", regex=True)
if training_args["remove_numbers"]:
    # remove words containing numbers
    data["text"] = data["text"].str.replace(r"\w*\d\w*", "", regex=True)
if training_args["remove_http"]:
    data["text"] = data["text"].str.replace("\shttps?\s", "", regex=True)

train_data, valid_data = train_test_split(
    data,
    test_size=0.2,
    shuffle=True,
    random_state=0,
    stratify=data["language"] if training_args["stratify_split"] else None,
)
if training_args["train_zero_shot"]:
    train_data = train_data[
        ~train_data["language"].isin(["Korean", "Dutch", "Arabic", "Hindi"])
    ]



# Model Training
model_args = get_model_args()
model = MultiLingualModel(model_args["model_name"], DEVICE)

# Loss and Optimization
total_steps = (len(train_data) // (model_args["batch_size"]) + 1) * model_args["epoch"]
model_args["criterion"] = nn.MSELoss()
model_args["optimizer"] = AdamW(
    model.parameters(), lr=model_args["learning_rate"], eps=1e-8
)
model_args["scheduler"] = get_linear_schedule_with_warmup(
    model_args["optimizer"], num_warmup_steps=0, num_training_steps=total_steps
)

language_score = reset_language_score(valid_data)
# Log Metrics
epoch_train_loss = []
epoch_valid_loss = []
epoch_valid_r = []

# validate the model
valid_loss, valid_y_pred, valid_y = train_or_valid(model_args, 0, model, train_data, valid_data, DEVICE, False)
compute_language_correlation(valid_data, 0, language_score)

# Begin Training
for epoch in range(model_args["epoch"]):

    # Train the model
    train_loss, _, _ = train_or_valid(model_args, epoch, model, train_data, valid_data, DEVICE)
    epoch_train_loss.append(train_loss)
    with torch.cuda.device(DEVICE):
        torch.cuda.empty_cache()
    # validate the model
    valid_loss, valid_y_pred, valid_y = train_or_valid(model_args, epoch, model, train_data, valid_data, DEVICE, False)
    compute_language_correlation(valid_data, epoch, language_score)
    epoch_valid_loss.append(valid_loss)
    epoch_valid_r.append(compute_r)

xi = list(range(model_args["epoch"]))
plt.rcParams["figure.figsize"] = (12, 5)
for lang, score in language_score.items():
    plt.plot(score, label=f"pearson_{lang}")

plt.xticks(xi, range(model_args["epoch"]))
plt.xlabel("Epoch")
plt.ylabel("Pearson's Score")
plt.title("Epoch vs Pearson's Score")
plt.legend(fancybox=True, shadow=True)
plt.savefig(f"images/pearson_score-xlmt_base-6lang.png")
plt.show()


# if __name__ == '__main__':
#     pass
