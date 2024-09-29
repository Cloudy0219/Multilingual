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
from utils.eval_util import reset_language_score, compute_language_correlation, compute_r
import matplotlib.pyplot as plt
from models.xlm_regressor import MultiLingualModel
import argparse
from utils.train_utils import Trainer


def run(args):
    # Set Random Seeds and device
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.device == -1:
        device = torch.device("cpu")
    elif args.device == -2:
        device = torch.device("cuda")
    else:
        device = torch.device("cuda:" + str(args.device))

    # Load Dataset
    data = pd.read_csv(args.train_data_path)
    # Strip leading and trailing inverted commas
    data["text"] = data["text"].apply(lambda x: x.strip("'"))
    if args.remove_emoji:
        data["text"] = data["text"].apply(lambda x: handle_emoji(x))
    if args.remove_mention:
        data["text"] = data["text"].str.replace(r"@[A-Za-z0-9_]+", "", regex=True)
    if args.remove_numbers:
        data["text"] = data["text"].str.replace(r"\w*\d\w*", "", regex=True)
    if args.remove_http:
        data["text"] = data["text"].str.replace("\shttps?\s", "", regex=True)

    # Need more processing about the split (dev should not be augmented)
    train_data, valid_data = train_test_split(
        data,
        test_size=0.2,
        shuffle=True,
        random_state=0,
        stratify=data["language"] if args.stratify_split else None,
    )
    if args.train_zero_shot:
        train_data = train_data[
            ~train_data["language"].isin(["Korean", "Dutch", "Arabic", "Hindi"])
        ]

    # Set Model
    model = MultiLingualModel(args.model_name, device, args.linear_layer)

    # Set loss and optimization
    total_steps = (len(train_data) // (args.batch_size) + 1) * args.epoch
    criterion = nn.MSELoss()
    optimizer = AdamW(
        model.parameters(), lr=args.learning_rate, eps=1e-8
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    # Set up evaluation
    language_score = reset_language_score(valid_data)
    epoch_train_loss = []
    epoch_valid_loss = []
    epoch_valid_r = []

    trainer = Trainer(args, criterion, optimizer, scheduler, model, train_data, valid_data, device)
    # Begin Training

    for epoch in range(args.epoch):
        # Train for one loop
        train_loss, _, _ = trainer.single_train_loop(epoch)
        epoch_train_loss.append(train_loss)
        with torch.cuda.device(trainer.device):
            torch.cuda.empty_cache()

        # Evaluation
        valid_loss, valid_y_pred, valid_y = trainer.evaluate(epoch)
        compute_language_correlation(valid_data, epoch, language_score)
        epoch_valid_loss.append(valid_loss)
        epoch_valid_r.append(compute_r)

    xi = list(range(args.epoch))
    plt.rcParams["figure.figsize"] = (12, 5)
    for lang, score in language_score.items():
        plt.plot(score, label=f"pearson_{lang}")

    plt.xticks(xi, range(args.epoch))
    plt.xlabel("Epoch")
    plt.ylabel("Pearson's Score")
    plt.title("Epoch vs Pearson's Score")
    plt.legend(fancybox=True, shadow=True)
    # plt.savefig(f"images/pearson_score-xlmt_base-6lang.png")
    plt.show()


# # validate the model
# valid_loss, valid_y_pred, valid_y = train_or_valid(args, 0, model, train_data, valid_data, device, False)
# compute_language_correlation(valid_data, 0, language_score)


if __name__ == '__main__':
    # Set Parser to collect hyper-parameters
    parser = argparse.ArgumentParser()
    # Add paramters
    parser.add_argument('--device', type=int, default=-1, help='Device for running experience. If use CPU, set -1; if use GPU, set the device number')
    parser.add_argument('--seed', type=int, default=0, help='')
    parser.add_argument('--linear_layer', type=bool, default=False, help='')
    parser.add_argument('--train_data_path', type=str, default="/home/qyzheng0219/SemEval-Intimacy/data/twitter_train.csv", help='')
    parser.add_argument('--test_data_path', type=str, default="/home/qyzheng0219/SemEval-Intimacy/data/semeval_test.csv", help='')
    parser.add_argument('--save_path', type=str, default="/qyzheng0219/regressor_checkpoints", help='')
    parser.add_argument('--prediction_path', type=str, default="/qyzheng0219/regressor_predictions/", help='')
    parser.add_argument('--remove_emoji', type=bool, default=True, help='Whether remove emoji')
    parser.add_argument('--remove_mention', type=bool, default=True, help='Remove mention of user ids')
    parser.add_argument('--remove_numbers', type=bool, default=True, help='Remove words with number')
    parser.add_argument('--remove_http', type=bool, default=True, help='Remove http link')
    parser.add_argument('--stratify_split', type=bool, default=True, help='Dataset split')
    parser.add_argument('--train_zero_shot', type=bool, default=True, help='')
    parser.add_argument('--batch_size', type=int, default=16, help='')
    parser.add_argument('--epoch', type=int, default=8, help='')
    parser.add_argument('--learning_rate', type=float, default=0.00001, help='')
    parser.add_argument('--model_name', type=str, default="cardiffnlp/twitter-xlm-roberta-base", help='')
    # Parse Parameters
    args = parser.parse_args()
    print(args)

    run(args)

