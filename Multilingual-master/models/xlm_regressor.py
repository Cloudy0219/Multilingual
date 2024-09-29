from torch import nn
from transformers import AutoTokenizer
# from transformers import XLMRobertaModel
from transformers import AutoModel



# Define Model
class MultiLingualModel(nn.Module):
    def __init__(self, model_name, DEVICE, add_linear=False):
        super().__init__()
        self.device = DEVICE
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=128, cache_dir="/qyzheng0219/lms/caches")
        self.model = AutoModel.from_pretrained(
            model_name, output_attentions=False, output_hidden_states=False, cache_dir="/qyzheng0219/lms/caches"
        ).to(DEVICE)
        self.add_linear = add_linear
        if self.add_linear:
            self.linear = nn.Sequential(nn.Dropout(0.2), nn.ReLU(), nn.Linear(768, 768)).to(
                DEVICE
            )
        self.regressor = nn.Sequential(
            nn.Dropout(0.1), nn.ReLU(), nn.Linear(768, 1)
        ).to(DEVICE)

    def forward(self, sentences):
        encoded_input = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=256,
        ).to(self.device)
        out = self.model(**encoded_input)[1]
        if self.add_linear:
            out = self.linear(out)
        out = self.regressor(out)
        return out, encoded_input












