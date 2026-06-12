import torch.nn as nn
from models.transform.bert import BERT
from models.transform.config import CONTEXT_LEN, PREDICTION_LEN, HIDDEN_SIZE, BertConfig


class ElectricityBert(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.value_proj = nn.Linear(1, HIDDEN_SIZE).to(device)
        self.day_emb = nn.Embedding(7, HIDDEN_SIZE).to(device)
        self.hour_emb = nn.Embedding(24, HIDDEN_SIZE).to(device)
        self.month_emb = nn.Embedding(13, HIDDEN_SIZE).to(device)
        self.bert = BERT(array_size=CONTEXT_LEN, input_size=HIDDEN_SIZE, cfg=BertConfig(), device=device)
        self.head = nn.Linear(HIDDEN_SIZE, PREDICTION_LEN * 2).to(device)

    def forward(self, values, day_of_week, hour, month):
        emb = (self.value_proj(values.unsqueeze(-1)) +
               self.day_emb(day_of_week) +
               self.hour_emb(hour) +
               self.month_emb(month))
        out = self.bert(emb)[:, -1, :]
        mean, log_std = self.head(out).chunk(2, dim=-1)
        return mean, log_std.exp()