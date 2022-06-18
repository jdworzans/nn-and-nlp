import pickle
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

SEQUENCE_LENGTH = 128

class PrusDataset(torch.utils.data.Dataset):
    def __init__(self, tokens, vocab, sequence_length: int = SEQUENCE_LENGTH):
        super().__init__()
        self.tokens = tokens
        self.vocab = vocab
        self.data = torch.tensor([self.vocab[token] for token in self.tokens])
        self.sequence_length = sequence_length

    def __len__(self):
        return (len(self.data) - 1) // self.sequence_length

    def __getitem__(self, idx):
        seq_idx = idx * self.sequence_length
        return (
            self.data[seq_idx:(seq_idx + self.sequence_length)],
            self.data[(seq_idx + 1):(seq_idx + self.sequence_length + 1)]
        )

class PrusModule(pl.LightningModule):
    def __init__(self, data_dir: str = "data", batch_size: int = 64, sequence_length: int = SEQUENCE_LENGTH) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        with (self.data_dir / "tokens_final.pickle").open("rb") as f:
            self.tokens = pickle.load(f)
        self.vocab = torch.load("vocab.pth")
        self.data = torch.tensor([self.vocab[token] for token in self.tokens])
        self.prus_dataset = PrusDataset(self.tokens, self.vocab, self.sequence_length)
    
        self.n_vocab = len(self.vocab)
        self.embedding_dim = 100
        self.lstm_size = 256
        self.num_layers = 2

        self.embedding = nn.Embedding(
            num_embeddings=self.n_vocab,
            embedding_dim=self.embedding_dim,
        )
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.lstm_size,
            num_layers=self.num_layers,
            dropout=0.2,
            batch_first=True,
        )
        self.fc = nn.Linear(self.lstm_size, self.n_vocab)

    def train_dataloader(self):
        return DataLoader(self.prus_dataset, batch_size=self.batch_size, num_workers=8)

    def forward(self, x, state=None):
        embed = self.embedding(x)
        out, new_state = self.lstm(embed, state)
        logits = self.fc(out)
        return logits, new_state

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred, state = self(x)
        loss = F.cross_entropy(y_pred.transpose(1, 2), y)
        self.log("train_loss", loss.item())
        return loss
