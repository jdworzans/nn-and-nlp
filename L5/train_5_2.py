from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.utilities.cli import LightningCLI
from scipy import sparse
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchtext import vocab


def process(word):
    result = []
    for i in [2, 3, 4]:
        result.append(("____" + word)[-i:] + "$")
        result.append("$" + (word + "____")[:i])
    return result


class POSDataset(torch.utils.data.Dataset):
    def __init__(self, filepath, words_vocab=None, label_vocab=None):
        super().__init__()
        self.filepath = filepath
        self.df = pd.read_csv(filepath, sep=" ", names=["word", "labels"])
        if words_vocab is None:
            self.words_vocab = vocab.build_vocab_from_iterator(map(process, self.df["word"]))
            self.words_vocab.set_default_index(-1)
        else:
            self.words_vocab = words_vocab

        if label_vocab is None:
            self.label_vocab = vocab.build_vocab_from_iterator(self.df["labels"].str.split("_"))
            self.label_vocab.set_default_index(-1)
        else:
            self.label_vocab = label_vocab

        self.X, self.y = self.get_X_y(self.df, self.words_vocab, self.label_vocab)

    @staticmethod
    def get_X_y(df, words_vocab, labels_vocab):
        X, y = sparse.lil_array((len(df), len(words_vocab))), sparse.lil_array((len(df), len(labels_vocab)))
        for word_idx, (subwords, labels) in enumerate(zip(map(process, df["word"]), df["labels"].str.split("_"))):
            X[word_idx, [words_vocab[subword] for subword in subwords]] = 1
            y[word_idx, [labels_vocab[label] for label in labels]] = 1
        return torch.tensor(X.toarray()).float(), torch.tensor(y.toarray()).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class POSModule(pl.LightningModule):
    def __init__(self, data_dir: str = "data", batch_size: int = 64) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.train_dataset = POSDataset(Path(data_dir) / "english_tags_dev.txt")
        self.test_dataset = POSDataset(Path(data_dir) / "english_tags_test.txt", self.train_dataset.words_vocab, self.train_dataset.label_vocab)
        val_length = int(0.8 * len(self.train_dataset))
        self.train_dataset, self.val_dataset = random_split(self.train_dataset, [len(self.train_dataset) - val_length, val_length])

        self.loss = nn.BCELoss()
        self.model = nn.Sequential(
            # nn.Linear(len(self.test_dataset.words_vocab), 64),
            # nn.LeakyReLU(inplace=True),
            # nn.Linear(64, 64),
            # nn.LeakyReLU(inplace=True),
            # # nn.Linear(64, 64),
            # # nn.LeakyReLU(inplace=True),
            # nn.Linear(64, len(self.test_dataset.label_vocab)),
            nn.Linear(len(self.test_dataset.words_vocab), len(self.test_dataset.label_vocab)),
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=8)

    def forward(self, x):
        logits = self.model(x)
        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def step(self, batch, batch_idx, prefix):
        x, y = batch
        y_pred = self(x)
        loss = self.loss(torch.sigmoid(y_pred), y)
        accuracy = (torch.sigmoid(y_pred).round() == y).all(-1).float().mean()
        self.log(f"{prefix}_loss", loss.item())
        self.log(f"{prefix}_acc", accuracy.item())
        return loss, accuracy

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "train")[0]

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "val")[0]

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "test")[0]

if __name__ == "__main__":
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="pos-model-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )
    early_stopping = EarlyStopping("val_loss", patience=10)

    cli = LightningCLI(
        POSModule,
        trainer_defaults={'gpus': 1, 'callbacks': [checkpoint_callback, early_stopping], 'max_epochs': 100},
        seed_everything_default=1234,
        save_config_overwrite=True
    )
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    cli.trainer.test(ckpt_path="best")
