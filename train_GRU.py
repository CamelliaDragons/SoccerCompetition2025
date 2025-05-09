
import joblib
import os

import visualizer

from tqdm.notebook import tqdm
import datasets
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

class MinMax:
    def __init__(self, min, max):
        self.min = min
        self.max = max
        if self.min >= self.max:
            raise ValueError("min must be less than max")

    def __call__(self, x):
        return (x - self.min) / (self.max - self.min)

    def inverse(self, x):
        return x * (self.max - self.min) + self.min

    def __repr__(self):
        return f"MinMax({self.min}, {self.max})"

def run(index, NUM_LAYER,HIDDEN_DIM):
    # HYPER PARAM
    MODEL_NAME = f"BiGRU_dataExpanded_{index=}_{NUM_LAYER=}_{HIDDEN_DIM=}"
    BATCH_SIZE = 512
    EPOCHS = 100
    HIDDEN_DIM = HIDDEN_DIM
    NUM_LAYER = NUM_LAYER
    LEARNING_RATE = 0.001

    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
        
    train = np.load("datas/train.npy")
    cols = np.load("datas/cols.npy", allow_pickle=True)
    min_max_d = np.load("datas/min_max_d.npy", allow_pickle=True).item()
    test = np.load("datas/test.npy")

    X_train, y_train = train[:, :20, :], train[:, 20:, :]
    X_test, y_test = test[:, :20, :], test[:, 20:, :]

    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    class EuclideanDistance(torchmetrics.Metric):
        """
        終端誤差を計算する
        """

        def __init__(self, cols, min_max_d, **kwargs):
            super().__init__(**kwargs)
            self.cols = cols
            self.min_max_d = min_max_d
            self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
            self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
            self._mins = torch.tensor(
                [min_max_d[col].min for col in cols], dtype=torch.float32
            )
            self._maxs = torch.tensor(
                [min_max_d[col].max for col in cols], dtype=torch.float32
            )
            self._scale = self._maxs - self._mins

        def update(self, preds, target):
            indices = torch.cat([torch.tensor([0]), torch.arange(5, 201, 9)])
            final_preds = preds[:, -1, :]
            final_target = target[:, -1, :]

            inversed_preds = torch.zeros_like(final_preds)
            inversed_target = torch.zeros_like(final_target)

            mins = self._mins.to(device=device, dtype=final_preds.dtype)
            scale = self._scale.to(device=device, dtype=final_preds.dtype)

            inversed_preds = final_preds.clone()
            inversed_target = final_target.clone()
            n_cols = len(self.cols)
            inversed_preds[:, :n_cols] = final_preds[:, :n_cols] * scale + mins
            inversed_target[:, :n_cols] = final_target[:, :n_cols] * scale + mins

            errors = torch.sqrt(
                (inversed_preds[:, indices] - inversed_target[:, indices]) ** 2
                + (inversed_preds[:, indices + 1] - inversed_target[:, indices + 1]) ** 2
            )
            self.sum += torch.sum(errors)
            self.count += errors.numel()

        def compute(self):
            return self.sum / self.count

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=16
    )
    val_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=16
    )

    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import WandbLogger


    class LitBiGRU(pl.LightningModule):
        def __init__(
            self, input_dim, hidden_dim, output_dim, num_layers, seq_length, lr=0.001
        ):
            super().__init__()
            self.save_hyperparameters()
            self.gru = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
            )
            self.fc = nn.Linear(hidden_dim * 2, output_dim * seq_length)
            self.criterion = nn.MSELoss()
            self.euclidean_distance = EuclideanDistance(cols, min_max_d)
            self.train_losses = []
            self.val_losses = []

        def forward(self, x):
            out, _ = self.gru(x)
            last_out = out[:, -1, :]
            output = self.fc(last_out)
            return output.view(-1, 30, self.hparams.output_dim)

        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = self.criterion(y_hat, y)
            self.log("train_loss", loss, on_step=False, on_epoch=True)
            self.log(
                "train_euclidean_distance",
                self.euclidean_distance(y_hat, y),
                on_step=False,
                on_epoch=True,
            )
            return loss

        def validation_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = self.criterion(y_hat, y)
            self.log("val_loss", loss, on_step=False, on_epoch=True)
            self.log(
                "val_euclidean_distance",
                self.euclidean_distance(y_hat, y),
                on_step=False,
                on_epoch=True,
            )
            return loss

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
            return optimizer


    checkpoint_callback = ModelCheckpoint(
        monitor="val_euclidean_distance",
        dirpath=f"checkpoints/{MODEL_NAME}",
        filename="bilstm-{epoch:02d}-{val_euclidean_distance:.4f}",
        save_top_k=3,
        mode="min",
    )

    csv_logger = pl.loggers.CSVLogger(
        save_dir="logs/",
        name=MODEL_NAME,
        version=0,
    )

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="gpu",
        devices=1,
        callbacks=[checkpoint_callback],
        logger=csv_logger,
        enable_progress_bar=False,
        gradient_clip_val=0.5,
    )

    input_dim = X_train.shape[2]
    hidden_dim = HIDDEN_DIM
    output_dim = y_train.shape[2]
    seq_length = y_train.shape[1]
    num_layers = NUM_LAYER
    learning_rate = LEARNING_RATE

    model = LitBiGRU(
        input_dim, hidden_dim, output_dim, num_layers, seq_length, lr=learning_rate
    )
    trainer.fit(model, train_loader, val_loader)

    # モデルの保存
    import time

    best_model_path = checkpoint_callback.best_model_path
    model = LitBiGRU.load_from_checkpoint(best_model_path)
    file_name = f"{MODEL_NAME}_{time.strftime('%Y%m%d%H%M%S')}.pth"

    os.makedirs(f"models/{MODEL_NAME}", exist_ok=True)
    torch.save(model.state_dict(), f"models/{MODEL_NAME}/{file_name}")

d = [{
    "index": 1,
    "NUM_LAYER": 1,
    "HIDDEN_DIM": 128,
},
{
    "index": 2,
    "NUM_LAYER": 1,
    "HIDDEN_DIM": 256,
},
{
    "index": 3,
    "NUM_LAYER": 1,
    "HIDDEN_DIM": 512,
},
{
    "index": 4,
    "NUM_LAYER": 2,
    "HIDDEN_DIM": 128,
},
{
    "index": 5,
    "NUM_LAYER": 2,
    "HIDDEN_DIM": 256,
},
{
    "index": 6,
    "NUM_LAYER": 2,
    "HIDDEN_DIM": 512,
},
{
    "index": 7,
    "NUM_LAYER": 4,
    "HIDDEN_DIM": 128,
},
{
    "index": 8,
    "NUM_LAYER": 4,
    "HIDDEN_DIM": 256,
},
{
    "index": 9,
    "NUM_LAYER": 4,
    "HIDDEN_DIM": 512
},
{
    "index": 10,
    "NUM_LAYER": 6,
    "HIDDEN_DIM": 128,
},
{
    "index": 11,
    "NUM_LAYER": 6,
    "HIDDEN_DIM": 256,
},
{
    "index": 12,
    "NUM_LAYER": 6,
    "HIDDEN_DIM": 512
},
{
    "index": 13,
    "NUM_LAYER": 8,
    "HIDDEN_DIM": 128,
},
{
    "index": 14,
    "NUM_LAYER": 8,
    "HIDDEN_DIM": 256,
},
{
    "index": 15,
    "NUM_LAYER": 8,
    "HIDDEN_DIM": 512
}
]

for i in d:
    run(**i)