r'''
Author       : PiKaChu_wcg
Date         : 2021-08-18 19:09:14
LastEditors  : PiKachu_wcg
LastEditTime : 2021-08-18 21:55:27
FilePath     : \ifly_pic\train.py
'''
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision import models
class Net(pl.LightningModule):
    def __init__(
        self,
        model,
        batch_size,
        epochs,
        data_path="data/debug",
        lr=1e-4
    ):
        super(Net,self).__init__()
        self.batch_size = batch_size
        self.epochs = epochs
        self.data_path=data_path[:-1] if data_path[-1]=="/" else data_path
        self.lr = lr
        self.model=model
        self.loss_fn=nn.CrossEntropyLoss()
    def forward(
        self,x
    ):
        o=self.model(x)
        return o
    def train_dataloader(self) :
        return DataLoader(
            ImageFolder(
                self.data_path+"/train",
                transform=transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.RandomResizedCrop([224,224]),
                    ]
                )
            ),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=4,
        )
    def val_dataloader(self) :
        return DataLoader(
            ImageFolder(
                self.data_path+"/valid",
                transform=transforms.Compose(
                    [transforms.ToTensor(),transforms.Resize([224,224])]
                )
            ),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=4,
        )    
    def configure_optimizers(self):
        optimizer = SGD(self.parameters(), lr=self.lr, weight_decay=0.001)
        return optimizer
    def training_step(self, batch, batch_nb):
        pred = self.forward(batch[0])
        loss=self.loss_fn(pred,batch[1])
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss
    def validation_step(self, batch, batch_nb):
        pred = self.forward(batch[0])
        loss=self.loss_fn(pred,batch[1])
        return loss
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack(outputs).mean()
        self.log(
            "val_loss",
            avg_loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return {"val_loss": avg_loss}

if __name__=='__main__':
    epochs=10
    output_path="runs/exp1"
    batch_size=2
    data_path="data/debug"
    lr=1e-3
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_path,
        verbose=True,
        every_n_epochs=1,
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )
    trainer = pl.Trainer(
        default_root_dir=output_path,
        gradient_clip_val=1,
        max_epochs=epochs,
        gpus=1,
        callbacks=[checkpoint_callback],
        precision=32,
        progress_bar_refresh_rate=50
    )
    net = Net(
        models.resnet18(pretrained=True),
        batch_size,
        epochs,
        data_path=data_path,
        lr=lr,
    )
    trainer.fit(net)