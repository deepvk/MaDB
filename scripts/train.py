import torch
import warnings
from tqdm import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchmetrics import ScaleInvariantSignalDistortionRatio
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from src.Modules import PipeLine
from src.model import Unet_model
from src.loss import MultiResSpecLoss
from src.dataset import my_dataset

import logging
import logging.config
import warnings

warnings.filterwarnings("ignore")


class Trainer:
    def __init__(
        self,
        model,
        pipline,
        loss,
        metrics,
        epochs=100,
        device="cuda",
        lr=3e-4,
        T_0=50,
        log_step_path="train_steps.log",
        log_epoch_path="train_epochs.log",
        path_checkpoint="",
    ):

        self.device = device
        self.epochs = epochs
        self.path_checkpoint = path_checkpoint

        self.model = model
        self.model.to(device)

        self.pipline = pipline
        self.si_sdr = metrics

        self.criterion = loss.to(device)

        self.optim = AdamW(self.model.parameters(), lr=lr)

        self.scheduler = CosineAnnealingWarmRestarts(self.optim, T_0=T_0)

        self.losses_epoch_train = []
        self.metrics_epoch_train = []
        self.losses_epoch_valid = []
        self.metrics_epoch_valid = []

        # logger epochs
        self.LOGGER = self.setup_logger("Train", log_epoch_path)

        # Logger step
        self.LOGGER_step = self.setup_logger("Train_step", log_step_path)

        self.min_crit_save = -100

    def setup_logger(
        self, name, log_file, level=logging.DEBUG, formatter_str="%(message)s"
    ):
        logger = logging.getLogger(name)
        logger.setLevel(level)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)

        formatter = logging.Formatter(formatter_str)
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        return logger

    def train_step(self, batch, batch_idx):

        self.optim.zero_grad()

        target, noise = batch
        predictions, *_ = self.pipline.pipline_model(self.model, noise.to(self.device))
        loss = self.criterion(predictions, target.to(self.device))
        metrics = self.si_sdr(predictions.detach().cpu(), target.cpu())
        loss.backward()
        self.optim.step()

        return loss, metrics

    def valid_step(self, batch, batch_idx):

        target, noise = batch

        predictions, *_ = self.pipline.pipline_model(self.model, noise.to(self.device))
        loss = self.criterion(predictions, target.to(self.device))
        metrics = self.si_sdr(predictions.detach().cpu(), target.cpu())

        return loss, metrics

    def save_checkpoint(self, epoch, loss, path):
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optim.state_dict(),
                "loss": loss,
            },
            path,
        )

    def fit(self, train_dataloader, valid_dataloader=None):
        self.LOGGER_step.info("step,train_loss_step,train_metric_step")

        if valid_dataloader is None:
            self.LOGGER.info("epochs,train_loss,train_metric")
        else:
            self.LOGGER.info("epochs,train_loss,train_metric,valid_loss,valid_metric")

        step = 0

        for epoch in tqdm(range(self.epochs)):
            losses_train = []
            metrics_train = []

            losses_val = []
            metrics_val = []

            self.model.train()
            for batch_idx, batch in enumerate(train_dataloader):
                loss, metrics = self.train_step(batch, batch_idx)
                losses_train.append(loss.item())
                metrics_train.append(metrics)

                step += 1

                self.LOGGER_step.info(f"{step},{losses_train[-1]},{metrics_train[-1]}")

            last_train_loss_mean = sum(losses_train) / len(losses_train)
            last_train_metric_mean = sum(metrics_train) / len(metrics_train)
            self.losses_epoch_train.append(last_train_loss_mean)
            self.metrics_epoch_train.append(last_train_metric_mean)

            if valid_dataloader is not None:

                losses_val = []
                metrics_val = []

                self.model.eval()
                for batch_idx, batch in enumerate(valid_dataloader):

                    loss, metrics = self.valid_step(batch, batch_idx)
                    metrics_val.append(metrics)
                    losses_val.append(loss.item())

                last_valid_loss_mean = sum(losses_val) / len(losses_val)
                last_valid_metric_mean = sum(metrics_val) / len(metrics_val)
                self.losses_epoch_valid.append(last_valid_loss_mean)
                self.metrics_epoch_valid.append(last_valid_metric_mean)

                if self.min_crit_save < last_valid_loss_mean:
                    self.min_crit_save = last_valid_loss_mean
                    self.save_checkpoint(
                        epoch,
                        last_train_loss_mean,
                        self.path_checkpoint + "checkpoint_best.pt",
                    )

                self.LOGGER.info(
                    f"{epoch+1},{last_train_loss_mean},{last_train_metric_mean},{last_valid_loss_mean},{last_valid_metric_mean}"
                )
            else:
                if self.min_crit_save < last_train_metric_mean:
                    self.min_crit_save = last_train_metric_mean
                    self.save_checkpoint(
                        epoch,
                        last_train_loss_mean,
                        self.path_checkpoint + "checkpoint_best.pt",
                    )

                self.LOGGER.info(
                    f"{epoch+1},{last_train_loss_mean},{last_train_metric_mean}"
                )

            self.scheduler.step()

        self.save_checkpoint(
            epoch, last_train_loss_mean, self.path_checkpoint + "checkpoint_last.pt"
        )


if __name__ == "__main__":
    from config.config import TrainConfig

    conf = TrainConfig()

    model = Unet_model()

    pipline = PipeLine(
        sample_rate=conf.sample_rate,
        window_length=conf.window_length,
        window_shift=conf.window_shift,
    )

    loss_criterion = MultiResSpecLoss(
        n_ffts=conf.loss_nfft,
        f_complex=conf.c_factor,
        factor=conf.factor,
        gamma=conf.gamma,
    )

    metrics = ScaleInvariantSignalDistortionRatio()

    trainer = Trainer(
        model,
        pipline,
        loss_criterion,
        metrics,
        epochs=conf.epochs,
        device=conf.device,
        lr=conf.lr,
        T_0=conf.T_0,
        log_step_path=conf.log_step_path,
        log_epoch_path=conf.log_epoch_path,
        path_checkpoint=conf.path_checkpoint,
    )

    data_train = my_dataset(
        root=conf.root_train,
        url=conf.url_train,
        path_demand=conf.path_demand,
        path_musan=conf.path_musan,
        sec=conf.sec,
        download=conf.train_download,
    )

    data_valid = my_dataset(
        root=conf.root_valid,
        url=conf.url_valid,
        path_demand=conf.path_demand,
        path_musan=conf.path_musan,
        sec=conf.sec,
        download=conf.valid_download,
    )

    train_dataloader = DataLoader(
        data_train,
        batch_size=conf.batch_size,
        shuffle=conf.shuffle_train,
        drop_last=conf.drop_last_train,
    )

    valid_dataloader = DataLoader(
        data_valid,
        batch_size=conf.batch_size,
        shuffle=conf.shuffle_valid,
        drop_last=conf.drop_last_valid,
    )

    trainer.fit(train_dataloader, valid_dataloader)
