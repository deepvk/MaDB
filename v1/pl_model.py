import torch
import torch.nn.functional as F
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import ScaleInvariantSignalNoiseRatio
import torch.nn.functional as F
import torch.nn as nn
from Modules import STFT, MVDR, mask_estimator
import torch

class SpectralLoss(nn.Module):

    def __init__(self):
        super().__init__()
    
    def forward(self, target_spec, predict_spec):
        target = torch.log(target_spec.abs()+1e-5)
        predict = torch.log(predict_spec.abs()+1e-5)
        
        return F.mse_loss(target, predict)
    
    
class PL_model(pl.LightningModule):
    def __init__(self, mic=1):
        super().__init__()
        self.model_mask = mask_estimator(1)
        fs = 16000  # Sampling frequency
        window_length = int(0.025 * fs)  # 25 ms window length
        window_shift = int(0.01 * fs)  # 10 ms window shift
        self.stft = STFT(n_fft = window_length, hop=window_shift)
        
        self.criterion = SpectralLoss()
        self.metric = ScaleInvariantSignalNoiseRatio()
    def forward(self, x):
        mvdr = MVDR(0)
        spec = self.stft.stft(x)
        mask = self.model_mask(spec)
        predict_spec = mvdr(spec, mask)
        return predict_spec, self.stft.istft(predict_spec, x.shape[-1]) 
    
    def training_step(self, batch, batch_idx):
        target_signal, mic_array_signal = batch
        
        predict_spec, predict_signal = self.forward(mic_array_signal)
        loss = self.criterion(self.stft.stft(target_signal), predict_spec)
        
        metric = self.metric(target_signal, predict_signal.view(target_signal.size()))
        self.log_dict(
            {
                "train_loss": loss,
                "si_snr_train": metric,
            },
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        target_signal, mic_array_signal = batch
        
        predict_spec, predict_signal = self.forward(mic_array_signal)
        loss = self.criterion(self.stft.stft(target_signal), predict_spec)
        
        metric = self.metric(target_signal, predict_signal.view(target_signal.size()))
        
        self.log_dict(
            {
                "valid_loss": loss,
                "si_snr_valid": metric,
            },
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=3e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=40
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "valid_loss",
        }