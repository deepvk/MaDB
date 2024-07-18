from typing import Tuple, Optional, Union
import math
import torch as th
import torch
import torch.nn as nn
import torchaudio as ta
from torchaudio.transforms import PSD
import torch.nn.functional as F

class STFT:
    def __init__(self, n_fft: int = 4096, hop=1, pad: int = 0):
        self.n_fft = n_fft
        self.pad = pad
        self.hop_length = hop

    def __pad1d(
        self,
        x: torch.Tensor,
        paddings: Tuple[int, int],
        mode: str = "constant",
        value: float = 0.0,
    ):
        """
        Tiny wrapper around F.pad, designed to allow reflect padding on small inputs.
        If the input is too small for reflect padding, we first add extra zero padding to the right before reflection occurs.
        """
        x0 = x
        length = x.shape[-1]
        padding_left, padding_right = paddings
        if mode == "reflect":
            max_pad = max(padding_left, padding_right)
            if length <= max_pad:
                extra_pad = max_pad - length + 1
                extra_pad_right = min(padding_right, extra_pad)
                extra_pad_left = extra_pad - extra_pad_right
                paddings = (
                    padding_left - extra_pad_left,
                    padding_right - extra_pad_right,
                )
                x = F.pad(x, (extra_pad_left, extra_pad_right))
        out = F.pad(x, paddings, mode, value)
        assert out.shape[-1] == length + padding_left + padding_right
        assert (out[..., padding_left : padding_left + length] == x0).all()
        return out

    def _spec(self, x: torch.Tensor):
        *other, length = x.shape
        x = x.reshape(-1, length)
        z = th.stft(
            x,
            self.n_fft * (1 + self.pad),
            self.hop_length or self.n_fft // 4,
            window=th.hann_window(self.n_fft).to(x),
            win_length=self.n_fft,
            normalized=True,
            center=True,
            return_complex=True,
            pad_mode="reflect",
        )
        _, freqs, frame = z.shape
        return z.view(*other, freqs, frame)

    def _ispec(self, z: torch.Tensor, length: int):
        *other, freqs, frames = z.shape
        n_fft = 2 * freqs - 2
        z = z.view(-1, freqs, frames)
        win_length = n_fft // (1 + self.pad)
        is_mps = z.device.type == "mps"
        if is_mps:
            z = z.cpu()
        z = th.istft(
            z,
            n_fft,
            self.hop_length,
            window=th.hann_window(win_length).to(z.real),
            win_length=win_length,
            normalized=True,
            length=length,
            center=True,
        )
        _, length = z.shape
        return z.view(*other, length)

    def stft(self, x: torch.Tensor):
        hl = self.hop_length
        x0 = x  # noqa
        le = int(math.ceil(x.shape[-1] / self.hop_length))
        pad = hl // 2 * 3
        x = self.__pad1d(
            x, (pad, pad + le * self.hop_length - x.shape[-1]), mode="reflect"
        )
        z = self._spec(x)[..., :-1, :]
        z = z[..., 2 : 2 + le]
        return z

    def istft(self, z: torch.Tensor, length: int = 0, scale: Optional[int] = 0):
        hl = self.hop_length // (4**scale)
        z = F.pad(z, (0, 0, 0, 1))
        z = F.pad(z, (2, 2))
        pad = hl // 2 * 3
        le = hl * int(math.ceil(length / hl)) + 2 * pad

        x = self._ispec(z, length=le)

        x = x[..., pad : pad + length]
        return x
    

class MVDR(nn.Module):
    
    def __init__(self, ref_chennel):
        super().__init__()
        self.psd = PSD()
        self.ref_channel = ref_chennel
        
        self.psd_speech_t = 0
        self.psd_noise_t = 0
        self.mask_sum_speech = 0
        self.mask_sum_noise = 0
        self.t = 0
        
    def reload(self):
        self.psd_speech_t = 0
        self.psd_noise_t = 0
        self.mask_sum_speech = 0
        self.mask_sum_noise = 0
        self.t = 0
        
    def recurent_update(self, psd_n, psd_t, mask, mask_t): #snatch, still # psd_n - new
        # psd_n - new
        # psd_t - old, story, agr
        # mask - new
        # mast_t - old, story, agr
        
        numerator = mask_t / (mask_t + mask.sum(dim=-1))
        
        denominator = 1 / (mask_t + mask.sum(dim=-1))
        psd_n = psd_t * numerator[..., None, None] + psd_n * denominator[..., None, None]
        
        return psd_n
    
    def forward(self, mix, mask_speech):
        B, C, F, T = mix.shape
        
        mask_noise = 1 - mask_speech
        
        psd_speech = self.psd(mix, mask_speech)
        psd_noise = self.psd(mix, mask_noise)
        
        ref_vector = torch.zeros((B, C), device=mix.device, dtype=torch.cdouble)  # (..., channel)
        ref_vector[..., self.ref_channel].fill_(1)
        
        
        if self.t == 0: # t_0
            self.psd_speech_t = psd_speech
            self.psd_noise_t = psd_noise
            self.mask_sum_speech = mask_speech.sum(dim=-1)
            self.mask_sum_noise = mask_noise.sum(dim=-1)
            self.t = 1
        else:
            self.psd_speech_t = self.recurent_update(psd_speech, self.psd_speech_t, mask_speech, self.mask_sum_speech)
            self.psd_speech_n = self.recurent_update(psd_noise, self.psd_noise_t, mask_noise, self.mask_sum_noise)
            self.mask_sum_speech = self.mask_sum_speech + mask_speech.sum(dim=-1)
            self.mask_sum_noise = self.mask_sum_noise + mask_noise.sum(dim=-1)
            self.t += 1
            
        w_vetor = ta.functional.mvdr_weights_souden(self.psd_speech_t, self.psd_noise_t, ref_vector)
        spec_enhanced = ta.functional.apply_beamforming(w_vetor, mix)

        return spec_enhanced
    


class mask_estimator(nn.Module):
    def __init__(self, n):
        super().__init__()
        
        self.lstm = nn.LSTM(input_size=200, 
                            hidden_size=256, 
                            num_layers=1, batch_first=True)
        
        self.fc1 = nn.Sequential(
            nn.Linear(256, 513),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(513, 513),
            nn.ReLU()
        )
        self.fc3 =  nn.Sequential(
            nn.Linear(513, 200),
            nn.Sigmoid()
        )
        
    def forward(self, spec):
        log_magnitude = torch.log(spec.abs() + 1e-5)
        mean_log_magnitude = log_magnitude.mean(dim=1)
        
        x = self.lstm(mean_log_magnitude.permute(0,2,1))[0]
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x.permute(0,2,1)