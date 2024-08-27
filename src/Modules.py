from torchaudio.transforms import PSD
from typing import Tuple, Optional
import torch.nn.functional as F
import torchaudio as ta
import torch.nn as nn
import torch as th
import torch
import math


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


class PipeLine:
    def __init__(self, sample_rate=16000, window_length=None, window_shift=None):

        if window_length is None:
            window_length = int(0.025 * sample_rate)  # 25 ms window length
        if window_shift is None:
            window_shift = int(0.01 * sample_rate)  # 10 ms window shift

        self.stft = STFT(n_fft=window_length, hop=window_shift)

    def pipline_mvdr(self, mvdr, mask, sample):
        spec_sample = self.stft.stft(sample)

        z = mvdr(spec_sample, mask[0])
        wave_predict = self.stft.istft(z, sample.shape[-1])
        return wave_predict

    def pipline_model(self, model, sample, h_t=None):
        spec_sample = self.stft.stft(sample)
        mag = spec_sample.abs()
        phase = spec_sample.angle()
        mask, h_t = model(torch.log(mag + 1e-5), h_t)

        mag_predict = mag * mask.clamp(0, 1)
        imag = mag_predict * th.sin(phase)
        real = mag_predict * th.cos(phase)

        z = th.complex(real, imag)
        wave_predict = self.stft.istft(
            z, sample.shape[-1]
        )  # .view(B, sample.shape[-1])
        return wave_predict, mask, h_t


class MVDR(nn.Module):

    def __init__(self, ref_channel):
        super().__init__()
        self.psd = PSD()
        self.ref_channel = ref_channel

        self.psd_speech_t = None
        self.psd_noise_t = None
        self.mask_sum_speech = None
        self.mask_sum_noise = None
        self.t = 0

    def reload(self):
        self.psd_speech_t = None
        self.psd_noise_t = None
        self.mask_sum_speech = None
        self.mask_sum_noise = None
        self.t = 0

    def recurent_update(self, psd_n, psd_t, mask, mask_t):
        numerator = mask_t / (mask_t + mask.sum(dim=-1).clamp(min=1e-10))
        denominator = 1 / (mask_t + mask.sum(dim=-1).clamp(min=1e-10))
        psd_n = (
            psd_t * numerator[..., None, None] + psd_n * denominator[..., None, None]
        )

        return psd_n

    def forward(self, mix, mask_speech):
        B, C, F, T = mix.shape

        mask_noise = 1 - mask_speech

        psd_speech = self.psd(mix, mask_speech)
        psd_noise = self.psd(mix, mask_noise)

        ref_vector = torch.zeros((B, C), device=mix.device, dtype=torch.cdouble)
        ref_vector[..., self.ref_channel].fill_(1)

        if self.t == 0:
            self.psd_speech_t = psd_speech
            self.psd_noise_t = psd_noise
            self.mask_sum_speech = mask_speech.sum(dim=-1)
            self.mask_sum_noise = mask_noise.sum(dim=-1)
            self.t = 1
        else:
            self.psd_speech_t = self.recurent_update(
                psd_speech, self.psd_speech_t, mask_speech, self.mask_sum_speech
            )
            self.psd_noise_t = self.recurent_update(
                psd_noise, self.psd_noise_t, mask_noise, self.mask_sum_noise
            )
            self.mask_sum_speech += mask_speech.sum(dim=-1)
            self.mask_sum_noise += mask_noise.sum(dim=-1)
            self.t += 1

        w_vector = ta.functional.mvdr_weights_souden(
            self.psd_speech_t, self.psd_noise_t, ref_vector
        )
        spec_enhanced = ta.functional.apply_beamforming(w_vector, mix)

        return spec_enhanced
