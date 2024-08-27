from typing import Dict, Final, Iterable, List, Optional, Union
from torch.autograd import Function
import torch.nn.functional as F
from torch import Tensor, nn
import torch as th
import torch


class angle(Function):
    """Similar to torch.angle but robustify the gradient for zero magnitude."""

    @staticmethod
    def forward(ctx, x: Tensor):
        ctx.save_for_backward(x)
        return torch.atan2(x.imag, x.real)

    @staticmethod
    def backward(ctx, grad: Tensor):
        (x,) = ctx.saved_tensors
        grad_inv = grad / (x.real.square() + x.imag.square()).clamp_min_(1e-10)
        return torch.view_as_complex(
            torch.stack((-x.imag * grad_inv, x.real * grad_inv), dim=-1)
        )


class Stft(nn.Module):
    def __init__(
        self, n_fft: int, hop: Optional[int] = None, window: Optional[Tensor] = None
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop or n_fft // 4
        if window is not None:
            assert window.shape[0] == n_fft
        else:
            window = torch.hann_window(self.n_fft)
        self.w: torch.Tensor
        self.register_buffer("w", window)

    def forward(self, input: Tensor):
        # Time-domain input shape: [B, *, T]
        t = input.shape[-1]
        sh = input.shape[:-1]
        out = torch.stft(
            input.reshape(-1, t),
            n_fft=self.n_fft,
            hop_length=self.hop,
            window=self.w,
            normalized=True,
            return_complex=True,
        )
        out = out.view(*sh, *out.shape[-2:])
        return out


class SpectralLoss(nn.Module):
    """
    Calculates the L1 loss between the target and predicted magnitudes, and between the target and predicted phases.
    The total loss is the sum of L1 loss for magnitude and L1 loss for phase:
    L1(target magnitude, predicted magnitude) + L1(target phase, predicted phase).
    """

    def __init__(self, n_fft=4096):
        super().__init__()
        self.stft = Stft(n_fft)

    def magnitude_phase(self, x):
        spectr = self.stft(x)
        magnitude = spectr.abs()
        phase = th.atan2(spectr.imag, spectr.real)
        return magnitude, phase

    def forward(self, target, predict):
        target_magnitude, target_phase = self.magnitude_phase(target)
        predict_magnitude, predict_phase = self.magnitude_phase(predict)

        loss = F.l1_loss(target_magnitude, predict_magnitude) + F.l1_loss(
            target_phase, predict_phase
        )

        return loss


class MultiResSpecLoss(nn.Module):
    """
    Determines the discrepancies between the anticipated and actual spectrogram based on Short-Time Fourier Transform (STFT)
    with varying windows, utilizing the Mean Square Error (MSE) loss function for calculation.
    We use Deep-FilterNet loss https://github.com/Rikorose/DeepFilterNet/blob/main/DeepFilterNet/df/loss.py#L95
    """

    gamma: Final[float]
    f: Final[float]
    f_complex: Final[Optional[List[float]]]

    def __init__(
        self,
        n_ffts: Iterable[int],
        gamma: float = 1,
        factor: float = 1,
        f_complex: Optional[Union[float, Iterable[float]]] = None,
    ):
        super().__init__()
        self.gamma = gamma
        self.f = factor
        self.stfts = nn.ModuleDict({str(n_fft): Stft(n_fft) for n_fft in n_ffts})
        if f_complex is None or f_complex == 0:
            self.f_complex = None
        elif isinstance(f_complex, Iterable):
            self.f_complex = list(f_complex)
        else:
            self.f_complex = [f_complex] * len(self.stfts)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        loss = torch.zeros((), device=input.device, dtype=input.dtype)
        for i, stft in enumerate(self.stfts.values()):
            Y = stft(input)
            S = stft(target)
            Y_abs = Y.abs()
            S_abs = S.abs()
            if self.gamma != 1:
                Y_abs = Y_abs.clamp_min(1e-12).pow(self.gamma)
                S_abs = S_abs.clamp_min(1e-12).pow(self.gamma)
            loss += F.mse_loss(Y_abs, S_abs) * self.f  # mse_loss
            if self.f_complex is not None:
                if self.gamma != 1:
                    Y = Y_abs * torch.exp(1j * angle.apply(Y))
                    S = S_abs * torch.exp(1j * angle.apply(S))
                loss += (
                    F.mse_loss(torch.view_as_real(Y), torch.view_as_real(S))
                    * self.f_complex[i]
                )  # mse_loss
        return loss
