import torch
import numpy as np
import soundfile as sf

from src.Modules import MVDR, PipeLine
from src.model import Unet_model

import warnings

warnings.filterwarnings("ignore")


class Inference:
    def __init__(self, model, pipline, block_size, stream=False):
        self.model = model
        self.pipline = pipline

        if stream:
            self.block_size = block_size
        else:
            self.block_size = 16000 * 3

        self.mvdr = None

    def find_ref_chennel(self, mic_sound, last_power):  # [B, C, T]
        # find reference channel
        spec_sample = self.pipline.stft.stft(mic_sound)
        power_spec = spec_sample.abs() ** 2

        mean_spec_power = power_spec.mean(dim=(-1, -2))  # average of F and T

        mean_power_last_and_now = (mean_spec_power + last_power) / 2

        ref_idx = torch.argmax(mean_power_last_and_now, dim=1)
        ref_mic = mic_sound[:, ref_idx]  # reference channel

        return ref_mic, mean_power_last_and_now, ref_idx

    def normalize_audio(self, audio_data):
        """Normalization of audio data in the range [-1, 1]."""
        max_abs_value = np.max(np.abs(audio_data))
        if max_abs_value > 0:
            return audio_data / max_abs_value
        return audio_data

    def run(self, input_file, output_file):
        last_power = 0
        first_none = True
        h_t = None
        pred = []

        audio, sr = sf.read(input_file)
        audio = torch.as_tensor(audio.T)

        framesize = self.block_size
        if audio.shape[0] % framesize != 0:
            audio = torch.cat(
                (
                    audio,
                    torch.zeros(audio.shape[0], framesize - audio.shape[0] % framesize),
                ),
                dim=-1,
            )

        audio_spl = torch.split(audio, framesize, dim=-1)
        for au in audio_spl:
            ref, last_power, ref_idx = self.find_ref_chennel(
                au[None].to(torch.float32), last_power
            )
            if first_none and last_power.sum() != 0:
                self.mvdr = MVDR(ref_idx.item())
                first_none = False

            if self.mvdr is not None:
                # self.mvdr.ref_channel = ref_idx.item()
                pred_denoise, mask, h_t = self.pipline.pipline_model(
                    self.model, ref, h_t
                )
                pred_mvdr = self.pipline.pipline_mvdr(
                    self.mvdr, mask.clamp(0, 1), au[None]
                )
                pred.append(pred_mvdr[0].detach())

            else:
                pred.append(data[0][0])

        sf.write(output_file, torch.concat(pred, dim=-1), sr)


if __name__ == "__main__":
    from config.config import InferenceConfig

    conf = InferenceConfig()

    model = Unet_model()
    model.load_state_dict(
        torch.load(conf.model_weigth_path, map_location="cpu")["model_state_dict"]
    )
    model.eval()

    pipline = PipeLine(
        sample_rate=conf.sample_rate,
        window_length=conf.window_length,
        window_shift=conf.window_shift,
    )

    inference = Inference(model.eval(), pipline, conf.backet_size, stream=conf.stream)

    inference.run(conf.in_path_audio, conf.out_path_audio)