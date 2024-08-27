from torch.utils.data import Dataset
import torch.nn.functional as F
import torchaudio as ta
import random
import torch
import os


class my_dataset(Dataset):
    def __init__(
        self,
        root,
        url,
        path_demand,
        path_musan,
        sec=2,
        sample_rate=16000,
        min_snr=0,
        max_snr=5,
        download=False,
    ):

        self._sample_rate = sample_rate
        self.min_snr = min_snr
        self.max_snr = max_snr

        self.path_demand = path_demand
        self.path_musan = path_musan
        self.path_demand_list = os.listdir(path_demand)
        self.path_musan_list = os.listdir(path_musan)
        self.data = ta.datasets.LIBRISPEECH(root=root, download=download, url=url)
        self.sec = sec * self._sample_rate

        self.path_musan_list.remove("README")
        self.path_musan_list.remove("speech")

    def cut(self, wave):
        C, T = wave.shape
        delta = self.sec - T
        if T <= self.sec:
            delta = self.sec - T
            return F.pad(wave, (0, delta))
        else:
            # return wave[:,:self.sec]
            # take a random segment
            delta = T - self.sec
            start = random.randint(0, delta)
            return wave[:, start : start + self.sec]

    def get_musan_noise(self):
        # random selection of the source type
        type_noise = random.choice(self.path_musan_list)

        # random selection of the source subtype
        sub_type = random.choice(os.listdir(f"{self.path_musan}/{type_noise}"))
        # random source selection
        name_noise = random.choice(
            os.listdir(f"{self.path_musan}/{type_noise}/{sub_type}")
        )

        noise, sample_rate = ta.load(
            f"{self.path_musan}/{type_noise}/{sub_type}/{name_noise}"
        )

        assert sample_rate == self._sample_rate

        return noise

    def get_demand_noise(self):

        # random selection of the source type
        type_noise = random.choice(self.path_demand_list)

        # random source selection
        name_noise = random.choice(os.listdir(f"{self.path_demand}/{type_noise}"))

        noise, sample_rate = ta.load(f"{self.path_demand}/{type_noise}/{name_noise}")

        assert sample_rate == self._sample_rate

        return noise

    def generate_mixture(self, waveform_clean, waveform_noise, target_snr):
        power_clean_signal = waveform_clean.pow(2).mean()
        power_noise_signal = waveform_noise.pow(2).mean()
        current_snr = 10 * torch.log10(power_clean_signal / power_noise_signal)
        waveform_noise *= 10 ** (-(target_snr - current_snr) / 20)
        return waveform_clean + waveform_noise

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        target = self.cut(self.data[idx][0])

        t_p = random.random()
        snr = random.randint(self.min_snr, self.max_snr)
        if t_p < 0.5:
            noise = self.cut(self.get_demand_noise())

        else:
            noise = self.cut(self.get_musan_noise())

        assert target.shape[-1] == noise.shape[-1]

        return target, self.generate_mixture(target, noise, snr)
