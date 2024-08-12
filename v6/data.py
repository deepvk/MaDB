from torch.utils.data import Dataset, DataLoader
import os
import torchaudio as ta
import torch
import torch.nn.functional as F
import random
import os

def generate_mixture(waveform_clean, waveform_noise, target_snr):
    power_clean_signal = waveform_clean.pow(2).mean()
    power_noise_signal = waveform_noise.pow(2).mean()
    current_snr = 10 * torch.log10(power_clean_signal / power_noise_signal)
    waveform_noise *= 10 ** (-(target_snr - current_snr) / 20)
    return waveform_clean + waveform_noise



class my_dataset(Dataset):
    def __init__(self, root, url, path_demand, path_musan, sec=2):
        
        self._sample_rate = 16000
        
        self.path_demand = path_demand
        self.path_musan = path_musan
        self.path_demand_list = os.listdir(path_demand)
        self.path_musan_list = os.listdir(path_musan)
        self.data = ta.datasets.LIBRISPEECH(root=root, download=False, url=url)
        #self.snr_noise = snr_noise
        
        if sec:
            self.sec = sec*self._sample_rate
        else:
             self.sec = sec
                
        self.path_musan_list.remove('README')
        self.path_musan_list.remove('speech')
    
    def cut(self, wave):
        C, T = wave.shape
        delta = self.sec - T
        if T <= self.sec:
            delta = self.sec - T
            return F.pad(wave, (0, delta))
        else:
            #return wave[:,:self.sec]
            # кропаем случайный сегмент
            delta = T - self.sec
            start = random.randint(0, delta)
            return wave[:, start:start + self.sec]

    def get_musan_noise(self):
        # случайный выбор типа источника
        type_noise = random.choice(self.path_musan_list)
        
        # случайный выбор подтипа источника
        sub_type = random.choice(os.listdir(f"{self.path_musan}/{type_noise}"))
        # случайный выбор источника     
        name_noise = random.choice(os.listdir(f"{self.path_musan}/{type_noise}/{sub_type}"))
        
        noise, sample_rate = ta.load(f"{self.path_musan}/{type_noise}/{sub_type}/{name_noise}")
        
        assert sample_rate == self._sample_rate
        
        return noise
        
    def get_demand_noise(self):
        
        # случайный выбор типа источника
        type_noise = random.choice(self.path_demand_list)
        
        # случайный выбор источника        
        name_noise = random.choice(os.listdir(f"{self.path_demand}/{type_noise}"))
        
        noise, sample_rate = ta.load(f"{self.path_demand}/{type_noise}/{name_noise}")
        
        assert sample_rate == self._sample_rate
        
        return noise
    
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        target = self.cut(self.data[idx][0])

        t_p = random.random()
        snr = random.randint(0, 5)
        if t_p < 0.5:
            noise = self.cut(self.get_demand_noise())
            
        else:
            noise = self.cut(self.get_musan_noise())
        
        assert target.shape[-1] == noise.shape[-1]
        
        return target, generate_mixture(target, noise, snr)