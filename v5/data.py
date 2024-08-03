from torch.utils.data import Dataset, DataLoader
import os
import torchaudio as ta
import torch
import torch.nn.functional as F

def generate_mixture(waveform_clean, waveform_noise, target_snr):
    power_clean_signal = waveform_clean.pow(2).mean()
    power_noise_signal = waveform_noise.pow(2).mean()
    current_snr = 10 * torch.log10(power_clean_signal / power_noise_signal)
    waveform_noise *= 10 ** (-(target_snr - current_snr) / 20)
    return waveform_clean + waveform_noise

def collate_fn(batch): # list -> [[target, mic, nosie],[...],[...]]
    # batch - список элементов датасета, каждый из которых может быть переменной длины
    # возвращаем батч с данными одинаковой длины по первому измерению
    
    lengths = [bb[0].shape[-1] for bb in batch]
    max_length = max(lengths)
    
    target_list = []
    mic_array_list = []
    noise_list = []
    
    for i, item in enumerate(batch):
        target, mic_array, noise = item
        
        target_list.append(F.pad(target, (0, max_length - lengths[i])))
        mic_array_list.append(F.pad(mic_array, (0, max_length - lengths[i])))
        noise_list.append(F.pad(noise, (0, max_length - lengths[i])))
    
    target_batch = torch.cat(target_list, 0)
    mic_batch = torch.stack(mic_array_list, 0)
    noise_batch = torch.cat(noise_list, 0)
    
    return target_batch, mic_batch, noise_batch

class my_dataset(Dataset):
    def __init__(self, path, sec=None):
        self.path_signal = os.listdir(f'{path}/mic')
        self.path_target = f'{path}/target'
        self.path_mic = f'{path}/mic'
        
        if sec:
            self.sec = sec*16000
        else:
             self.sec = sec
    def __len__(self):
        return len(self.path_signal)
    
    def __getitem__(self, idx):
        path = self.path_signal[idx]
        
        target, sample_rate = ta.load(f'{self.path_target}/{path}')
        mic_array, sample_rate = ta.load(f'{self.path_mic}/{path}')
        
        target = F.pad(target, (0, mic_array.shape[-1] - target.shape[-1]))
        
        if self.sec:
            return target[:,:self.sec], mic_array[:,:self.sec], generate_mixture(target,  torch.rand(1,target.shape[-1]), 10)[:,:self.sec]
        else:
            return  target, mic_array, generate_mixture(target,  torch.rand(1,target.shape[-1]), 10)