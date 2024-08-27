import torchaudio.transforms as T
import matplotlib.pyplot as plt
import torchaudio as ta
import librosa.display
import streamlit as st
import numpy as np
import librosa
import torch
import io


def scec_show(st, wave, sample_rate):
    D = librosa.amplitude_to_db(abs(librosa.stft(wave)), ref=np.max)
    fig, ax = plt.subplots(figsize=(15, 5), nrows=1, ncols=1, sharex=True)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    img = librosa.display.specshow(D, sr=sample_rate, ax=ax)

    st.pyplot(fig, bbox_inches="tight", pad_inches=0)


def show_nisqas_metrics(list_metrics):
    st.write(f"NOI: {list_metrics[0]:.2f} ")
    st.write(f"COL: {list_metrics[1]:.2f} ")
    st.write(f"DISC: {list_metrics[2]:.2f} ")
    st.write(f"LOUD: {list_metrics[3]:.2f} ")
    st.write(f"MOS: {list_metrics[4]:.2f} ")


def show_audio(wave, sample_rate):  # [B, 1, T], int
    st_frame = st.empty()
    st.audio(wave.numpy(), format="audio/wav", sample_rate=sample_rate)
    scec_show(st_frame, wave.numpy()[0], sample_rate)


def ref_chennel(samples, stft):
    # поиск опорного канала
    spec_sample = stft.stft(samples)
    power_spec = spec_sample.abs() ** 2

    mean_spec_power = power_spec.mean(dim=(-1, -2))  # беру среднее по F и по T

    ref_idx = torch.argmax(mean_spec_power, dim=1)
    ref_mic = samples[:, ref_idx]  # референс канал

    return ref_mic, ref_idx


def file_procces(file, need_sample):
    audio_bytes = io.BytesIO(file.read())
    waveform, sample_rate = ta.load(audio_bytes)  # [C, T]

    if sample_rate != need_sample:
        resampler = T.Resample(sample_rate, need_sample, dtype=waveform.dtype)
        sample_rate = need_sample
        waveform = resampler(waveform)

    return waveform, sample_rate
