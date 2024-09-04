import streamlit as st

from nisqa_check import nisqa_get_metrics
from web_streamlit.utils import (
    show_audio,
    show_nisqas_metrics,
    ref_chennel,
    file_procces,
)
from src.Modules import MVDR
import warnings

warnings.filterwarnings("ignore")


def file_infer(pipline, model):
    st.header("File")
    uploaded_file = st.file_uploader("Download the audio file", type=["wav"])
    st.write(
        """
            NOI - noisiness\n
            COL - coloration\n
            DISC - discontinuty\n
            LOUD - loudness\n
            MOS - mean opinion score
            """
    )
    if uploaded_file is not None:
        waveform, sample_rate = file_procces(uploaded_file, 16000)

        st.write(
            f""" 
                 Original

                 Count channel: {waveform.shape[0]}
                 """
        )

        st.audio(waveform.numpy(), format="audio/wav", sample_rate=sample_rate)
        ref_waveform, ref_channel = ref_chennel(
            waveform[None], pipline.stft
        )  # [B, C, T]
        mvdr = MVDR(ref_channel.item())

        st.write(f"Reference channel: {ref_channel.item()}")
        col1, col2 = st.columns([10, 2])
        with col1:
            show_audio(ref_waveform[0], sample_rate)
        with col2:
            show_nisqas_metrics(nisqa_get_metrics(ref_waveform[0], sample_rate))

        st.write(f"Noise reduction output for the reference channel: {ref_channel.item()}")
        col3, col4 = st.columns([10, 2])
        with col3:
            wave_predict, mask, h_t = pipline.pipline_model(model, ref_waveform)
            show_audio(wave_predict[0].detach(), sample_rate)
        with col4:
            show_nisqas_metrics(nisqa_get_metrics(wave_predict[0], sample_rate))

        st.write("MVDR")
        col5, col6 = st.columns([10, 2])
        with col5:
            predict_mvdr = pipline.pipline_mvdr(mvdr, mask.clamp(0, 1), waveform[None])
            show_audio(predict_mvdr.detach(), sample_rate)
        with col6:
            show_nisqas_metrics(nisqa_get_metrics(predict_mvdr, sample_rate))
