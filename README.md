 [![githubio](https://img.shields.io/badge/GitHub.io-Audio_Samples-blue?logo=Github&style=flat-square)](https://maks00170.github.io/beam_github_page/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Beamformer 🗣  🎙️🎙️🎙️🎙️

Beamformer MVDR (Minimum Variance Distortionless Response)

## Structure
- [`config`](./config) ‒ configuration file
- [`scripts`](./scripts) ‒ different useful scripts, e.g. inference and training
- [`src`](./src) ‒ main source code
- [`web_streamlit`](./web_streamlit) ‒ source code for the inference in web service

## Installation

_(Optional)_ Create new venv or conda env

Then just `pip install -r requirements.txt`

Note that there may be some problems with `torch` installation. If so, follow official [PyTorch instructions](https://pytorch.org/get-started/locally/)
### Docker
#### To set up environment with Docker

If you don't have Docker installed, please follow the links to find installation instructions for [Ubuntu](https://docs.docker.com/desktop/install/linux-install/), [Mac](https://docs.docker.com/desktop/install/mac-install/) or [Windows](https://docs.docker.com/desktop/install/windows-install/).

Build docker image:

    docker build -t beamform .

Run docker image:

    bash run_docker.sh

## Data
We used the following datasets:
* For training: librispeech-clean-100.
* For validation: librispeech-test-clean.
* Noise datasets: MUSAN and DEMAND.

[![LIBRISPEECH dataset](https://img.shields.io/badge/LIBRISPEECH%20-E0FFFF)](https://www.openslr.org/12)
[![MUSAN dataset](https://img.shields.io/badge/MUSAN%20-4169E1)](https://www.openslr.org/17/)
[![Demand dataset](https://img.shields.io/badge/Demand%20-CD5C5C)](https://www.kaggle.com/datasets/chrisfilo/demand)

## Training
1. Configure train arguments in `config/config.py`.
2. Run:      

        python -m scripts.train

## Inference
1. Configure inference arguments in `config/config.py`.
2. Run: 
 
        python -m scripts.inference
      
## Web Inference
The service implements the calculation of the NISQA-s metric
1. Clone repository

        git clone https://github.com/deepvk/NISQA-s.git ./web-streamlit/NISQAs

2. Change [web_streamlit/NISQAs/config/nisqa_s.yaml](/web_streamlit/NISQAs/config/nisqa_s.yaml#L72):  

``` yaml
ckp: src/weights/nisqa_s.tar --> web_streamlit/NISQAs/src/weights/nisqa_s.tar
```

3. Run: 

        python -m streamlit run web_streamlit/app.py

## Metrics
Used https://github.com/deepvk/NISQA-s

NOI - noisiness

COL - coloration

DISC - discontinuty

LOUD - loudness

MOS - mean opinion score


We have generated a dataset (100 tracks) using pyroomacustics with various configurations. 
For the noise source, we used noise from the demand data

| Method | NOI | COL | DISC | LOUD | MOS |
|-------------|-----|-----|-----|-----|-----|
| Original    | 1.7  | 2.27 | 2.8  | 2.46 | 2.76 |
| MVDR        | 1.87 | 2.68 | 2.9  | 2.64 | 2.48 |
| MVDR stream | 1.66 | 2.5  | 2.78 | 2.56 | 2.24 |
| SDWMW       | 1.77 | 2.39 | 2.75 | 2.51 | 2.82 |