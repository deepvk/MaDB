# BEAM
Модель:
https://drive.google.com/file/d/1LwDXGyJguMSeNrGfOYFsro4Lp1SVxWiL/view

`Modules.py`:

    1.STFT
    2.MVDR
    3.mask_estimator

`pl_model.py`:

    1. Spectral loss
    2. pytorch_lightning model

`data_valid.ipynb`: анализ предикта

`train.ipynb`: пример (теста) запуска обучалки

`RIR.ipynb`: пример генерации RIR

- [`v1`](./v1) ‒ обучалка MVDR в forward, модель из frame by frame.
- [`v2`](./v2) ‒ оверфит модели (conv-lstm-fc).
- [`v3`](./v3) - модель unet. unet.ipynb - legacy
- [`v4`](./v4) - различные вариации модели (unet) с causal conv: fc, sigmoid, conv
- [`v5`](./v5) - модель unet, оверфит с белым шумом.
- [`v6`](./v6) - аналогично v5, модель обучена на шуме с demand и musan
