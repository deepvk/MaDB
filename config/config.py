from dataclasses import dataclass


@dataclass
class TrainConfig:

    # PipeLine
    sample_rate: int = 16000 # Sample rate.
    window_length: int = 400 # Window length STFT (25 ms).
    window_shift: int = 160 # Window shift STFT (10 ms).

    # loss
    loss_nfft: tuple = (400, 256) # Number of FFT bins for calculating loss.
    gamma: float = 0.3 # Gamma parameter for adjusting the focus of the loss on certain aspects of the audio spectrum.
    factor: int = 100 # Factors for different components of the loss function.
    c_factor: int = 100

    # Data

    ## Data train
    root_train: str = "dataset/data" # Root directory for training data.
    url_train: str = "train-clean-100" # Training dataset URL.
    train_download: bool = False # Download training dataset.

    ## Data valid
    root_valid: str = "dataset/data1" # Root directory for validation data.
    url_valid: str = "test-clean" # Validation dataset URL.
    valid_download: bool = False # Download validation dataset.

    sec: int = 3 # Length (in seconds) of each audio segment used during training.
    min_snr: int = 0 # Minimum SNR voice/noise.
    max_snr: int = 5 # Maximum SNR voice/noise.
    path_demand: str = "dataset/demand" # Directory path where the Demand dataset is stored.
    path_musan: str = "dataset/musan" # Directory path where the Musan dataset is stored.

    # Dataloader
    batch_size: int = 32 # Batch size for training.
    shuffle_train: bool = True # Shuffle the training dataset.
    drop_last_train: bool = True # Drop the last incomplete batch in train data.

    shuffle_valid: bool = False # Shuffle the validation dataset.
    drop_last_valid: bool = False # Drop the last incomplete batch in valid data.

    # Trainer
    epochs: int = 100 # Number of training epochs.
    device: str = "cuda" # The computing platform for training: 'cuda' for NVIDIA GPUs or 'cpu'.

    lr: float = 3e-4 # Learning rate for the optimizer.
    T_0: int = 50  # Period of the cosine annealing schedule in learning rate adjustment.
    log_step_path: str = "train_steps.log" # Path for logging steps.
    log_epoch_path: str = "train_epochs.log" # Path for logging epochs.
    path_checkpoint: str = "" # Path to save the checkpoint of the model.
 

@dataclass
class InferenceConfig:

    # Model
    model_weigth_path: str = "src/weights/checkpoint_denoise_last.pt" # Checkpoint that will be used for inference

    # PipeLine
    sample_rate: int = 16000 # Sample rate.
    window_length: int = 400 # Window length STFT (25 ms).
    window_shift: int = 160 # Window shift STFT (10 ms).

    # Inference
    backet_size: int = 800 # Framesize for file (50 ms)
    stream: bool = False # Use the streaming mode of inference
    in_path_audio: str = "src/sample/kitchen.wav" # Path to the input audio file
    out_path_audio: str = "src/sample/kitchen_predict.wav" # Path to the output audio file
