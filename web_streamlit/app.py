import torch
import warnings

from file_proc import file_infer
from src.model import Unet_model
from src.Modules import PipeLine

warnings.filterwarnings("ignore")

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

    file_infer(pipline, model)
