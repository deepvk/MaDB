import numpy as np
import argparse
import torch
import yaml

from NISQAs.src.core.model_torch import model_init
from NISQAs.src.utils.process_utils import process


def yamlparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yaml",
        default="web_streamlit/NISQAs/config/nisqa_s.yaml",
        type=str,
        help="YAML file with config",
    )
    args = parser.parse_args()
    args = vars(args)
    return args


args = yamlparser()
with open(args["yaml"], "r") as ymlfile:
    args_yaml = yaml.load(ymlfile, Loader=yaml.FullLoader)
args = {**args_yaml, **args}
model, h0, c0 = model_init(args)


def nisqa_get_metrics(audio, sr):
    global h0, c0
    audio = audio[0]
    framesize = sr * args["frame"]
    if audio.shape[0] % framesize != 0:
        audio = torch.cat((audio, torch.zeros(framesize - audio.shape[0] % framesize)))
    audio_spl = torch.split(audio, framesize, dim=0)
    if args["warmup"]:
        _, _, _ = process(torch.zeros((1, framesize)), sr, model, h0, c0, args)
    out_all = []
    # print("NOI    COL   DISC  LOUD  MOS")
    # np.set_printoptions(precision=3)
    for audio in audio_spl:
        out, h0, c0 = process(audio, sr, model, h0, c0, args)
        out_all.append(out[0].numpy())

    avg_out = np.mean(out_all, axis=0)
    return avg_out
    # print("Average over file:")
    # print(avg_out)
