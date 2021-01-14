import time
from functools import partial

import torch
import wavio
from TTS.utils.audio import AudioProcessor
from TTS.utils.generic_utils import setup_model
from TTS.utils.io import load_config
from TTS.utils.synthesis import synthesis
from TTS.utils.text.symbols import phonemes, symbols
from TTS.vocoder.utils.generic_utils import setup_generator

# runtime settings
use_cuda = False

# model paths
BASE_DIR = "./"
TTS_MODEL = BASE_DIR + "tts_model.pth.tar"
TTS_CONFIG = BASE_DIR + "config.json"
VOCODER_MODEL = BASE_DIR + "vocoder_model.pth.tar"
VOCODER_CONFIG = BASE_DIR + "config_vocoder.json"

# 读取配置文件
TTS_CONFIG = load_config(TTS_CONFIG)
VOCODER_CONFIG = load_config(VOCODER_CONFIG)

# 加载 audio processor
ap = AudioProcessor(**TTS_CONFIG.audio)
# 加载 TTS MODEL
# multi speaker
speaker_id = None
speakers = []

# load the model
num_chars = len(phonemes) if TTS_CONFIG.use_phonemes else len(symbols)
model = setup_model(num_chars, len(speakers), TTS_CONFIG)

# load model state
cp = torch.load(TTS_MODEL, map_location=torch.device("cpu"))

# load the model
model.load_state_dict(cp["model"])
if use_cuda:
    model.cuda()
model.eval()

# set model stepsize
if "r" in cp:
    model.decoder.set_r(cp["r"])

# LOAD VOCODER MODEL
vocoder_model = setup_generator(VOCODER_CONFIG)
vocoder_model.load_state_dict(torch.load(VOCODER_MODEL, map_location="cpu")["model"])
vocoder_model.remove_weight_norm()
vocoder_model.inference_padding = 0

ap_vocoder = AudioProcessor(**VOCODER_CONFIG["audio"])
if use_cuda:
    vocoder_model.cuda()
vocoder_model.eval()


def tts(model, text, file_name, CONFIG, use_cuda, ap, use_gl):
    t_1 = time.time()
    waveform, alignment, mel_spec, mel_postnet_spec, stop_tokens, inputs = synthesis(
        model,
        text,
        CONFIG,
        use_cuda,
        ap,
        truncated=False,
        enable_eos_bos_chars=CONFIG.enable_eos_bos_chars,
    )
    if not use_gl:
        waveform = vocoder_model.inference(
            torch.FloatTensor(mel_postnet_spec.T).unsqueeze(0)
        )
        waveform = waveform.flatten()
    if use_cuda:
        waveform = waveform.cpu()
    waveform = waveform.numpy()
    rtf = (time.time() - t_1) / (len(waveform) / ap.sample_rate)
    tps = (time.time() - t_1) / len(waveform)

    print(waveform.shape)
    print(" > Run-time: {}".format(time.time() - t_1))
    print(" > Real-time factor: {}".format(rtf))
    print(" > Time per step: {}".format(tps))
    wavio.write(file_name, waveform, CONFIG.audio["sample_rate"], sampwidth=2)  # 将 wav 写入文件
    return alignment, mel_postnet_spec, stop_tokens, waveform


tts_run = partial(
    tts, model=model, CONFIG=TTS_CONFIG, use_cuda=use_cuda, ap=ap, use_gl=False
)

if __name__ == "__main__":
    sentence = "Bill got in the habit of asking himself “Is that thought true?” and if he wasn’t absolutely certain it was, he just let it go."
    file_name = "myfile.wav"
    align, spec, stop_tokens, wav = tts(
        model, sentence, file_name, TTS_CONFIG, use_cuda, ap, use_gl=False
    )
