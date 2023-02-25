
from synthesizer.hparams import hparams
from synthesizer import audio
from encoder import inference as encoder
from parallel_wavegan.utils import load_model

import os
import librosa
import numpy as np 
import torch 
import soundfile as sf

def infer(wav_path, save_path):
    wav, _ = librosa.load(str(wav_path), hparams.sample_rate)
    if hparams.rescale:
        wav = wav / np.abs(wav).max() * hparams.rescaling_max
    # Trim silence
    if hparams.trim_silence:
        wav = encoder.preprocess_wav(wav, normalize=False, trim_silence=True)
    spec = audio.melspectrogram(wav, hparams).astype(np.float32)
    spec = torch.from_numpy(spec.T).to('cpu')

    vocoder = load_model('/mnt/hdd1/adibian/SV2TTS/persian-SV2TTS/vocoder/hifigan/checkpoint-2500000steps.pkl')
    vocoder.remove_weight_norm()
    vocoder = vocoder.eval().to('cpu')
    with torch.no_grad():
        wav = vocoder.inference(spec)
    wav = wav / np.abs(wav).max() * 0.95
    save_path = os.path.join(save_path, wav_path.split('/')[-1])
    sf.write(save_path, wav, hparams.sample_rate)

def main(data_path, save_path):
    for file in os.listdir(data_path):
        wav_path = os.path.join(data_path, file)
        print(file)
        infer(wav_path, save_path)

if __name__ == "__main__":
    data_path = "/mnt/hdd1/adibian/SV2TTS/persian-SV2TTS/evaluate_vocoder/test_data"
    save_path = "/mnt/hdd1/adibian/SV2TTS/persian-SV2TTS/evaluate_vocoder/results"
    main(data_path, save_path)