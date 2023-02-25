import numpy as np
import torch
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from pathlib import Path

import sys
sys.path.insert(1, str(sys.path[0])+'/..')

from encoder import inference as encoder
from synthesizer.inference import Synthesizer
from vocoder import inference as vocoder

from parallel_wavegan.utils import load_model

def get_spec(models_path, ref_wav_path):
    encoder.load_model(Path(os.path.join(models_path, 'encoder.pt')))
    synthesizer = Synthesizer(Path(os.path.join(models_path, 'reserved_synthesizer(old_audio_with_new_params_new_wavs).pt')))
    
    # ref_wav_path = os.path.join(main_path, 'dataset/persian_data/train_data/book-1/', ref_wav_path) ## refrence wav
    wav = Synthesizer.load_preprocess_wav(ref_wav_path)
    
    encoder_wav = encoder.preprocess_wav(wav)
    embed, partial_embeds, _ = encoder.embed_utterance(encoder_wav, return_partials=True)
    
    texts = ["SIL M A T N B E G O F T AA R E F AA R S I SIL"]
    embeds = [embed] * len(texts)
    specs = synthesizer.synthesize_spectrograms(texts, embeds)
    spec = np.concatenate(specs, axis=1)
    spec = torch.from_numpy(spec.T).to('cpu')
    return spec

def get_wavform(models_path, spec):
    vocoder = load_model('/mnt/hdd1/adibian/SV2TTS/persian-SV2TTS/vocoder/hifigan/checkpoint-2500000steps.pkl')
    vocoder.remove_weight_norm()
    vocoder = vocoder.eval().to('cpu')
    with torch.no_grad():
        wav = vocoder.inference(spec)
    wav = wav / np.abs(wav).max() * 0.95
    return wav

def plot_result(spec, waveform):
    # x = np.concatenate(specs, axis=1)
    # pred_spectrogram = torch.from_numpy(spec.T).to('cpu')
    fig, axs = plt.subplots(2, 1, figsize=(9, 8))
    ax1 = axs[0]
    ax2 = axs[1]
    im = ax1.imshow(np.rot90(spec), interpolation="none")
    fig.colorbar(mappable=im, orientation="vertical", ax=ax1)
    ax1.title.set_text('Spectrogram \n "SIL M A T N B E G O F T AA R E F AA R S I SIL (Persian text-to-speech)"')
    ax1.set_xlabel('Frams')
    ax1.set_ylabel('Chanels')
    times = np.linspace(0, len(waveform)/24000, num=len(waveform))
    ax2.plot(times, waveform)
    ax2.title.set_text('Waveform')
    ax2.set_xlabel('Time(s)')
    ax2.set_ylabel('Amplitude')
    plt.tight_layout()
    plt.savefig('pictures/spec_and_wave.pdf', format="pdf", bbox_inches="tight")
    plt.savefig('pictures/spec_and_wave.png')
    plt.close()

def main():
    model_path = "/mnt/hdd1/adibian/SV2TTS/persian-SV2TTS/saved_models/my_run"
    ref_utterance = "/mnt/hdd1/adibian/SV2TTS/persian-SV2TTS/dataset/persian_data/train_data/speaker-001/book-1/utterance-00018-000152-1.wav"
    spec = get_spec(model_path, ref_utterance)
    print('spec shape: ' + str(spec.shape))
    waveform = get_wavform(model_path, spec)
    print("waveform is created!")
    plot_result(spec, waveform)

if __name__ == "__main__":
    main()