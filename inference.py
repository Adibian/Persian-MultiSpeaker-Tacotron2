import numpy as np
import torch
import sys

from encoder import inference as encoder
from synthesizer.inference import Synthesizer
from vocoder import inference as vocoder_wavrnn
from parallel_wavegan.utils import load_model as vocoder_hifigan

import soundfile as sf
import os
import argparse


main_path = os.getcwd()
models_path = os.path.join(main_path, 'saved_models/final_models/')

def wavRNN_infer(text, ref_wav_path, test_name):
    encoder.load_model(os.path.join(models_path, 'encoder.pt'))
    synthesizer = Synthesizer(os.path.join(models_path, 'synthesizer.pt'))
    vocoder_wavrnn.load_model(os.path.join(models_path, 'vocoder_WavRNN.pt'))
    
    ref_wav_path = os.path.join(main_path, 'dataset/persian_data/train_data/book-1/', ref_wav_path) ## refrence wav
    wav = Synthesizer.load_preprocess_wav(ref_wav_path)
    
    encoder_wav = encoder.preprocess_wav(wav)
    embed, partial_embeds, _ = encoder.embed_utterance(encoder_wav, return_partials=True)
    
    texts = [text]
    embeds = [embed] * len(texts)
    specs = synthesizer.synthesize_spectrograms(texts, embeds)
    breaks = [spec.shape[1] for spec in specs]
    spec = np.concatenate(specs, axis=1)
    
    wav = vocoder_wavrnn.infer_waveform(spec)
    b_ends = np.cumsum(np.array(breaks) * Synthesizer.hparams.hop_size)
    b_starts = np.concatenate(([0], b_ends[:-1]))
    wavs = [wav[start:end] for start, end, in zip(b_starts, b_ends)]
    breaks = [np.zeros(int(0.15 * Synthesizer.sample_rate))] * len(breaks)
    wav = np.concatenate([i for w, b in zip(wavs, breaks) for i in (w, b)])
    wav = wav / np.abs(wav).max() * 0.97
    
    res_path = os.path.join(main_path, 'results/', test_name+".wav")
    sf.write(res_path, wav, Synthesizer.sample_rate)
    print('\nwav file is saved.')


def hifigan_infer(text, ref_wav_path, test_name):
    encoder.load_model(os.path.join(models_path, 'encoder.pt'))
    synthesizer = Synthesizer(os.path.join(models_path, 'synthesizer.pt'))
    vocoder = vocoder_hifigan(os.path.join(models_path, 'vocoder_HiFiGAN.pkl'))
    vocoder.remove_weight_norm()
    vocoder = vocoder.eval().to('cpu')

    wav = Synthesizer.load_preprocess_wav(ref_wav_path)
    
    encoder_wav = encoder.preprocess_wav(wav)
    embed, partial_embeds, _ = encoder.embed_utterance(encoder_wav, return_partials=True)
    
    texts = [text]
    embeds = [embed] * len(texts)
    specs = synthesizer.synthesize_spectrograms(texts, embeds)
    spec = np.concatenate(specs, axis=1)
    x = torch.from_numpy(spec.T).to('cpu')
    
    with torch.no_grad():
        wav = vocoder.inference(x)
    wav = wav / np.abs(wav).max() * 0.97
    
    res_path = os.path.join(main_path, 'results/', test_name+".wav")
    sf.write(res_path, wav, Synthesizer.sample_rate)
    print('\nwav file is saved.')


def main(args):
    if str(args.vocoder).lower() == "wavrnn":
        wavRNN_infer(args.text, args.ref_wav_path, args.test_name)
    elif str(args.vocoder).lower() == "hifigan":
        hifigan_infer(args.text, args.ref_wav_path, args.test_name)
    else:
        print("--vocoder must be one of HiFiGAN or WavRNN")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocoder", type=str, help= "vocoder name: HiFiGAN or WavRNN")
    parser.add_argument("--text", type=str, help="input text")
    parser.add_argument("--ref_wav_path", type=str, help="path to refrence wav to create speaker from that")
    parser.add_argument("--test_name", type=str, default="test1", help="name of current test to save the result wav")
    args = parser.parse_args()

    main(args)
    
    