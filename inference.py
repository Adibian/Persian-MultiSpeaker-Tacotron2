import numpy as np
import torch
import sys

from encoder import inference as encoder
from synthesizer.inference import Synthesizer
from vocoder import inference as vocoder

import librosa

main_path = os.getcwd()
models_path = os.path.join(main_path, 'saved_models/my_run/')

def main(dir_name, ref_wav_path, text):
    encoder.load_model(os.path.join(models_path, 'encoder.pt'))
    synthesizer = Synthesizer(os.path.join(models_path, 'synthesizer.pt'))
    vocoder.load_model(os.path.join(models_path, 'vocoder.pt'))
    
    ref_wav_path = os.path.join(main_path, 'dataset/persian_data/train_data/book-1/', ref_wav_path) ## refrence wav
    wav = Synthesizer.load_preprocess_wav(ref_wav_path)
    
    encoder_wav = encoder.preprocess_wav(wav)
    embed, partial_embeds, _ = encoder.embed_utterance(encoder_wav, return_partials=True)
    
    texts = [text]
    embeds = [embed] * len(texts)
    specs = synthesizer.synthesize_spectrograms(texts, embeds)
    breaks = [spec.shape[1] for spec in specs]
    spec = np.concatenate(specs, axis=1)
    
    wav = vocoder.infer_waveform(spec)
    b_ends = np.cumsum(np.array(breaks) * Synthesizer.hparams.hop_size)
    b_starts = np.concatenate(([0], b_ends[:-1]))
    wavs = [wav[start:end] for start, end, in zip(b_starts, b_ends)]
    breaks = [np.zeros(int(0.15 * Synthesizer.sample_rate))] * len(breaks)
    wav = np.concatenate([i for w, b in zip(wavs, breaks) for i in (w, b)])
    wav = wav / np.abs(wav).max() * 0.97
    
    res_path = os.path.join(main_path, 'results/', dir_name)
    os.makedirs(os.path.dirname(res_path), exist_ok=True)
    save_path = os.path.join(res_path, ref_wav_path.split('/')[-1])
    librosa.output.write_wav(wav, rate=Synthesizer.sample_rate)
    
if __name__ == "__main__":
   main(sys.argv[1], sys.argv[2], sys.argv[3])
    
