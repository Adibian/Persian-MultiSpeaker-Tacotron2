
import matplotlib.pyplot as plt
import os
import numpy as np
from pathlib import Path

import sys
sys.path.insert(1, str(sys.path[0])+'/..')

from encoder import inference as encoder
from synthesizer.inference import Synthesizer
from parallel_wavegan.utils import load_model

main_path = os.getcwd()
models_path = os.path.join(main_path, '../saved_models/my_run/')

encoder.load_model(Path(models_path + 'encoder.pt'))
synthesizer = Synthesizer(Path(models_path + 'reserved_synthesizer(old_audio_with_new_params_new_wavs).pt'))

def main():
    main_path = "/mnt/hdd1/adibian/SV2TTS/persian-SV2TTS/dataset/persian_data/train_data"
    embed_per_speaker = {}
    for speaker_dir in os.listdir(main_path):
        speaker_path = os.path.join(main_path, speaker_dir, 'book-1')
        print(speaker_path)
        speaker_embed = None
        n_embed = 0
        for file_name in os.listdir(speaker_path):
            if '.txt' in file_name:
                file_path = os.path.join(speaker_path, file_name)
                with open(file_path, 'r') as f:
                    text = f.readline()
                n_phonemes = len(text.split())
                if n_phonemes >= 20 and n_phonemes < 120:
                    ref_wav_path = file_path.replace('.txt', '.wav') ## refrence wav
                    wav = Synthesizer.load_preprocess_wav(ref_wav_path)
                    encoder_wav = encoder.preprocess_wav(wav)
                    embed, _, _ = encoder.embed_utterance(encoder_wav, return_partials=True)
                    if speaker_embed is None:
                        speaker_embed = embed
                    else:
                        speaker_embed += embed
                    n_embed += 1
        embed_per_speaker[str(speaker_dir)] = speaker_embed/n_embed
    for speaker in embed_per_speaker.keys():
        np.save(os.path.join('speaker_embed', speaker+'.np'), embed_per_speaker[speaker])

if __name__ == "__main__":
    main()