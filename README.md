## MultiSpeaker Tacotron2 for Persian Language

This repository contains a Persian language adaptation of [Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis (SV2TTS)](https://arxiv.org/pdf/1806.04558.pdf). The core implementation is based on [this repository](https://github.com/CorentinJ/Real-Time-Voice-Cloning/tree/master), modified to work with Persian text and phoneme data.

<img src="https://github.com/majidAdibian77/persian-SV2TTS/blob/master/results/model.JPG" width="800"> 

---

## Quickstart

### Data Structure

Organize your data as follows:
```
dataset/persian_date/
    train_data/
        speaker1/book-1/
            sample1.txt
            sample1.wav
            ...
        ...
    test_data/
        ...
```

### Preprocessing

1. **Audio Preprocessing**  
```
python synthesizer_preprocess_audio.py dataset --datasets_name persian_data --subfolders train_data --no_alignments
```
2. **Embedding Preprocessing**  
```
python synthesizer_preprocess_embeds.py dataset/SV2TTS/synthesizer
```

### Train the Synthesizer

To begin training the synthesizer model:
```
python synthesizer_train.py my_run dataset/SV2TTS/synthesizer
```

---

## Inference

To generate a wav file, place all trained models in the `saved_models/final_models` directory. If you haven’t trained the speaker encoder or vocoder models, you can use pretrained models from `saved_models/default`.

### Using WavRNN as Vocoder

```
python inference.py --vocoder "WavRNN" --text "یک نمونه از خروجی" --ref_wav_path "/path/to/sample/reference.wav" --test_name "test1"
```

### Using HiFiGAN as Vocoder (Recommended)
WavRNN is an old vocoder and if you want to use HiFiGAN you must first download a pretrained model in English.
1. **Install Parallel WaveGAN**  
```
pip install parallel_wavegan
```
2. **Download Pretrained HiFiGAN Model**  
```
from parallel_wavegan.utils import download_pretrained_model
download_pretrained_model("vctk_hifigan.v1", "saved_models/final_models/vocoder_HiFiGAN")
```
3. **Run Inference with HiFiGAN**
```
python inference.py --vocoder "HiFiGAN" --text "یک نمونه از خروجی" --ref_wav_path "/path/to/sample/reference.wav" --test_name "test1"
```
## Demo
Check out [some audio samples](https://github.com/majidAdibian77/persian-SV2TTS/tree/master/results/output_samples) from the trained model in this directory.

## References:
- [Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis](https://arxiv.org/pdf/1806.04558.pdf) Ye Jia, *et al*.,
- [Real-Time-Voice-Cloning repository](https://github.com/CorentinJ/Real-Time-Voice-Cloning/tree/master),
- [ParallelWaveGAN repository](https://github.com/kan-bayashi/ParallelWaveGAN)

## License  
This project is based on [Real-Time-Voice-Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning),  
which is licensed under the MIT License.  
The modifications for Persian language support are © [YEAR] [YOUR NAME].  

  
