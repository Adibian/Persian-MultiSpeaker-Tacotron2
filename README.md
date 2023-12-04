## MultiSpeaker Tacotron2 in Persian language
This repository is an implementation of [Transfer Learning from Speaker Verification to
Multispeaker Text-To-Speech Synthesis](https://arxiv.org/pdf/1806.04558.pdf) (SV2TTS) in Persian language.

The main code is from [this repository](https://github.com/CorentinJ/Real-Time-Voice-Cloning/tree/master) and it is changed for Persian language.

<img src="https://github.com/majidAdibian77/ResGrad/blob/master/resgrad.PNG" width="800"> 

## Quickstart
Data structures:
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

Preprocessing:
```
python synthesizer_preprocess_audio.py dataset --datasets_name persian_data --subfolders train_data --no_alignments
python synthesizer_preprocess_embeds.py dataset/SV2TTS/synthesizer
```

Train synthesizer:
```
python synthesizer_train.py my_run dataset/SV2TTS/synthesizer
```

For synthesizing wav file you must put all final models to `saved_models/final_models` directory.
If you do not train speaker encoder and vocoder models you can use pretrained models exist in `saved_models/default`.

Inference using WavRNN as vocoder:
```
python inference.py --vocoder "WavRNN" --text "یک نمونه از خروجی" --ref_wav_path "/path/to/sample/refrence.wav" --test_name "test1"
```
But WavRNN is an old vocoder and if you want to use HiFiGAN you must first download a pretrained model in English:
First install parallel_wavegan package. See pthis package](https://github.com/kan-bayashi/ParallelWaveGAN) for more information.
```
pip install parallel_wavegan
```
Then download pretrained HiFiGAN to your saved models:
```
from parallel_wavegan.utils import download_pretrained_model
download_pretrained_model("vctk_hifigan.v1", "saved_models/final_models/vocoder_HiFiGAN")
```
Now you can use HiFiGAN as vocoder in inference command:
```
python inference.py --vocoder "HiFiGAN" --text "یک نمونه از خروجی" --ref_wav_path "/path/to/sample/refrence.wav" --test_name "test1"
```

## References :notebook_with_decorative_cover:
- [Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis](https://arxiv.org/pdf/1806.04558.pdf) Ye Jia, *et al*.,
- [Real-Time-Voice-Cloning repository](https://github.com/CorentinJ/Real-Time-Voice-Cloning/tree/master),
- [ParallelWaveGAN repository]([https://arxiv.org/abs/2006.04558](https://github.com/kan-bayashi/ParallelWaveGAN))