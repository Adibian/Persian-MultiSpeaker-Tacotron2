import numpy as np
import torch
import os
import time
import noisereduce as nr
from pathlib import Path
import soundfile as sf

from encoder import inference as encoder
from synthesizer.inference import Synthesizer
from vocoder import inference as vocoder

import librosa

main_path = os.getcwd()
models_path = os.path.join(main_path, 'saved_models/my_run/')

encoder.load_model(Path(models_path + 'encoder.pt'))
synthesizer = Synthesizer(Path(models_path + 'synthesizer.pt'))
vocoder.load_model(Path(models_path + 'vocoder.pt'))

def inference(dir_name, ref_wav_path, text):
    ref_wav_path = os.path.join(main_path, ref_wav_path) ## refrence wav
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
    
    res_path = os.path.join(main_path, 'results', 'v1')
    os.makedirs(res_path, exist_ok=True)
    
    res_path_normal = os.path.join(res_path, 'normal', dir_name)
    os.makedirs(res_path_normal, exist_ok=True)
    save_path = os.path.join(res_path_normal, ref_wav_path.split('/')[-1])
    # librosa.output.write_wav(save_path, wav, rate=Synthesizer.sample_rate)
    sf.write(save_path, wav, Synthesizer.sample_rate)
    
    res_path_denoised = os.path.join(res_path, 'denoised', dir_name)
    os.makedirs(res_path_denoised, exist_ok=True)
    save_path = os.path.join(res_path_denoised, ref_wav_path.split('/')[-1])
    reduced_noise = nr.reduce_noise(y=wav, sr=Synthesizer.sample_rate)
    #librosa.output.write_wav(save_path, reduced_noise, rate=Synthesizer.sample_rate)
    sf.write(save_path, reduced_noise, Synthesizer.sample_rate)
    print('\nwaves are saved.')
    
def main():
    """
    print("SEEN text and SEEN autterance and parallel(correlated):")
    ## SEEN text and SEEN autterance and parallel(correlated):
    inference("seen_text_seen_autt_para", "dataset/persian_data/train_data/speaker-001/book-1/utterance-00029-000526-1.wav", "SIL KH O B SIL M A AH L U M E AH A S T D O AH AA Y E H A Z R A T E Z A H R AA S A L AA M O L L AA H AH A L A Y H AA SIL B AA D O AH AA Y E M A N SIL KH E Y L I M O T A F AA V E T AH A S T SIL")
    inference("seen_text_seen_autt_para", "dataset/persian_data/train_data/speaker-006/book-1/utterance-00200-006399-1.wav", "SIL F A R D AA Y E AH AA N R U Z SIL S O R AA Q E AH A L I KH AA D E M R AA G E R E F T A M SIL G O F T A N D R A F T E S I S T AA N SIL")
    inference("seen_text_seen_autt_para", "dataset/persian_data/train_data/speaker-021/book-1/utterance-00579-016206-1.wav", "M A N H A M M E S L E S I D N I Y E SIL AH AA SH E Q E B AA Z I G A R I B U D A M SIL M A N AH A Z P E D A R O M AA D A R I B AA Z I G A R SIL Z AA D E SH O D E B U D A M SIL")

    print("SEEN text and SEEN autterance and non-parallel(not correlated):")
    ## SEEN text and SEEN autterance and non-parallel(not correlated):
    inference("seen_text_seen_autt_nonpara", "dataset/persian_data/train_data/speaker-001/book-1/utterance-00030-000598-1.wav", "SIL KH O B SIL M A AH L U M E AH A S T D O AH AA Y E H A Z R A T E Z A H R AA S A L AA M O L L AA H AH A L A Y H AA SIL B AA D O AH AA Y E M A N SIL KH E Y L I M O T A F AA V E T AH A S T SIL")
    inference("seen_text_seen_autt_nonpara", "dataset/persian_data/train_data/speaker-006/book-1/utterance-00173-005043-3.wav", "SIL F A R D AA Y E AH AA N R U Z SIL S O R AA Q E AH A L I KH AA D E M R AA G E R E F T A M SIL G O F T A N D R A F T E S I S T AA N SIL")
    inference("seen_text_seen_autt_nonpara", "dataset/persian_data/train_data/speaker-021/book-1/utterance-00579-016206-1.wav", "SIL Z O H R E K E S AA L A T B AA R I AH A Z M A D R E S E B A R G A SH T A M SIL K A S I T U Y E KH AA N E N A B U D SIL")

    print("UNSEEN text and SEEN autterance:")
    ## UNSEEN text and SEEN autterance:
    inference("unseen_text_seen_autt", "dataset/persian_data/train_data/speaker-001/book-1/utterance-00030-000598-1.wav", "AH AA N H AA H A R D O Q A V I P O R J A Z A B E V AA V B AA N O F U Z B U D A N D SIL")
    inference("unseen_text_seen_autt", "dataset/persian_data/train_data/speaker-006/book-1/utterance-00173-005043-3.wav", "H A R D O B E M A N T O S I Y E O AH A N D A R Z M I D AA D A N D V A L I T O S I Y E H AA Y E AH AA N H AA M O T E F AA V E T B U D SIL")
    inference("unseen_text_seen_autt", "dataset/persian_data/train_data/speaker-021/book-1/utterance-00579-016206-1.wav", "H A R D O B E M A N T O S I Y E O AH A N D A R Z M I D AA D A N D V A L I T O S I Y E H AA Y E AH AA N H AA M O T E F AA V E T B U D SIL")

    print("SEEN text and UNSEEN autterance:")
    ## SEEN text and UNSEEN autterance:
    inference("seen_text_unseen_autt", "dataset/persian_data/test_data/speaker-002/book-1/utterance-00125-003383-1.wav", "SIL S O N I Y AA V A Q T I AH I N R AA KH AA N D V AA Q E AH A N T A R S I D SIL AH A L B A T T E SIL T A R S A SH M A N T E Q I B U D SIL")
    inference("seen_text_unseen_autt", "dataset/persian_data/test_data/speaker-008/book-1/utterance-01713-039077-1.wav", "M A S A L A N M AA AH AA N J AA SIL D A R M O Q E AH E AH A N J AA M K AA R B A R AA Y E R A AH AA Y A T AH O S U L E B E H D AA SH T I AH E M K AA N AA T I N A D AA SH T I M SIL")
    inference("seen_text_unseen_autt", "dataset/persian_data/test_data/speaker-048/book-1/utterance-01369-031946-1.wav", "B E H N A Z A R A M Z A M AA N A SH R E S I D E B U D K E H P E Y E SIL AH E SH Q O AH A L AA Q E AH A M B E R A V A M SIL")

    print("UNSEEN text and UNSEEN autterance and parallel(correlated):")
    ## UNSEEN text and UNSEEN autterance and parallel(correlated):
    inference("unseen_text_unseen_autt_para", "dataset/persian_data/test_data/speaker-002/book-1/utterance-00125-003383-1.wav", "P A S AH A Z AH AA N H A M SIL N E V I S A N D E G AA N I B E T AA R I KH AH I R AA N R U Y AH AA V A R D A N D SIL K A H AH A Z H A M AA N S O N N A T E M A K T A B E K E L AA S I K D A R B AA Z G O F T A N S A R G O Z A SH T E SH AA H AA N O S A R D AA R AA N SIL P E Y R A V I M I K A R D A N D SIL")
    inference("unseen_text_unseen_autt_para", "dataset/persian_data/test_data/speaker-008/book-1/utterance-01713-039077-1.wav", "AH O S T AA D M O H A M M A D AH A B D O L L AA H E AH A N AA N T A S R I H M I K O N A D K E H B E R A Q M E P AA F E SH AA R I Y E AH A R B AA B AA N E K E L I S AA SIL")
    inference("unseen_text_unseen_autt_para", "dataset/persian_data/test_data/speaker-048/book-1/utterance-01369-031946-1.wav", "V AA V D A R S AA L E H E Z AA R O S I S A D O SIL P A N J AA H O H A F T SIL D A R F E H R E S T E AH AA S AA R E M E L L I Y E AH I R AA N SIL B E H S A B T R E S I D E AH A S T SIL")

    print("UNSEEN text and UNSEEN autterance and non-parallel(not correlated):")
    ## UNSEEN text and UNSEEN autterance and non-parallel(not correlated):
    inference("unseen_text_unseen_autt_nonpara", "dataset/persian_data/test_data/speaker-002/book-1/utterance-00125-003383-1.wav", "SIL K A M K A M SIL P E S A R H AA Y E AH O Z V E K AA N U N AH A N J O M A N E AH E S L AA M I J E B H E R A F T A N H AA Y E SH AA N SH O R U AH SH O D SIL")
    inference("unseen_text_unseen_autt_nonpara", "dataset/persian_data/test_data/speaker-008/book-1/utterance-01713-039077-1.wav", "SIL CH E N AA N B O R U Z P E Y D AA K A R D K E H Z A N O SH O H A R AH A Q L A B D U R AH A Z H A M Z E N D E G I M I K A R D A N D SIL")
    inference("unseen_text_unseen_autt_nonpara", "dataset/persian_data/test_data/speaker-048/book-1/utterance-01369-031946-1.wav", "V AA V D A R N A Z D I K I Y E M E Y D AA N E B A H M A N I Y E SH A H R E S T AA N E B U SH E H R SIL Q A R AA R G E R E F T E AH A S T SIL")
    """
    
    #### New Speaker ####
    
    print("SEEN text and UNSEEN speaker:")
    ## SEEN text and UNSEEN speaker: 
    inference("seen_text_unseen_speaker", "dataset/persian_data/test_data/speaker-051/book-1/utterance-01429-033675-1.wav", "SIL G U Y A N D AA KH O D AA H AA F E Z I K A R D SIL S I Z AA R T AA G O F T SIL KH O D AA H AA F E Z SIL G U Y A N D AA SIL")
    inference("seen_text_unseen_speaker", "dataset/persian_data/test_data/speaker-062/book-1/utterance-01776-039743-1.wav", "SIL G U Y A N D AA SIL S AA K E T SH O D SIL N A F A S E AH A M I Q I K E SH I D O G O F T SIL H A R F AA T M A N R O M O Z T A R E B SIL M I K O N E SIL")
    inference("seen_text_unseen_speaker", "dataset/persian_data/test_data/speaker-064/book-1/utterance-01821-040816-1.wav", "AH AA Y E Y E Q O R AH AA N N AA Z E L M I SH O D S A R I H A N AH AA N AH A F K AA R R AA R A D D M I K A R D SIL")
    inference("seen_text_unseen_speaker", "dataset/persian_data/test_data/speaker-052/book-1/utterance-01441-033998-1.wav", "H A M E Y E Z E N D AA N I Y AA N E S I Y AA S I T AA V A Q T I M A N D A R N I Y O Y O R K B U D A M AH AA Z AA D SH O D A N D SIL")
    inference("seen_text_unseen_speaker", "dataset/persian_data/test_data/speaker-066/book-1/utterance-01830-041070-1.wav", "M A R V AA N O AH A Z U N E V E SH T A N D L I B I AH AA Z AA D E AH A S T M A R G B A R Q A Z Z AA F I SIL S E D AA Y E D I G A R I AH AA M A D SIL")

    print("UNSEEN text and UNSEEN speaker and parallel(correlated):")
    ## UNSEEN text and UNSEEN speaker and parallel(correlated): 
    inference("unseen_text_unseen_speaker_para", "dataset/persian_data/test_data/speaker-051/book-1/utterance-01429-033675-1.wav", "M A S AH A L E Y E AH A V V A L D A R AH I N SH O Q L KH U B KH A N D I D A N AH A S T SIL K E H H I CH K A S I N A T A V AA N E S T E T AA H AA L H AA Z E R B E M AA N A N D E M A N AH A Z KH A N D E Q A SH SIL K O N A D SIL")
    inference("unseen_text_unseen_speaker_para", "dataset/persian_data/test_data/speaker-062/book-1/utterance-01776-039743-1.wav", "B AA AH I N K E B AA Z I S E SH A M B E SH A B V A D A R V A S A T E H A F T E B A R G O Z AA R SH O D SIL AH A Z I M T A R I N G E R D E H A M AA Y I AH O M U M I P A S AH A Z AH AA Z AA D I SIL SH E K L G E R E F T SIL")
    inference("unseen_text_unseen_speaker_para", "dataset/persian_data/test_data/speaker-064/book-1/utterance-01821-040816-1.wav", "AH A M AA N E F R A T SIL T O T E AH E SIL V A K O SH T AA R M I Y AA N AH I R AA N I Y AA N O Y U N AA N I Y AA N SIL T AA Y E K SIL Q A R N B A AH D SIL AH E D AA M E Y AA F T SIL")
    inference("unseen_text_unseen_speaker_para", "dataset/persian_data/test_data/speaker-052/book-1/utterance-01441-033998-1.wav", "M A S AH U D K E H B I R U N B U D S E D AA Y E M O H S E N E SIL AH A Z I Z O R A H I M R AA SH E N I D E B U D H A M I N K E SIL V AA R E D AH O T AA Q SH O D B E G E R Y E AH O F T AA D O G O F T SIL")
    inference("unseen_text_unseen_speaker_para", "dataset/persian_data/test_data/speaker-066/book-1/utterance-01830-041070-1.wav", "SIL F E K R M I K O N A M M O Z U AH AA T E Z I Y AA D I V U J U D D AA R A N D K E M I T U N I M B E KH AA T E R E AH U N H AA AH A Z H A M D I G E Q A D R D AA N I K O N I M SIL AH A M M AA AH A Q L A B AH AA N H AA R AA N AA D I D E M I G I R I M SIL")

    print("UNSEEN text and UNSEEN speaker and non-parallel(not correlated):")
    ## UNSEEN text and UNSEEN speaker and non-parallel(not correlated): 
    inference("unseen_text_unseen_speaker_nonpara", "dataset/persian_data/test_data/speaker-051/book-1/utterance-01429-033675-1.wav", "B AA AH I N K E B AA Z I S E SH A M B E SH A B V A D A R V A S A T E H A F T E B A R G O Z AA R SH O D SIL AH A Z I M T A R I N G E R D E H A M AA Y I AH O M U M I P A S AH A Z AH AA Z AA D I SIL SH E K L G E R E F T SIL")
    inference("unseen_text_unseen_speaker_nonpara", "dataset/persian_data/test_data/speaker-062/book-1/utterance-01776-039743-1.wav", "M A S AH A L E Y E AH A V V A L D A R AH I N SH O Q L KH U B KH A N D I D A N AH A S T SIL K E H H I CH K A S I N A T A V AA N E S T E T AA H AA L H AA Z E R B E M AA N A N D E M A N AH A Z KH A N D E Q A SH SIL K O N A D SIL")
    inference("unseen_text_unseen_speaker_nonpara", "dataset/persian_data/test_data/speaker-064/book-1/utterance-01821-040816-1.wav", "V A Q T I CH A SH M B AA Z K A R D A M D I D A M P E D A R A M D A R AH O T AA Q N I S T V AA V M AA D A R A M P A H L U Y E M A N N E SH A S T E SIL")
    inference("unseen_text_unseen_speaker_nonpara", "dataset/persian_data/test_data/speaker-052/book-1/utterance-01441-033998-1.wav", "SIL H A N U Z SIL B U Y R A N G E T AA Z E Y E D I V AA R H AA B E H M A SH AA M M I R E S I D SIL")
    inference("unseen_text_unseen_speaker_nonpara", "dataset/persian_data/test_data/speaker-066/book-1/utterance-01830-041070-1.wav", "SIL K AA M I Y O N I R AA D A R N A Z A R M I G I R I M K E H R A S M A N AH E AH L AA M M I SH O D B AA Y A D SH O M AA R V I ZH E AH I AH A Z Z E N D AA N I Y AA N R AA SIL")

if __name__ == "__main__":
   main()
    
