import numpy as np
import torch
import os
import time
import noisereduce as nr
from pathlib import Path
import soundfile as sf

from encoder import inference as encoder
from synthesizer.inference import Synthesizer
from parallel_wavegan.utils import load_model


main_path = os.getcwd()
models_path = os.path.join(main_path, 'saved_models/my_run/')

encoder.load_model(Path(models_path + 'encoder.pt'))
# synthesizer = Synthesizer(Path(models_path + 'synthesizer_best.pt'))
synthesizer = Synthesizer(Path(models_path + 'new_config/synthesizer_000050.pt'))

## Folowing lines is for HiFiGAN
# fs = 24000
# vocoder = load_model('vocoder/hifigan/checkpoint-2500000steps.pkl')
fs = 22050
vocoder = load_model('vocoder/hifigan2/checkpoint-110000steps.pkl')
vocoder.remove_weight_norm()
vocoder = vocoder.eval().to('cpu')

male_or_female = {'006': 'male', '011': 'male', '017': 'male', '023': 'male', '030': 'male', '038': 'male', '053': 'male', '056': 'male', '057': 'male',  ## seen speaker
                  '004': 'female', '008': 'female', '025': 'female', '028': 'female', '031': 'female', '042': 'female', '063': 'female', ## seen speaker
                   
                  '067': 'male', '068': 'male', '069': 'male', '070': 'male', '073': 'male', ## unseen speaker 
                  '071': 'female', '072': 'female', '074': 'female', '075': 'female', '076': 'female'}  ## unseen speaker

def inference(speaker, dir_name, ref_wav_path, text):
    ref_wav_path = os.path.join(main_path, ref_wav_path) ## refrence wav
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
    wav = wav / np.abs(wav).max() * 0.95
    
    # res_path = os.path.join(main_path, 'results', 'best_result')
    res_path = os.path.join(main_path, 'results', 'temp')
    os.makedirs(res_path, exist_ok=True)
    
    res_path_normal = os.path.join(res_path, dir_name, male_or_female[speaker], speaker)
    os.makedirs(res_path_normal, exist_ok=True)
    save_path = os.path.join(res_path_normal, ref_wav_path.split('/')[-1])
    sf.write(save_path, wav, fs)
    print('\nwav file is saved.')
    
def main():
    # print("UNSEEN text and SEEN autterance:")
    # ## UNSEEN text and SEEN autterance:
    # inference('004', "unseen_text_seen_autt", "dataset/persian_data/train_data/speaker-004/book-1/utterance-00074-002128-1.wav", "SIL T A R S AH A Z AH E SH Q SIL T A R S AH A Z T A M AA M E Z E N D E G I S T SIL V A Z E N D E G I B E D U N E AH E SH Q SIL B I M A AH N I S T SIL") ##  ترس از عشق ترس از تمام زندگی ست و زندگی بدون عشق بی معنی ست
    # inference('004', "unseen_text_seen_autt", "dataset/persian_data/train_data/speaker-004/book-1/utterance-00074-002096-1.wav", "SIL AH A Z K A S AA N I K E H M I KH AA H A N D AH AA R E Z U H AA Y A T R AA SIL K U CH A K V A B I AH A R Z E SH N E SH AA N D A H A N D D U R I K O N SIL") ## از کسانی که می خواهند آرزوهایت را کوچک و بی ارزش نشان دهند دوری کن

    # inference('006', "unseen_text_seen_autt", "dataset/persian_data/train_data/speaker-006/book-1/utterance-00205-006519-1.wav", "SIL Z E N D E G I M O B AA R E Z E Y E D AA AH E M I SIL B E Y N E F A R D B U D A N V A AH O Z V I AH A Z J AA M E AH E B U D A N AH A S T SIL") ## زندگی مبارزه ی دائمی بین فرد بودن و و عضوی از جامعه بودن است
    # inference('006', "unseen_text_seen_autt", "dataset/persian_data/train_data/speaker-006/book-1/utterance-00205-006514-1.wav", "SIL AH E N S AA N SIL B E V AA S E T E Y E Z A R A B AA N E Q A L B A SH Z E N D E  N I S T SIL B A L K E H B E L O T F E KH O D AA Z E N D E AH A S T SIL") # انسان به واسطه ضربان قلبش زنده نیست بلکه به لطف خدا زنده است

    # inference('008', "unseen_text_seen_autt", "dataset/persian_data/train_data/speaker-008/book-1/utterance-00223-007099-1.wav", "SIL D A R D I K E AH E N S AA N R AA B E S O K U T V AA M I D AA R A D SIL S A N G I N T A R AH A Z D A R D I S T K E AH E N S AA N R AA B E F A R Y AA D V AA M I D AA R A D SIL") ##  دردی که انسان را به سکوت وا می دارد بدتر از دردی است که انسان را به فریاد وا می دارد
    # inference('008', "unseen_text_seen_autt", "dataset/persian_data/train_data/speaker-008/book-1/utterance-00224-007153-1.wav", "SIL Z E N D E G I Y E H A Q I Q I Y E B A R KH I AH A Z M A SH AA H I R SIL B A AH D AH A Z M A R G E SH AA N SH O R U AH SIL SH O D E H SIL")  ## زندگی حقیقی برخی از مشاهیر بعد از مرگشان شروع شده است

    # inference('011', "unseen_text_seen_autt", "dataset/persian_data/train_data/speaker-011/book-1/utterance-00312-009540-1.wav", "SIL AH A N D I SH I D A N B E H N E G A R E SH E M AA N SH E K L M I D A H A D SIL V A D A R Z E N D E G I Y E M AA N N A Q SH E M O H E M M I D AA R A D SIL") ## اندیشیدن به نگرشمان شکل می دهد و در زندگیمان نقش مهمی دارد
    # inference('011', "unseen_text_seen_autt", "dataset/persian_data/train_data/speaker-011/book-1/utterance-00316-009642-1.wav", "SIL Z E N D E G I M E S L E D O CH A R KH E S A V AA R I AH A S T SIL K E AH A G A R SIL B E J E L O H A R E K A T N A K O N I Z A M I N M I KH O R I SIL") ## زندگی مثل دوچرخه سواریست که اگر به جلو حرکت نکنی زمین می خوری

    # inference('017', "unseen_text_seen_autt", "dataset/persian_data/train_data/speaker-017/book-1/utterance-00481-013070-1.wav", "SIL D A R H A R K AA R I SIL AH E B T E D AA M O S A M M A M SH A V I D K E B AA Y A D K AA R I AH A N J AA M SH A V A D V A S E P A S R AA H R AA KH AA H I D Y AA F T SIL") ## در هر کاری ابتدا مصمم شوید که باید کاری انجام شود و سپس راه را خواهید یافت
    # inference('017', "unseen_text_seen_autt", "dataset/persian_data/train_data/speaker-017/book-1/utterance-00448-012473-1.wav", "SIL T A N H AA D AA V A R E Z E N D E G I KH E R A D AH A S T SIL K E R AA H N A M AA Y E P I CH I D E G I H AA Y E F A R I B A N D E H AH A S T SIL")  ## تنها داور زندگی خرد است که راهنمای پیچیدگی های فریبنده است
 
    # inference('023', "unseen_text_seen_autt", "dataset/persian_data/train_data/speaker-023/book-1/utterance-00623-016797-1.wav", "SIL R U D A K I AH A Z SH AA AH E R AA N E AH I R AA N I Y E Q A R N E CH AA H AA R E H E J R I AH A S T SIL")  ## رودکی از شاعران ایرانی قرن چهار هجری است
    # inference('023', "unseen_text_seen_autt", "dataset/persian_data/train_data/speaker-023/book-1/utterance-00634-016986-1.wav", "SIL B O Z O R G T A R I N AH AA M E L E AH A H A M M I Y Y A T E KH A L I J E F AA R S SIL V O J U D E M A AH AA D E N E S A R SH AA R E N A F T O G AA Z D A R AH AA N AH A S T SIL")  ## بزرگترین عامل اهمیت خلیج فارس وجود معادن سر شار نفت و گاز در آن است
 
    # inference('025', "unseen_text_seen_autt", "dataset/persian_data/train_data/speaker-025/book-1/utterance-00702-018392-1.wav", "SIL D A R SH AA H N AA M E H SIL T U L E AH O M R E P A H L A V AA N AA N I K E AH A Z N A S L E S AA M B U D A N D B E S I Y AA R T U L AA N I Y AA D SH O D E AH A S T SIL")  ## در شاهنامه طول عمر پهلوانانی که از نسل سام بودند بسیار طولانی یاد شده است
    # inference('025', "unseen_text_seen_autt", "dataset/persian_data/train_data/speaker-025/book-1/utterance-00690-018178-1.wav", "SIL T E H R AA N SH A H R I B AA G U N AA G U N I Y E G O R U H H AA Y E Q O U M I AH A S T SIL V A L I J A M AH I Y Y A T E KH AA R E J I Y E AH AA N K A M AH A S T SIL")  ## تهران شهری با گوناگونی گروه های قومی است ولی جمعیت خارجی آن کم است

    # inference('028', "unseen_text_seen_autt", "dataset/persian_data/train_data/speaker-028/book-1/utterance-00781-020808-1.wav", "SIL S A R Z A M I N E AH I R AA N SIL M I Z B AA N E T A M A D D O N H AA Y E K O H A N I CH O N AH I L AA M V A J I R O F T B U D E AH A S T SIL")  ## سرزمین ایران میزبان تمدن های کهنی چون ایلام و جیرفت بوده است
    # inference('028', "unseen_text_seen_autt", "dataset/persian_data/train_data/speaker-028/book-1/utterance-00790-021055-1.wav", "SIL AH A R Z E SH CH I Z I N I S T J O Z M A AH N AA Y I K E SH O M AA B A R AA Y E AA N B A R M I G O Z I N I D SIL") ## ارزش چیزی نیست جز معنایی که شما برای آن برمیگزینید

    # inference('030', "unseen_text_seen_autt", "dataset/persian_data/train_data/speaker-030/book-1/utterance-00830-021555-1.wav", "SIL AH A B L A H T A R I N D U S T AA N E M AA SIL KH A T A R N AA K T A R I N D O SH M A N AA N E M AA H A S T A N D SIL")  ## ابله ترین دوستان ما خطرناک ترین دشمنان ما هستند
    # inference('030', "unseen_text_seen_autt", "dataset/persian_data/train_data/speaker-030/book-1/utterance-00830-021545-1.wav", "SIL J AA Y I K E H AH E AH T E M AA D I V O J U D N A D AA R A D S O KH A N G O F T A N B I H U D E AH A S T SIL")  ## جایی که اعتمادی وجود ندارد سخن گفتن بیهوده است

    # inference('031', "unseen_text_seen_autt", "dataset/persian_data/train_data/speaker-031/book-1/utterance-00873-022640-1.wav", "SIL D AA N E SH E AH E L AA H I AH A Z R AA H E K E T AA B H AA K A S B N E M I SH A V A D SIL B A L K E B AA Y A D AH AA N R AA D A R V O J U D E KH O D D A R K K A R D SIL")  ## دانش الهی از راه کتاب ها کسب نمی شود بلکه باید آن را در وجود خود درک کرد
    # inference('031', "unseen_text_seen_autt", "dataset/persian_data/train_data/speaker-031/book-1/utterance-00873-022636-1.wav", "SIL AH E B N E S I N AA D A R K U D A K I N O B U Q E SH AA Y AA N I AH A Z KH O D B O R U Z D AA D SIL V A D A R J A V AA N I AH A Z D AA N E SH M A N D AA N E D A R B AA R SH O D SIL") ## ابن سینا در کودکی نبوغ شایانی از خود بروز داد و در جوانی از دانشمندان دربار شد

    # inference('038', "unseen_text_seen_autt", "dataset/persian_data/train_data/speaker-038/book-1/utterance-01069-025495-2.wav", "SIL AH U CH AA H AA R S A D O P A N J AA H K E T AA B N E V E SH T E AH A S T SIL K E H SH O M AA R E Z I Y AA D I AH A Z AH AA N H AA D A R M O R E D E F A L S A F E H AH A S T SIL") ## او ۴۵۰ کتاب نوشته است که شمار زیادی از آنها در مورد فلسفه است
    # inference('038', "unseen_text_seen_autt", "dataset/persian_data/train_data/speaker-038/book-1/utterance-01075-025602-1.wav", "SIL AH A Z SH AA H N AA M E AH I N G U N E B A R D AA SH T K A R D E H AH A N D SIL K E H F E R D O S I B AA Z A B AA N E P A H L A V I AH AA SH E N AA B U D E H AH A S T SIL")  ## از شاهنامه اینگوه برداشت کرده اند که فردوسی ب زبان پهلوی آشنا بوده است

    # inference('042', "unseen_text_seen_autt", "dataset/persian_data/train_data/speaker-042/book-1/utterance-01178-027592-1.wav", "SIL B I SH T A R E AH O M R E S A AH D I M O S AA D E F B AA H A M L E Y E M O Q O L B E AH I R AA N B U D E H AH A S T SIL")  ## بیشتر عمر سعدی مصادف با حمله مغول به ایران بوده است
    # inference('042', "unseen_text_seen_autt", "dataset/persian_data/train_data/speaker-042/book-1/utterance-01162-027289-1.wav", "SIL K E T AA B E G O L E S T AA N E S A AH D I B E H N A S R AH A S T V A Y E K S AA L P A S AH A Z B U S T AA N N E V E SH T E SH O D E AH A S T SIL")  ## کتاب گلستان سعدی به نثر است و یک سال پس از بوستان نوشته شده است

    # inference('053', "unseen_text_seen_autt", "dataset/persian_data/train_data/speaker-053/book-1/utterance-01448-034133-1.wav", "SIL M O L A V I D A R D A H E Y E G O Z A SH T E B E S O R AH A T D A R Q A R B M A SH H U R SH O D SIL V A K E T AA B A SH D A R AH AA M R I K AA P O R F O R U SH T A R I N K E T AA B E S AA L SH O D SIL")  ## مولوی در دهه ی گذشته در غرب به سرعت مشهور شد و کتابش در آمریکا پر فروش ترین کتاب سال شد
    # inference('053', "unseen_text_seen_autt", "dataset/persian_data/train_data/speaker-053/book-1/utterance-01451-034228-1.wav", "SIL P A S AH A Z SH AA H N AA M E D A R F A R AA M A R Z N AA M E H A M AH A Z R O S T A M Y AA D SH O D E H AH A S T SIL")  ## پس از شاهنامه در فرامرزنامه هم از رستم یاد شده است

    # inference('056', "unseen_text_seen_autt", "dataset/persian_data/train_data/speaker-056/book-1/utterance-01573-036320-1.wav", "SIL B AA AH AA N K E H H AA F E Z SH AA AH E R I M A SH H U R B U D E H SIL AH A M M AA SH A R H AH A H V AA L E AH U N AA SH E N AA KH T E M I B AA SH A D SIL")  ## با آن که حافظ شاعری مشهور بود اما شرح احوال او نا شناخته است
    # inference('056', "unseen_text_seen_autt", "dataset/persian_data/train_data/speaker-056/book-1/utterance-01573-036322-1.wav", "SIL AH A Z AH A V AA S E T E D A H E Y E H A F T AA D SIL B AA S O R AH A T G E R E F T A N E R A V A N D E S A B T E AH AA S AA R E M E L L Y SIL T A KH R I B E SH AA N H A M SH E D D A T G E R E F T SIL")  ## در اواسط دهه هفتاد با سرعت گرفتن ثبت آثار ملی تخریبشان هم شدت گرفت

    # inference('057', "unseen_text_seen_autt", "dataset/persian_data/train_data/speaker-057/book-1/utterance-01587-036400-13.wav", "SIL B O R J E AH AA Z AA D I N A M AA D E AH A S L I Y E SH A H R E T E H R AA N AH A S T SIL V A T E H R AA N B E H AH I N N A M AA D SH E N AA KH T E M I SH A V A D SIL")  ## برج آزادی نماز شهر تهران است و تهران به این نماد شناخته می شود
    # inference('057', "unseen_text_seen_autt", "dataset/persian_data/train_data/speaker-057/book-1/utterance-01579-036365-1.wav", "SIL D A R AH I N B A N AA SIL KH O T U T E M O V AA Z I Y E P AA Y E H AA SIL Y AA D AH AA V A R E S A B K E M E AH M AA R I Y E H A KH AA M A N E SH I AH A S T SIL")  ## در این بنا خطوط موازی پایه ها یاداور سبک معماری هخامنشی است

    # inference('063', "unseen_text_seen_autt", "dataset/persian_data/train_data/speaker-063/book-1/utterance-01798-040099-2.wav", "SIL N AA M E T AA R I KH I Y E AH I N P A H N E Y E AH AA B I SIL H A M V AA R E B AA N AA M E S A R Z A M I N E AH I R AA N P E I V A N D D AA SH T E AH A S T SIL")  ## نام تاریخی این پهنه آبی همواره با نام سرزمین ایران پیوند داشته است
    # inference('063', "unseen_text_seen_autt", "dataset/persian_data/train_data/speaker-063/book-1/utterance-01797-040093-3.wav", "SIL M O L AA N AA SIL N E Z AA AH E B E Y N E M O AH T A Q E D I N B E J A B R V A M O AH T A Q E D I N B E AH E KH T I Y AA R R AA N AA SH I AH A Z H O K M E AH E L AA H I M I D AA N A D SIL")  ## مولانا نزاع بین معتقدین به جبر و معتقدین به اختیار را ناشی از حکم الهی می داند

    # print("UNSEEN text and UNSEEN autterance and non-parallel(not correlated):")
    # ## UNSEEN text and UNSEEN autterance and non-parallel(not correlated):
    # inference('004', "unseen_text_unseen_autt_nonpara", "dataset/persian_data/test_data/speaker-004/book-1/utterance-00048-001423-1.wav", "SIL N AA D E R E AH E B R AA H I M I CH A N D I N F I L M E M O S T A N A D SIL V A H A M CH E N I N D O M A J M U AH E Y E T E L V E Z I Y U N I R AA N E V E SH T E AH A S T SIL") ## نادر ابراهیمی چندین فیلم مستند و همچنین دو مجموعه تلویزیونی را نوشته است
    # inference('004', "unseen_text_unseen_autt_nonpara", "dataset/persian_data/test_data/speaker-004/book-1/utterance-00052-001517-1.wav", "SIL SH I R AA Z B E AH O N V AA N E Y E K I AH A Z M O H E M T A R I N M A R AA K E Z E G A R D E SH G A R I V A T U R I S T I Y E AH I R AA N M A T R A H B U D E AH A S T SIL")  ## شیراز به عنوان یکی از مهمترین مراکز گردشگری و ایران مطرح بوده است
 
    # inference('006', "unseen_text_unseen_autt_nonpara", "dataset/persian_data/test_data/speaker-006/book-1/utterance-00168-004841-1.wav", "SIL T E H R AA N V A T A B R I Z SIL AH A V V A L I N SH A H R H AA Y I B U D A N D K E H N A H AA D E SH A H R D AA R I D A R AH AA N H AA T A AH S I S SH O D SIL")  ## تهران و تبریز اولین شهرهایی بودند که نهاد شهرداری در آنها تاسیس شد
    # inference('006', "unseen_text_unseen_autt_nonpara", "dataset/persian_data/test_data/speaker-006/book-1/utterance-00168-004843-1.wav", "SIL F I L M E B A CH CH E H AA Y E AH AA S E M AA N B AA F I L M E D O Z D E D O CH A R KH E M O Q AA Y E S E SH O D SIL V A M O R E D E T A H S I N Q A R AA R G E R E F T SIL")  ## فیلم بچه های آسمان با فیلم دزد دوچرخه مقایسه شد و مورد تحسین قرار گرفت
 
    # inference('008', "unseen_text_unseen_autt_nonpara", "dataset/persian_data/test_data/speaker-008/book-1/utterance-00207-006575-1.wav", "SIL D A R S A D E Y E N U Z D A H O M SIL AH I N SH A H R SIL D AA R AA Y E Y E K H E S AA R V A Q A L AH E Y E N E Z AA M I J A H A T E H E F AA Z A T AH A Z H A M A L AA T B U D SIL")  ## در سده نوزدهم مشهد دارای یک حصار و قلعه نظامی جهت حفاظت از حملات بود
    # inference('008', "unseen_text_unseen_autt_nonpara", "dataset/persian_data/test_data/speaker-008/book-1/utterance-00209-006609-1.wav", "SIL H A R CH A N D KH A Z A R Q A R N H AA S T B E H D A R Y AA SH E N AA KH T E M I SH A V A D SIL V A L I B E H H I CH D A R Y AA Y I M O T T A S E L N I S T SIL")  ## هرچند خزر قرن هاست ب دریا شناخته می شود ولی به هیچ دریایی متصل نیست
 
    # inference('011', "unseen_text_unseen_autt_nonpara", "dataset/persian_data/test_data/speaker-011/book-1/utterance-00284-008935-1.wav", "SIL J A N G E J A H AA N I Y E AH A V V A L D A R AH I R AA N SIL H A M Z A M AA N B AA H O K U M A T E AH A H M A D SH AA H B U D SIL")  ## جنگ جهانی اول در ایران همزان با حکومت احمد شاه بود
    # inference('011', "unseen_text_unseen_autt_nonpara", "dataset/persian_data/test_data/speaker-011/book-1/utterance-00286-008968-1.wav", "SIL D A R N A T I J E Y E T A D AA V O M E M O B AA R E Z E Y E K AA R E G A R AA N SIL V A P E I V A S T A N E M A R D O M  B E J O N B E SH SIL M E L L I SH O D A N E S A N AH A T E N A F T M O H A Q Q A Q SH O D SIL")  ## در نتیجه مبارزه کارگردان و پیوستن مردم به جنبش ملی شدن صنعت نفت محقق شد

    # inference('017', "unseen_text_unseen_autt_nonpara", "dataset/persian_data/test_data/speaker-017/book-1/utterance-00447-012444-1.wav", "SIL D A R S AA L E B A AH D M O S A D D E Q B E D AA D G AA H E L AA H E R A F T SIL T AA B E SH E K AA Y A T E SH E R K A T E AH E N G E L I S I P AA S O KH D A H A D SIL")  ## در سال بعد مصدق به دادگاه لاهه رفت تا به شکایت شرکت انگلیسی پاسخ دهد
    # inference('017', "unseen_text_unseen_autt_nonpara", "dataset/persian_data/test_data/speaker-017/book-1/utterance-00447-012446-3.wav", "SIL T I M E M E L L I Y E F U T B AA L E AH I R AA N SIL T AA K O N U N S E B AA R Q A H R A M AA N E J AA M E M E L L A T H AA Y E AH AA S S I Y A SH O D E AH A S T SIL")  ## تیم ملی فوتبال ایران تا کنون سه بار قهرمان جام ملت های آسیا شده است

    # inference('023', "unseen_text_unseen_autt_nonpara", "dataset/persian_data/test_data/speaker-023/book-1/utterance-00610-016590-1.wav", "SIL B AA Z I B AA T A SH V I Q E H A V AA D AA R AA N V A KH O SH U N A T E B AA Z I K O N AA N SIL B E S U D E M I Z B AA N D A R J A R Y AA N B U D SIL")  ## بازی با تشویق هواداران و خشونت بازیکنان در جریان بود
    # inference('023', "unseen_text_unseen_autt_nonpara", "dataset/persian_data/test_data/speaker-023/book-1/utterance-00612-016635-3.wav", "SIL T I M E V AA L I B AA L E AH I R AA N H O D U D E S I S AA L E P I SH SIL B A R AA Y E AH A V V A L I N B AA R D A R AH E N T E KH AA B I Y E AH O L A M P I K SH E R K A T K A R D SIL")  ## تیم والیبال ایران حدود ۳۰ سال پیش برای اولین بار در انتخابیه المپیک شرکت کرد

    # inference('025', "unseen_text_unseen_autt_nonpara", "dataset/persian_data/test_data/speaker-025/book-1/utterance-00678-017882-1.wav", "SIL T A R K I B AA T E SH I M I Y AA Y I E AH AA L I N I Z M O M K E N AH A S T M A V AA D D E M O Q A Z Z I D A R N A Z A R G E R E F T E SH A V A N D SIL")  ## ترکیبات شیمیایی الی نیز ممکن است مواد مغذی در نظر گرفته شوند
    # inference('025', "unseen_text_unseen_autt_nonpara", "dataset/persian_data/test_data/speaker-025/book-1/utterance-00684-018072-1.wav", "SIL M O T E AH A S S E F AA N E H AH E M R U Z E AH A F R AA D E B E S I Y AA R K A M I V O J U D D AA R A N D K E AH A Z L E B AA S H AA Y E M A H A L L I AH E S T E F AA D E K O N A N D SIL") ## متاسفانه امروزه افراد بسیار کمی وجود دارند که از لباس های محلی استفاده کنند

    # inference('028', "unseen_text_unseen_autt_nonpara", "dataset/persian_data/test_data/speaker-028/book-1/utterance-00781-020815-1.wav", "SIL D A R T AA R I KH E AH A F S AA N E AH I Y E AH I R AA N AH AA M A D E AH A S T SIL K E H P O KH T E Q A Z AA AH A Z D O R AA N E P AA D E SH AA H I Y E Z A H H AA K AH AA Q AA Z SH O D SIL")  ## در تاریخ افسانه ای ایران آمده است که پخت غذا از دوران پادشاهی ضحاک آغاز شد
    # inference('028', "unseen_text_unseen_autt_nonpara", "dataset/persian_data/test_data/speaker-028/book-1/utterance-00783-020857-1.wav", "SIL B E S I Y AA R I AH A Z F I B R H AA B E V A S I L E Y E D A S T G AA H E G A V AA R E SH J A Z B N E M I SH A V A N D SIL")  ## بسیاری از فیبرها به وسیله دستگاه گوارش جذب نمی شوند

    # inference('030', "unseen_text_unseen_autt_nonpara", "dataset/persian_data/test_data/speaker-030/book-1/utterance-00830-021538-1.wav", "SIL T O R K A M A N H AA Y E AH I R AA N D A R AH O S T AA N E G O L E S T AA N V A B A KH SH I D A R KH O R AA S AA N E SH O M AA L I S O K U N A T D AA R A N D SIL")  ## ترکمن های ایران در استان گلستان و بخشی در خراسان شمالی سکونت دارند
    # inference('030', "unseen_text_unseen_autt_nonpara", "dataset/persian_data/test_data/speaker-030/book-1/utterance-00832-021595-4.wav", "SIL P U SH AA K E KH O R AA S AA N I B E H AH A N V AA AH E L E B AA S H AA Y E M A H A L L I Y E AH A H AA L I Y E KH O R AA S AA N AH E T L AA Q M I SH A V A D SIL")  ## پوشاک خراسانی به انواع لباس های محلی اهالی خراسان اطلاق می شود

    # inference('031', "unseen_text_unseen_autt_nonpara", "dataset/persian_data/test_data/speaker-031/book-1/utterance-00851-022197-1.wav", "SIL AH A S L I T A R I N T A H D I D E P I SH E R U Y E P A L A N G E AH I R AA N I SIL Q A T AH SH O D A N E Z I S T G AA H H AA Y E AH I N J AA N E V A R AH A S T SIL")  ## اصلی ترین تهدید پیش روی پلنگ ایرانی قطع شدن زیستگاه های این جانور است
    # inference('031', "unseen_text_unseen_autt_nonpara", "dataset/persian_data/test_data/speaker-031/book-1/utterance-00861-022419-2.wav", "SIL AH A R A B H AA CH AA H AA R H A R F E AH A S L I R AA D A R AH A L E F B AA Y E KH O D N A D AA SH T A N D SIL V A AH AA N R AA H O R U F E AH A J A M I M I N AA M I D A N D SIL")  ## عرب ها چهار حرف اصلی را در الفبای خود نداشتند و آن را حروف عجمی می نامیدند

    # inference('038', "unseen_text_unseen_autt_nonpara", "dataset/persian_data/test_data/speaker-038/book-1/utterance-01064-025401-1.wav", "SIL Q A R N H AA Y E N O H O M T AA Y AA Z D A H O M E H E J R I SIL Q A R N H AA Y E D E R A KH SH AA N D A R H O N A R E KH O SH N E V I S I Y E AH I R AA N M I B AA SH A D SIL")  ## قرن های نهم تا یادهم هجری قرن های درخشان در خوشنویسی ایران است
    # inference('038', "unseen_text_unseen_autt_nonpara", "dataset/persian_data/test_data/speaker-038/book-1/utterance-01065-025407-1.wav", "SIL P A S AH A Z M A R G E N AA D E R SH AA H SIL P E Y K A R A SH B E H AH AA R AA M G AA H I K E H SIL P I SH T A R B A R AA Y E KH O D A SH S AA KH T E B U D M O N T A Q E L SH O D SIL")  ## پس از مرگ نادر شاه پیکرش به آرامگاهی که پیشتر برای خودش ساخته بود منتقل شد

    # inference('042', "unseen_text_unseen_autt_nonpara", "dataset/persian_data/test_data/speaker-042/book-1/utterance-01161-027274-1.wav", "SIL L A SH K A R H AA Y E H E Z AA R T AA Y I Y E AH A SH K AA N I D AA R AA Y E P A R CH A M I AH A B R I SH A M I B AA N E SH AA N E AH E ZH D E H AA B U D A N D SIL")  ## لشکرهایی هزارتایی اشکانی دارای پرچم های ابریشمی با نشان اژدها بودند
    # inference('042', "unseen_text_unseen_autt_nonpara", "dataset/persian_data/test_data/speaker-042/book-1/utterance-01165-027352-1.wav", "SIL B I SH A K Q E D M A T E F A R SH B E H Z A M AA N I B A R M I G A R D A D SIL K E H AH E N S AA N B E F E K R E AH AA S AA Y E SH E KH O D AH O F T AA D SIL")  ## بی شک قدمت فرش به زمانی برمیگردد که انسان به فکر اسایش خود افتاد

    # inference('053', "unseen_text_unseen_autt_nonpara", "dataset/persian_data/test_data/speaker-053/book-1/utterance-01442-034005-1.wav", "SIL KH O SH K S AA L I D A R AH I N D O R E H SIL B A KH SH E B O Z O R G I AH A Z K AA R AA Y I V A T A V AA N E K E SH AA V A R Z I R AA AH A Z B E I N B O R D SIL")  ## خشک سالی در این دوره بخش بزرگی از کارایی و کشاورزی را از بین برد
    # inference('053', "unseen_text_unseen_autt_nonpara", "dataset/persian_data/test_data/speaker-053/book-1/utterance-01442-034007-1.wav", "SIL T A Q A Z Z O L H AA Y E AH A V V A L I Y E R A N G O B U Y E V AA Q E AH G A R AA Y AA N E D AA SH T A N D SIL")  ## تغزل های اولیه رنگ و بوی واغع گرایانه داشتند

    # inference('056', "unseen_text_unseen_autt_nonpara", "dataset/persian_data/test_data/speaker-056/book-1/utterance-01546-036128-13.wav", "SIL AH U AH A S B I B E N AA M E R A KH SH D AA SH T K E H D A R T A M AA M E N A B A R D H AA SIL Y AA R O Y AA V A R A SH B U D SIL")  ## او اسبی به نام رخش داشت که در تمام نبردها یار و یاورش بود
    # inference('056', "unseen_text_unseen_autt_nonpara", "dataset/persian_data/test_data/speaker-056/book-1/utterance-01546-036128-14.wav", "SIL D A R D O R E Y E P A H L A V I SIL M U S I Q I AH A Z D A R B AA R F AA S E L E G E R E F T V A B E KH AA N E H AA Y E M A R D O M AH AA M A D SIL")  ## در دوره پهلوی موسیقی از دربار فاصله گرفت و به خانه های مردم آمد

    # inference('057', "unseen_text_unseen_autt_nonpara", "dataset/persian_data/test_data/speaker-057/book-1/utterance-01581-036375-3.wav", "SIL R A N G H AA Y E AH A S L I S E R A N G I H A S T A N D K E H T A R K I B E H I CH K O D AA M AH A Z R A N G H AA AH AA N H AA R AA N E M I S AA Z A D SIL")  ## رنگ های اصلی سه رنگی هستند که ترکیب هیچ رنگی آن ها را نمی سازد
    # inference('057', "unseen_text_unseen_autt_nonpara", "dataset/persian_data/test_data/speaker-057/book-1/utterance-01584-036394-4.wav", "SIL AH E Q L I M D A R N O Q AA T E M O KH T A L E F SIL B AA AH A R Z E J O Q R AA F I Y AA Y I V A AH E R T E F AA AH M O SH A KH A S M I SH A V A D SIL")  ## اقلیم در نقاط مختلف با عرض جغرافیایی و ارتفاع مشخص می شود
 
    # inference('063', "unseen_text_unseen_autt_nonpara", "dataset/persian_data/test_data/speaker-063/book-1/utterance-01794-039985-2.wav", "SIL R A N G H AA D AA R AA Y E M O SH A KH A S E AH I B E N AA M E S A R D I V A G A R M I H A S T A N D SIL")  ## رنگ ها دارای مشخصه ای به نام گرمی و سردی هستند
    # inference('063', "unseen_text_unseen_autt_nonpara", "dataset/persian_data/test_data/speaker-063/book-1/utterance-01795-040025-2.wav", "SIL K A H K E SH AA N E R AA H E SH I R I SIL D AA R AA Y E B I SH AH A Z S A D M E L Y U N S E T AA R E M I B AA SH A D SIL")  ## کهکشان راه شیری دارای بیش از صد میلیون ستاره می باشد


    # #### New Speaker ####

    # print("UNSEEN text and UNSEEN speaker and non-parallel(not correlated):")
    # ## UNSEEN text and UNSEEN speaker and non-parallel(not correlated): 

    # inference('067', "unseen_text_unseen_speaker_nonpara", "dataset/persian_data/test_data/unseen_speakers/speaker-067/1.wav", "SIL CH E SH M V A D O R B I N SIL H A R D O M I T A V AA N A N D M I Z AA N E N U R E V O R U D I R AA K O N T O R O L K O N A N D SIL")  ## چشم و دوربین هر دو می توانند میزان نور ورودی را کنترل کنند
    # inference('067', "unseen_text_unseen_speaker_nonpara", "dataset/persian_data/test_data/unseen_speakers/speaker-067/2.wav", "SIL B A AH Z I AH A Z AH A F R AA D B E S O R AH A T B E KH AA B M I R A V A N D V A D A R B A R KH I D I G A R AH I N AH A M R T A D R I J I AH A S T SIL")  #بعضی از افراد به سرعت به خواب می روند و در برخی دیگر این امر تدریجی است 
    
    # inference('068', "unseen_text_unseen_speaker_nonpara", "dataset/persian_data/test_data/unseen_speakers/speaker-068/3.wav", "SIL M I V E Y E SH AA T U T B O Z O R G T A R AH A Z T U T E S E F I D AH A S T V A R A N G E AH AA N Q E R M E Z E T I R E AH A S T SIL")  ## میوه شاتوت بزرگتر از توت سفید است و رنگ آن قرمز تیره است
    # inference('068', "unseen_text_unseen_speaker_nonpara", "dataset/persian_data/test_data/unseen_speakers/speaker-068/4.wav", "SIL M E S Y E K I AH A Z CH AA H AA R F E L E Z I AH A S T K E H R A N G E T A B I AH I Y E AH AA N KH AA K E S T A R I Y AA N O Q R E AH I N I S T SIL")  ## مس یکی از چهار فلزی است که رنگ طبیعی آن خاکستری یا نقره ای نیست
    
    # inference('069', "unseen_text_unseen_speaker_nonpara", "dataset/persian_data/test_data/unseen_speakers/speaker-069/5.wav", "SIL B E H G O F T E Y E T AA R I KH SIL M E S R I Y AA N E B AA S T AA N SIL AH A V V A L I N M A R D O M AA N I B U D A N D K E H K AA Q A Z R AA S AA KH T A N D SIL")  ##  به گفته تاریخ مصریان باستان اولین مردمانی بودند که کاغذ را ساختند
    # inference('069', "unseen_text_unseen_speaker_nonpara", "dataset/persian_data/test_data/unseen_speakers/speaker-069/6.wav", "SIL B E L AA K CH E Y N SIL Z A N J I R E AH I AH A Z D AA D E H AA S T SIL K E K A S I N E M I T A V AA N A D CH I Z I D A R AH AA N T A Q I R D A H A D SIL")  ## بلاک چین زنجیره ای از داده هاست که کسی نمی تواند چیزی در آن تغییر دهد 
    
    # inference('070', "unseen_text_unseen_speaker_nonpara", "dataset/persian_data/test_data/unseen_speakers/speaker-070/7.wav", "SIL N A S L E P A N J O M V A SH E SH O M E AH I N T E R N E T Y A AH N I SIL Y E K AH E T T E S AA L E J A H AA N I Y E S A R I AH T A R V A P AA Y D AA R T A R SIL")  ## نسل پنجم و ششم اینترنت یعنی یک اتصال سریعتر و پایدارتر
    # inference('070', "unseen_text_unseen_speaker_nonpara", "dataset/persian_data/test_data/unseen_speakers/speaker-070/8.wav", "SIL G E R AA F E N B AA V O J U D E S A B O K I Y E Z I Y AA D I K E D AA R A D SIL S A D T AA S I S A D B A R AA B A R M O H K A M T A R AH A Z F U L AA D M I B AA SH A D SIL") ## گرافن با وجود سبکی که دارد صد تا سیصد برابر سبک تر از فولاد است
    
    # inference('071', "unseen_text_unseen_speaker_nonpara", "dataset/persian_data/test_data/unseen_speakers/speaker-071/9.wav", "SIL K AA M P I Y U T E R H AA Y E K O V AA N T O M I SIL S Y S T E M H AA Y I B AA Q O D R A T V A S O R AH A T E Q E Y R E Q AA B E E L E T A S A V V O R H A S T A N D SIL")  ## کامپیوترهای کوانتومی سیستمهایی با قدرت و سرعت غیر قابل تصور هستند
    # inference('071', "unseen_text_unseen_speaker_nonpara", "dataset/persian_data/test_data/unseen_speakers/speaker-071/10.wav", "SIL AH A G A R D A R H A M AA N Z A M AA N E H A Y AA T E AH I N D AA N E SH M A N D B E AH U AH A H A M I Y A T M I D AA D A N D SIL D O N Y AA B I SH T A R P I SH R A F T M I K A R D SIL")  ## اگر در همان زمان حیات این دانشمند به او اهمیت می دادند دنیا بیشتر پیشرفت میکرد
    
    # inference('072', "unseen_text_unseen_speaker_nonpara", "dataset/persian_data/test_data/unseen_speakers/speaker-072/11.wav", "SIL Y E K I AH A Z M O H E M T A R I N G AA M H AA Y E R E S I D A N B E M O V A F F A Q I Y Y A T SIL D AA SH T A N E T A AH A H H O D AH A S T SIL")  ## یکی از مهمترین گام های رسیدن به موفقیت داشتن تعهد است
    # inference('072', "unseen_text_unseen_speaker_nonpara", "dataset/persian_data/test_data/unseen_speakers/speaker-072/12.wav", "SIL J A N G A L H AA Y E H I R K AA N I Y E AH I R AA N V A AH AA Z A R B AA Y J AA N AH A Z M O H E M T A R I N M A N AA T E Q E Z I S T K O R E D A R J A H AA N H A S T A N D SIL")  ## جنگل های هیرکانی در ایران و آذربایجان از مهمترین مناطق زیست کره در جهان هستند
    
    # inference('073', "unseen_text_unseen_speaker_nonpara", "dataset/persian_data/test_data/unseen_speakers/speaker-073/13.wav", "SIL M A N AA T E Q E Z I Y AA D I AH A Z J A H AA N T A H T E T A AH S I R E B AA R AA N H AA Y E AH A S I D I H A S T A N D SIL")  ## مناطق زیادی از جهان تحت تاثیر باران های اسیدی هستند
    # inference('073', "unseen_text_unseen_speaker_nonpara", "dataset/persian_data/test_data/unseen_speakers/speaker-073/14.wav", "SIL T A N H AA P E S T AA N D AA R AA N I K E Q AA D E R B E P A R V AA Z M I B AA SH A N D KH O F F AA SH H AA H A S T A N D SIL")  ## تنها پستاندارانی که قادر به پرواز هستند خفاش ها هستند
    
    # inference('074', "unseen_text_unseen_speaker_nonpara", "dataset/persian_data/test_data/unseen_speakers/speaker-074/15.wav", "SIL SH AA Y A D B E T A V AA N G O F T SH A H R I Y AA R SIL AH A N D I SH E H AA Y E H AA F E Z R AA B E Z A B AA N E S A AH D I B A Y AA N K A R D E H AH A S T SIL")  ## شاید بتوان گفت شهریار اندیشه های سعدی را به زبان حافظ بیان کرده است
    # inference('074', "unseen_text_unseen_speaker_nonpara", "dataset/persian_data/test_data/unseen_speakers/speaker-074/16.wav", "SIL B I SH A K Y E K I AH A Z M O H E M T A R I N SH A H R H AA D A R H O Z E Y E M E AH M AA R I SIL SH A H R E Y A Z D AH A S T SIL")  ## بی شک یکی از مهمترین شهر ها در حوزه معماری شهر یزد است
    
    # inference('075', "unseen_text_unseen_speaker_nonpara", "dataset/persian_data/test_data/unseen_speakers/speaker-075/17.wav", "SIL AH I N AH AA B SH AA R D A R AH E M T E D AA D E Y E K SH I B E P E L L E K AA N I SIL B E S U R A T E T A B I AH I AH A Z S A KH R E H AA F O R U M I R I Z A D SIL")  ## این آبشار در امتداد یک شیب پلکانی به صورت طبیعی از صخره ها فرو میریزد
    # inference('075', "unseen_text_unseen_speaker_nonpara", "dataset/persian_data/test_data/unseen_speakers/speaker-075/18.wav", "SIL S A Y Y AA R E Y E Z O H R E Q A M A R N A D AA R A D SIL V A D O V V O M I N S A Y Y AA R E Y E N A Z D I K B E KH O R SH I D AH A S T SIL")  ## سیاره ی زهره قمر ندارد و دومین سیاره نزدیک به خورشید است
    
    # inference('076', "unseen_text_unseen_speaker_nonpara", "dataset/persian_data/test_data/unseen_speakers/speaker-076/19.wav", "SIL P A S T T A R I N N O Q T E Y E D A SH T E L U T H O D U D E D E V I S T O CH E H E L M E T R AH A Z S A T H E D A R Y AA B AA L AA T A R AH A S T SIL")  ## پست‌ترین نقطه دشت لوت حدود ۲۴۰ متر از سطح دریا بالاتر است
    # inference('076', "unseen_text_unseen_speaker_nonpara", "dataset/persian_data/test_data/unseen_speakers/speaker-076/20.wav", "SIL M O D D A T Z A M AA N E B I D AA R I Y E Y E K SH A KH S E B AA L E Q SIL CH AA H AA R B A R AA B A R E Y E K K U D A K AH A S T SIL")  ## مدت زمان بیداری یک شخص بالغ ۴ برابر یک کودک است 

if __name__ == "__main__":
   main()
    
