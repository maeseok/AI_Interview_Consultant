import os
import numpy as np
import librosa
import re
import pandas as pd
from IPython.display import Audio
import soundfile as sf
import noisereduce as nr
from konlpy.tag import Okt, Komoran
import warnings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import random
from difflib import SequenceMatcher
from transformers import pipeline
from random import sample
from sentence_transformers import SentenceTransformer, util
from google.oauth2 import service_account
from google.cloud import speech
import io
import openai


warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
# 서비스 계정 인증 설정
credentials = service_account.Credentials.from_service_account_file(
    "./my-project-stt-441803-1c4103572f18.json"
)

result={}

file_path = './직무별_면접_질문_리스트_한글변환.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1')


# 산업과 직무 매핑
industry_map = {
    "0": "공통질문",
    "1": "경영관리",
    "2": "영업마케팅",
    "3": "공공서비스",
    "4": "연구개발",
    "5": "ICT",
    "6": "디자인",
    "7": "생산제조",
    "8": "의료"
}

job_mapping = {
    "경영관리": ["경영기획", "웹기획PM", "DBA", "ERP"],
    "영업마케팅": ["마케팅", "영업고객상담", "웹마케팅", "서비스"],
    "공공서비스": ["건설", "관광레저", "미디어문화"],
    "연구개발": ["연구개발", "연구", "빅데이터AI", "소프트웨어", "하드웨어"],
    "ICT": ["UI", "IT강사", "QA", "게임개발자", "네트워크서버보안", "시스템프로그래머", "웹디자인", "웹프로그래머", "응용프로그래머", "웹운영자"],
    "디자인": ["디자인", "영상제작편집"],
    "생산제조": ["무역유통"],
    "의료": ["의료질문"]
}

# 오디오 전처리 코드
def preprocess(audio_data):
    y_denoised = nr.reduce_noise(audio_data, sr=16000)
    rms_level = librosa.feature.rms(y=y_denoised)[0]
    rms_mean = rms_level.mean()

    if rms_mean < 0.05:
        gain_factor = 0.05 / rms_mean
        y_denoised = y_denoised * gain_factor
    # 노이즈 제거 일단 보류
    y_trimmed, _ = librosa.effects.trim(y_denoised, top_db=20)
    return y_trimmed

# 오디오 전처리 적용 및 전처리 안한 데이터도 따로 보관하는 코드
def preprocess_audio(file_path):
    try:
        audio_data, sr = librosa.load(file_path, sr=16000)
    except Exception as e:
        print(f"오디오 파일을 로드하는 중 오류 발생: {e}")
        return

    # 전처리 되기 전 오디오 데이터로 피드백 생성
    y = audio_data

    # 오디오 전처리
    audio_data = preprocess(audio_data)
    # 전처리전 오디오 저장 - 전처리하면 STT 인식 잘 안됨
    try:
        sf.write('./preprocess_audio.wav', y, sr)
        print(f"전처리전 오디오 파일이 저장되었습니다: ")
    except Exception as e:
        print(f"전처리전 오디오 파일 저장 중 오류 발생: {e}")
    return y, audio_data, sr, librosa.get_duration(y=audio_data, sr=sr)  # 추가된 부분

# 음성 파일을 비동기 방식으로 STT 변환하는 함수
def transcribe(audio_file):
    client = speech.SpeechClient(credentials=credentials)  # credentials 직접 사용

    # 음성 파일을 읽어 Google STT에 전송
    with io.open(audio_file, "rb") as f:
        content = f.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,  # 인코딩 형식
        sample_rate_hertz=16000,  # 샘플링 레이트를 WAV 파일의 샘플 레이트로 설정
        language_code="ko-KR",  # 한국어 설정
        enable_word_time_offsets=True  # 단어별 타임스탬프 활성화
    )

    # LongRunningRecognize API 요청
    operation = client.long_running_recognize(config=config, audio=audio)
    print("Waiting for operation to complete...")
    response = operation.result(timeout=600)  # 10분 타임아웃 설정

    # 타임스탬프를 문장 단위로 분리
    segments = []
    sentence_words = []
    sentence_start_time = None
    text = ""

    for result in response.results:
        alternative = result.alternatives[0]


        for word_info in alternative.words:
            word = word_info.word
            start_time = word_info.start_time.total_seconds()
            end_time = word_info.end_time.total_seconds()

            # 문장 시작 시간 설정
            if sentence_start_time is None:
                sentence_start_time = start_time

            sentence_words.append(word)
            text += word + " "
            sentence_endings = [
                ".", "?", "!", "다", "요", "입니다", "어요", "니까", "어", 
                "습니까", "네요", "군요", "겠네요", "합니다"
            ]
            # 문장의 끝 판단 로직
            if any(word.endswith(ending) for ending in sentence_endings):  # 단어가 특정 끝말로 끝나는지 확인
                # 다음 단어가 공백이거나 구두점으로 끝날 경우만 문장 끝으로 판단
                if word_info == alternative.words[-1] or (
                    len(alternative.words) > 1 and
                    not alternative.words[alternative.words.index(word_info) + 1].word[0].isalnum()
                ):
                    sentence_text = " ".join(sentence_words)
                    segments.append({
                        "text": sentence_text.strip(),
                        "start_time": sentence_start_time,
                        "end_time": end_time
                    })
                    # 문장 정보 초기화
                    sentence_words = []
                    sentence_start_time = None

    return text, segments


# 말의 빠르기 : 음절 수 계산 함수 (한글만 고려)
def count_syllables(text):
    okt = Okt()
    morphs = okt.morphs(text)
    syllables = sum(len(word) for word in morphs if any(c.isalpha() for c in word))
    return syllables

# 말의 빠르기 :  문장별 SPM 계산 함수 (=음절/발화시간)
def calculate_sentence_spm(sentence, duration):
    total_syllables = count_syllables(sentence)
    spm = (total_syllables / duration) * 60  # SPM 계산
    return spm


# 말의 빠르기 : spm에 기반해 문장별 말의 빠르기 평가 (빠르기 기준 선정에는 1000개의 면접 데이터 활용)
# 266.41903531438413 297.041079487888 331.9210823501441 382.7353522053983
def evaluate_speed(spm):
    if spm < 266.42 :   # 266.42 : 말의 속도 상위 0.8 ~1
        result['speed'] =  "느림 말의 속도감 있는 전달이 필요합니다.\n"
    elif 266.42 <= spm < 297.04 :  # 297 : 말의 속도 상위 0.6 ~ 0.8
        result['speed'] = "적절"
    elif 297.04 <= spm < 331.92 :  # 331.92 : 말의 속도 상위 0.4 ~ 0.6
        result['speed'] = "적절"
    elif  331.92 <= spm < 382.74 :   # 382.74 : 말의 속도 상위 0.2 ~ 0.4
        result['speed'] = "주의 말의 속도가 빠른 편 입니다. 주의가 필요합니다.\n"
    else:  # 말의 속도 상위 0.0 ~ 0.2
        result['spped'] = "위험 말의 속도가 굉장히 빠릅니다. 차분한 전달이 필요합니다.\n"

# 목소리 크기 : RMS 기반 문장 단위 목소리 변화 계산 함수
def analyze_sentence_volume(audio_data, start, end, sr=16000):
    segment_audio = audio_data[int(start * sr):int(end * sr)]
    rms = np.sqrt(np.mean(segment_audio ** 2))
    return rms

def find_silence(audio_data, sr):
    # 앞뒤의 침묵을 제거
    trimmed_audio, _ = librosa.effects.trim(audio_data, top_db=20)

    # 에너지 계산
    energy = librosa.feature.rms(y=trimmed_audio)
    threshold = 0.001
    silence_indices = np.where(energy < threshold)[1]
    silence_times = librosa.frames_to_time(silence_indices, sr=sr)
    frame_duration = librosa.get_duration(y=trimmed_audio, sr=sr) / len(energy[0])

    # 전체 침묵 시간 계산
    total_silence_time = round(frame_duration * len(silence_times), 2)
    total_audio_time = len(trimmed_audio) / sr
    silence_percentage = round((total_silence_time / total_audio_time) * 100, 2)

    # 연속된 긴 침묵 구간 계산 (4초 이상)
    silence_durations = np.diff(silence_times)
    long_silences = silence_durations[silence_durations > 4]

    return total_audio_time, long_silences, silence_percentage

def silence_feedback(total_silence_time, total_audio_time, silence_percentage, long_silence_exists):
    # 무음 비율 기준
    if silence_percentage < 10:
        if not long_silence_exists:
            result['break_time']="훌륭"
        else:
            result['break_time']=(f"훌륭 총 녹음 시간인 {total_audio_time} 초 동안, 휴지 구간이 거의 없지만 긴 무음 구간이 발견되었습니다. 더욱 매끄러운 답변을 위해 연습이 필요합니다.1")
    elif 10 <= silence_percentage < 20:
        if not long_silence_exists:
            result['break_time']=(f"적절 녹음 시간인 {total_audio_time} 초 동안, 답변 중 약간의 휴지 구간이 있었습니다. 연습을 통해 개선할 수 있습니다.2")
        else:
            result['break_time']=(f"적절 녹음 시간인 {total_audio_time} 초 동안, 약간의 휴지 구간이 있었고 긴 무음 구간이 발견되었습니다. 긴 생각을 피하기 위해 연습이 필요합니다.3")
    elif 20 <= silence_percentage < 30:
        if not long_silence_exists:
            result['break_time']=(f"주의 녹음 시간인 {total_audio_time} 초 동안, 휴지 구간이 꽤 자주 발생했습니다. 더 매끄럽게 이어가는 연습이 필요합니다.4")
        else:
            result['break_time']=(f"주의 녹음 시간인 {total_audio_time} 초 동안, 휴지 구간이 꽤 자주 발생했으며 긴 무음 구간이 발견되었습니다. 답변을 더 간결하게 만드는 연습이 필요합니다.5")
    else:
        if not long_silence_exists:
            result['break_time']=(f"위험 녹음 시간인 {total_audio_time} 초 동안, 휴지 구간이 잦습니다. 답변을 더욱 간결하게 만드는 연습이 필요합니다.6")
        else:
            result['break_time']=(f"위험 녹음 시간인 {total_audio_time} 초 동안, 휴지 구간이 잦고, 긴 무음 구간이 발견되었습니다. 유창한 답변을 위해 많은 연습이 필요합니다.7")


# 간투어 탐지 함수
def detect_filler_words(transcribed_text):
    filler_words = ['음', '일단', '응', '왜 그러냐', '어', '그', '뭐', '저기', '사실', '아', '이제', '응', '어떻게', '그래서', '뭐랄까', '사실은', '아마', '그런데', '진짜', '왜냐면', '즉', '정말', '어쩌면', '뭐지', '하']
    filler_count = 0
    words = transcribed_text.split()

    for word in words:
        if word in filler_words:
            filler_count += 1

    return filler_count

# 피드백 생성 함수
def noise_feedback(filler_count):
    if filler_count == 0:
        result['meaningless'] = "없음"
    elif filler_count == 1:
        result['meaningless'] = "1회 간투어 1회 감지, 주의가 필요합니다.3"
    elif filler_count in [2, 3]:
        result['meaningless'] = "2회 간투어 2~3회 감지, 답변을 더욱 매끄럽게 만드는 연습이 필요합니다.6"
    else:  # 4개 이상
        result['meaningless'] = "4회 간투어 4회 이상, 보다 간결하고 유창한 답변을 위해 많은 연습이 필요합니다.8"
    
# 답변 유사도 측정
def similarity_feedback(question, answer):
    #질문-답변 유사도 측정 모델 설정
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # 질문과 답변의 유사도 측정
    question_embedding = model.encode(question, convert_to_tensor=True)
    answer_embedding = model.encode(answer, convert_to_tensor=True)

    similarity = util.pytorch_cos_sim(question_embedding, answer_embedding)
    similarity_score = similarity.item()
    print(similarity_score)

    if similarity_score < 0.5:
        result['similarity'] = "낮음 질문과 답변의 의미적 일치가 부족합니다.5"
    else:
        result['similarity'] = "높음"
    
# 꼬리질문 생성 
def process_text_with_langchain(transcription):
    # OpenAI API 키 설정
    os.environ["OPENAI_API_KEY"] = "APIKEY"

    # OpenAI 최신 모델 초기화
    llm = ChatOpenAI(model_name="gpt-4o-mini")

    # 텍스트 교정용 Prompt
    correction_prompt = PromptTemplate(
        input_variables=["text"],
        template = """
    다음 텍스트는 음성 인식을 통해 자동으로 전사된 문장으로, 발음 오류와 철자 오류가 포함되어 있을 수 있습니다.
    주어진 텍스트의 **발음 인식 오류와 오타**를 교정하여 더 자연스럽고 정확한 문장으로 수정해 주세요.
    문맥에 맞지 않는 단어가 있는 경우, **발음이 비슷하면서 문맥에 맞는 다른 단어**로 교체해 주세요.
    브랜드 이름이나 전문 용어는 잘못된 발음이 있으면 교정하고, 텍스트의 **원래 의미는 최대한 유지**해 주세요.
    추가 설명이나 요약을 하지 말고, 단순히 잘못된 부분을 수정하는 데 집중해 주세요.

    예시)
    - '타나비 어려워따' → '탄압이 어려웠다'
    - '키드 디자인' → '캐릭터 디자인'
    - 연음으로 인해 의미 전달이 어려운 부분을 교정

    텍스트:
    {text}

    수정된 텍스트:
    """
    )

    # 질문 생성용 Prompt
    question_prompt = PromptTemplate(
        input_variables=["text"],
        template = """
    주어진 텍스트를 바탕으로 적절한 꼬리질문을 하나 생성해 주세요.
    질문은 주어진 텍스트와 관련된 심화 내용을 묻는 형태로 이는 면접상황에서 면접자의 자질을 파악하기 위한 과정입니다.
    지엽적인 질문은 피하고 질문자가 어떻게 문제를 해결했는지, 답변에 어폐가 없었는지에 집중해 질문해주세여.
    만약 직무와 관련된 구체적인 답변이 들어왔다면 이에 관한 질문을 해도 좋습니다.

    텍스트:
    {text}

    생성된 질문:
    """
    )

    # LLM Chains 설정
    correction_chain = LLMChain(llm=llm, prompt=correction_prompt)
    question_chain = LLMChain(llm=llm, prompt=question_prompt)

    # 텍스트 교정 실행
    corrected_text = correction_chain.run(text=transcription)

    # 교정된 텍스트 기반 질문 생성 실행
    generated_question = question_chain.run(text=corrected_text)
    result['generated_question'] = generated_question
    return corrected_text, generated_question

# 발음 정확도
def pronunciation_accuracy(text_x, text_o):
    okt = Okt()

    # 형태소 분석을 통해 각 텍스트의 낱말을 추출
    words_x = set(okt.morphs(text_x))
    words_o = set(okt.morphs(text_o))

    # text_o의 낱말 중 text_x에 일치하는 낱말의 개수 계산
    matched_words = words_x.intersection(words_o)
    accuracy_score = len(matched_words) / len(words_o)  # 일치율 계산

    # 결과 출력
    return accuracy_score

# 발음 피드백
def pronunciation_feedback(accuracy_score):
    if accuracy_score >= 0.8:
        result['pronunciation'] = "정확"
    elif accuracy_score >= 0.7:
        result['pronunciation'] = "준수3"
    elif accuracy_score >= 0.6:
        result['pronunciation'] = "주의 정확한 발음을 하려는 노력이 필요합니다.6"
    else:
        result['pronunciation'] = "위험 발음이 많이 부정확합니다. 또박또박 발음하는 연습을 하세요.8"

# 질문 생성 함수
def generate_question(selected_industry, job_index):
    jobs = job_mapping.get(selected_industry, [])
    if 0 <= job_index < len(jobs):
        job = jobs[job_index]
        questions = df[job].dropna().tolist()  
        return job, random.choice(questions)
    else:
        return None, "잘못된 번호 선택입니다."


# 오디오 파일 분석 함수
def analyze_audio(file_path, question):
    # 오디오 전처리
    y, audio_data, sr, total_audio_time = preprocess_audio(file_path)

    # 16000으로 지정한 오디오
    text, segments = transcribe('./preprocess_audio.wav')
    result['answer'] = text
    # 인식한 텍스트 반환
    print("답변:", text)
    print("\n분석 결과:")
    # 전체 SPM 평균 계산
    total_duration = sum(segment["end_time"] - segment["start_time"] for segment in segments)
    total_syllables = sum(count_syllables(segment["text"]) for segment in segments)
    overall_spm = (total_syllables / total_duration) * 60
    overall_speed_rating = evaluate_speed(overall_spm)

    # 전체 RMS 평균 계산
    overall_rms = np.mean([analyze_sentence_volume(audio_data, segment["start_time"], segment["end_time"], sr) for segment in segments])

    # 각 문장에 대해 SPM 및 볼륨 변화 분석
    results = []
    spm_changes = []
    volume_changes = []

    # 연속적인 빠르기와 크기 변화를 추적하기 위한 변수
    speed_increase_start = []
    speed_increase_end = []
    speed_decrease_start = []
    speed_decrease_end = []
    volume_increase_start = []
    volume_increase_end = []
    volume_decrease_start = []
    volume_decrease_end = []

    for i, segment in enumerate(segments):
        sentence = segment["text"].strip()
        start = segment["start_time"]
        end = segment["end_time"]
        duration = end - start

        # 말의 빠르기 계산
        spm = calculate_sentence_spm(sentence, duration)
        speed_rating = evaluate_speed(spm)
        spm_changes.append(spm)

        # 목소리 크기 변화 분석
        rms_value = analyze_sentence_volume(audio_data, start, end, sr)
        volume_changes.append(rms_value)


        # 목소리 크기 변화 추적
        if i >= 2:  # 최소 세 번째 문장부터 연속적 변화 확인
            prev_rms = volume_changes[i-2:i+1]
            if all(prev_rms[j] < prev_rms[j + 1] - 0.01 for j in range(2)):  # 연속 0.01 이상 증가
                volume_increase_start.append(segments[i-2]["text"].strip())
                volume_increase_end.append(segments[i]["text"].strip())
            else:
                pass

            if all(prev_rms[j] > prev_rms[j + 1] + 0.01 for j in range(2)):  # 연속 0.01 이상 감소
                volume_decrease_start.append(segments[i-2]["text"].strip())
                volume_decrease_end.append(segments[i]["text"].strip())
            else:
                pass
        # 목소리 크기 변화 추적
        if i >= 4:  # 최소 세 번째 문장부터 연속적 변화 확인
            prev_rms = volume_changes[i-4:i+1]
            if all(prev_rms[j] < prev_rms[j + 1] - 0.01 for j in range(2)):  # 연속 0.01 이상 증가
                volume_increase_start.appned(segments[i-4]["text"].strip())
                volume_increase_end.append(segments[i+1]["text"].strip())
            else:
                pass

            if all(prev_rms[j] > prev_rms[j + 1] + 0.01 for j in range(2)):  # 연속 0.01 이상 감소
                volume_decrease_start.append(segments[i-4]["text"].strip())
                volume_decrease_end.append(segments[i+1]["text"].strip())
            else:
                pass
        results.append({
            "sentence": sentence,
            "duration": duration,
            "spm": spm,
            "speed_rating": speed_rating,
            "rms_value": rms_value
        })

    # 속도 일정/비일정 여부 판단 함수
    def check_consistency_spm(values, threshold=100): # 기존 91.16
        return "일정하게 유지되고 있으며" if max(values) - min(values) < threshold else "일정하지 않으며"

    def check_consistency_rms(values, threshold=0.05):
        result['size'] = "일정" if max(values) - min(values) < threshold else "위험"

    # 속도에 대한 전반적인 평가
    speed_consistency = check_consistency_spm(spm_changes)
    print(f"말의 속도는 {speed_consistency} {speed_rating}")

    # 빠르기와 크기 변화 출력
    if speed_increase_start is not None and speed_consistency=="일정하지 않으며" :
        for i in range(len(speed_increase_start)):
            start_text = speed_increase_start[i]
            end_text = speed_increase_end[i]
            result['speed'] += f"특히 '{start_text}' ~ '{end_text}' 구간에서 점점 빨라지고 있습니다.3"

    elif speed_decrease_start is not None and speed_consistency=="일정하지 않으며" :
        for i in range(len(speed_decrease_start)):
            start_text = speed_decrease_start[i]
            end_text = speed_decrease_end[i]
            result['speed'] += f"특히 '{start_text}' ~ '{end_text}' 구간에서 점점 느려지고 있습니다.3"

    # 목소리 크기에 대한 전반적인 평가
    volume_consistency = check_consistency_rms(volume_changes)
    print(f"전반적으로 목소리의 크기는 {volume_consistency}")

    # 목소리 크기 변화 출력
    if volume_increase_start is not None and volume_consistency=="위험" :
        for i in range(len(volume_increase_start)):
            start_text = volume_increase_start[i]
            end_text = volume_increase_end[i]
            result['size'] +=f"특히 '{start_text}'부터 '{end_text}'까지 목소리의 크기가 점점 커지고 있습니다.3"

    elif volume_decrease_start is not None and volume_consistency=="위험" :
        for i in range(len(volume_decrease_start)):
            start_text = volume_decrease_start[i]
            end_text = volume_decrease_end[i]
            result['speed'] += f"특히 '{start_text}'부터 '{end_text}'까지 목소리의 크기가 점점 작아지고 있습니다.3"
    
    # 발음 피드백
    corrected_text, generated_question = process_text_with_langchain(text)
    accuracy_score = pronunciation_accuracy(text, corrected_text)
    feedback = pronunciation_feedback(accuracy_score)

    # 침묵 피드백
    total_audio_time, long_silences, silence_percentage = find_silence(y, sr)
    long_silence_exists = len(long_silences) > 0
    silence_feedback_result = silence_feedback(total_audio_time, total_audio_time, silence_percentage, long_silence_exists)

    # 간투어 피드백
    filler_count = detect_filler_words(text)
    feedback = noise_feedback(filler_count)

    # 유사도 피드백 - 이 부분은 텍스트 피드백
    sil_feedback = similarity_feedback(question, corrected_text)


    return result

import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer, AutoFeatureExtractor, HubertForSequenceClassification, AutoConfig
from transformers.optimization import AdamW, get_constant_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader

class MyLitModel(pl.LightningModule):
    def __init__(self, num_labels, n_layers=1, projector=True, classifier=True, dropout=0.2, lr_decay=1, fold_idx=0):
        super(MyLitModel, self).__init__()
        self.audio_model = AutoModel.from_pretrained('team-lucid/hubert-base-korean')
        
        # Transformer 추가
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.audio_model.config.hidden_size,
                nhead=4,
                dim_feedforward=512,
                dropout=dropout,
                activation='gelu'
            ),
            num_layers=1
        )
        
        # 분류기 추가
        self.classifier = nn.Linear(self.audio_model.config.hidden_size, num_labels)
        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, audio_values, audio_attn_mask):
        # Hubert에서 특성 추출
        output = self.audio_model(input_values=audio_values, attention_mask=audio_attn_mask)
        hidden_states = output.last_hidden_state  # (batch_size, seq_len, hidden_size)
        # Transformer에 입력
        hidden_states = hidden_states.permute(1, 0, 2)  # Transformer 입력 형식: (seq_len, batch_size, hidden_size)
        transformer_out = self.transformer(hidden_states)  # (seq_len, batch_size, hidden_size)
        transformer_out = transformer_out.permute(1, 0, 2)  # 다시 원래 형식으로 변경: (batch_size, seq_len, hidden_size)

        # 마지막 타임스텝의 출력만 사용하거나 평균 풀링
        pooled_out = transformer_out.mean(dim=1)  # (batch_size, hidden_size)
        
        # 분류기 통과
        logits = self.classifier(self.dropout(pooled_out))  # (batch_size, num_labels)
        return logits

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        audio_values = batch['audio_values']
        audio_attn_mask = batch['audio_attn_mask']
        # 최종 예측값 
        logits = self(audio_values, audio_attn_mask)
        preds = torch.round(torch.sigmoid(logits.view(-1)))

        return preds

class MyDataset(Dataset):
    def __init__(self, audio, audio_feature_extractor, label=None):
        self.label = np.array(label if label is not None else [0] * len(audio)).astype(np.int64)
        self.audio = audio
        # 오디오 데이터를 모델이 이해할 수 있는 형태로 변환
        self.audio_feature_extractor = audio_feature_extractor

        # 하나의 오디오 파일만 주어질 경우 리스트로 변환
        if isinstance(self.audio, str):
            self.audio = [self.audio]
            self.label = [self.label]  # 하나의 레이블만 처리
    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        label = self.label[idx]
        audio = self.audio[idx]
        audio_feature = self.audio_feature_extractor(raw_speech=audio, return_tensors='np', sampling_rate=16000)
        audio_values, audio_attn_mask = audio_feature['input_values'][0], audio_feature['attention_mask'][0]

        item = {
            'label':label,
            'audio_values':audio_values,
            'audio_attn_mask':audio_attn_mask,
        }

        return item

from torch.nn.utils.rnn import pad_sequence

def collate_fn(samples):
    batch_labels = [sample['label'] for sample in samples]
    batch_audio_values = [torch.tensor(sample['audio_values']) for sample in samples]
    batch_audio_attn_masks = [torch.tensor(sample['audio_attn_mask']) for sample in samples]

    batch = {
        'label': torch.tensor(batch_labels),
        'audio_values': pad_sequence(batch_audio_values, batch_first=True),
        'audio_attn_mask': pad_sequence(batch_audio_attn_masks, batch_first=True),
    }

    return batch

def analyze_emotion(file_path):
    trainer = pl.Trainer(
    accelerator='cpu', 
    precision=16,
    )

    # 원하는 모델 경로를 지정 (하나의 모델만 사용)
    pretrained_model_path = './emotion_model.ckpt'  # 예시 모델 경로

    # 모델 로드
    pretrained_model = MyLitModel.load_from_checkpoint(
        pretrained_model_path,
        num_labels=1,
    )
    pretrained_model.eval()
    audio_feature_extractor = AutoFeatureExtractor.from_pretrained('team-lucid/hubert-base-korean')
    audio_feature_extractor.return_attention_mask=True
    test_audios,_=librosa.load(file_path, sr=16000)
    test_ds = MyDataset([test_audios], audio_feature_extractor)
    # 데이터셋 로드해 배치 단위로 처리
    test_dl = DataLoader(test_ds, batch_size=16, collate_fn=collate_fn)
    # 예측 수행
    final_pred = trainer.predict(pretrained_model, test_dl)
    if final_pred[0].item()==1:
        result['emotion']='5' #fear
    else:
        result['emotion']='0' #not fear



########## 문장 평가 부분
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification

def generate_feedback(expression_score, organization_score, content_score):
    feedback = []

    # 표현 피드백 - 17, 13, 8, 5
    if expression_score >= 2.6:
        feedback.append("답변의 표현: 표현력이 매우 뛰어나고 다양한 어휘와 적절한 표현을 사용하여 내용을 잘 전달하고 있습니다.0")
    elif expression_score >= 2.3:
        feedback.append("답변의 표현: 표현이 대체로 적절하며 자연스러운 표현을 사용하고 있습니다. 조금 더 다양한 단어를 사용하면 문장의 표현력이 더욱 돋보일 것입니다.4")
    elif expression_score >= 2.0:
        feedback.append("답변의 표현: 표현이 적절한 편이나, 일부 문장에서 어휘 선택과 문장 구조에 개선이 필요합니다.7")
    else:
        feedback.append("답변의 표현: 표현력이 부족하며, 다양한 어휘와 표현 사용을 통해 전달력을 높이는 것이 필요합니다.9")

    # 구성 피드백 - 17, 13, 10, 8
    if organization_score >= 2.7:
        feedback.append("답변의 구성: 구성이 매우 체계적이고 읽기 쉽게 잘 작성되었습니다.0")
    elif organization_score >= 2.4:
        feedback.append("답변의 구성: 구성이 체계적이고 읽기 쉽게 작성되었습니다. 문단 간 연결이 매끄럽습니다.4")
    elif organization_score >= 2.0:
        feedback.append("답변의 구성: 구성은 적절하지만 문장의 전개나 문단 연결에 약간의 개선이 필요합니다.7")
    else:
        feedback.append("답변의 구성: 구성이 약하고 글의 흐름이 자연스럽지 않습니다. 글의 구조와 연결성을 강화해 보세요.9")

    # 내용 피드백 - 17, 13, 10, 8
    if content_score >= 2.2:
        feedback.append("답변의 내용: 내용이 충실하며 주제에 대한 이해도가 높음을 보여줍니다. 구체적인 사례와 깊이 있는 설명이 돋보입니다.0")
    elif content_score >= 2.0:
        feedback.append("답변의 내용: 내용이 적절하며 주제에 대한 이해도가 보입니다. 그러나 세부적인 구체성이 약간 부족할 수 있습니다.4")
    elif content_score >= 1.8:
        feedback.append("답변의 내용: 내용은 적절하지만 핵심 아이디어나 구체적인 예시가 부족한 부분이 있습니다. 조금 더 깊이 있는 답변이 필요합니다.7")
    else:
        feedback.append("답변의 내용: 내용이 부족하며, 추가적인 설명과 구체적인 예시가 필요합니다. 더 깊이 있는 답변이 요구됩니다.9")

    return feedback

# 예측 생성 및 피드백 함수
def predict_feedback(paragraph):
    # CSV 파일 경로
    file_path = './nlp_final.csv'
    df = pd.read_csv(file_path)

    # GPU 장치 설정
    device = torch.device("mps" if torch.cuda.is_available() else "cpu")

    # 토크나이저 및 모델 설정
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    model_save_path = './saved_model'
    model = BertForSequenceClassification.from_pretrained(model_save_path)
    model.to(device)  # 모델을 GPU로 이동
    
    # 토큰화 및 텐서 변환
    inputs = tokenizer(paragraph, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {key: val.to(device) for key, val in inputs.items()}  # 입력 텐서를 GPU로 이동

    # 모델 예측
    model.eval()  # 평가 모드로 전환
    with torch.no_grad():
        outputs = model(**inputs)
    scores = outputs.logits[0].cpu().numpy()  # GPU에서 CPU로 이동 후 Numpy 배열로 변환

    # 피드백 생성
    feedback = generate_feedback(scores[0], scores[1], scores[2])
    return scores, feedback



def analyze_sentence(sentence):

    # 예측 및 피드백 생성
    scores, feedback = predict_feedback(sentence)

    # 결과 출력
    Score = {
        'expression' : f"{scores[0]:.2f}",
        'structure' : f"{scores[1]:.2f}",
        'content' : f"{scores[2]:.2f}",
        'exp_feed' : feedback[0],
        'str_feed' : feedback[1],
        'con_feed' : feedback[2]
    }
    return Score