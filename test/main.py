import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import librosa
import re
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Audio
import soundfile as sf
import noisereduce as nr
import whisper_timestamped as whisper_t
from konlpy.tag import Okt
import warnings
import random
from sentence_transformers import SentenceTransformer, util
from feedback import *

# 특정 경고 무시
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

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


# 전체 흐름 관리 함수
def main():
    # 1. 질문 생성
    print("어떤 항목의 면접 질문을 찾으시겠습니까? (공통 질문 포함):")
    for key, industry in industry_map.items():
        print(f"{key}. {industry}")
    
    selected_industry_key = input("선택한 번호: ")
    
    if selected_industry_key == "0":
        selected_job = "공통질문"
        questions = df[selected_job].dropna().tolist()
        question = random.choice(questions)
    else:
        selected_industry = industry_map[selected_industry_key]
        print(f"선택한 항목: {selected_industry}")

        jobs = job_mapping[selected_industry]
        print("선택할 수 있는 항목:")
        for idx, job in enumerate(jobs):
            print(f"{idx + 1}. {job}")

        job_index = int(input("항목 번호를 입력하세요: ")) - 1
        selected_job, question = generate_question(selected_industry, job_index)

    # 질문 출력
    print(f"\n질문: {question}")

    # 2. 오디오 파일 입력
    file_path = input("응답한 오디오 파일 경로를 입력하세요 (예: response.m4a): ")
    # 3. 오디오 분석
    analyze_audio(file_path, question)
    
    # 되긴하는데 시간 너무 오래걸림
    #analyze_emotion(file_path)
main()