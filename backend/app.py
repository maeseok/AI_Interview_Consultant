from flask import Flask, request, render_template, redirect, url_for, request, jsonify
import random
import os
import warnings
import random
from feedback import *
from flask_cors import CORS

# 특정 경고 무시
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

file_path = './직무별_면접_질문_리스트_한글변환.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1')
# 산업과 직무 매핑 (생략된 부분은 그대로 유지)
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

app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



# 메인 페이지
@app.route("/main", methods=["POST"])
def main():
    # React에서 보낸 직군 데이터 받기
    data = request.get_json()
    selected_job = data.get("job")
    print(selected_job)
    if selected_job == "공통질문":
        question = random.choice(df[selected_job].dropna().tolist())
    else:
        subcategory = data.get("subcategory")
        question = random.choice(df[subcategory].dropna().tolist())
    #question= "학생들이 쉽게 이해할 수 있도록 프로그래밍 개념을 설명하는 방법은 무엇인가요?"
    #print(question)
    return jsonify({"question": question})  # 질문을 JSON으로 반환

# 파일 업로드 페이지
@app.route("/upload", methods=["POST"])
def upload():
    if 'audio_file' not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files['audio_file']
    question = request.form.get('question')

    if not file or not question:
        return jsonify({"error": "Missing file or question."}), 400
    
    # 파일 저장 처리 (예: local 디렉토리)
    file_path = f"./uploads/{file.filename}"
    file.save(file_path)
    Feedback = analyze_audio(file_path, question)

    Feedback['emotion'] = analyze_emotion(file_path) # 감정 분석
    Feedback['analysis'] = analyze_sentence(Feedback['answer']) # 문장 평가
    
    print(Feedback) # 최종 피드백 결과

    # 처리 후 결과 반환 (예: 업로드 성공 메시지)
    return jsonify({"message": "File uploaded successfully!", 
                    "redirect_url": "/result",
                    "feedback":Feedback})


if __name__ == "__main__":
    app.run(debug=True)
