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
    #질문: "학생들이 쉽게 이해할 수 있도록 프로그래밍 개념을 설명하는 방법은 무엇인가요?"
    
    # Feedback = {'answer': '프로그래밍 개념을 쉽게 설명하려면, 실생활 예시를 통해 비유하는 것이 중요합니다. 예를 들어, 변수는 상자처럼 생각하고, 반복문은 반복되는 일을 자동화하는 도구로 설명할 수 있습니다. 또한, 학생들이 직접 코딩을 해보며 경험을 쌓도록 유도하면, 개념을 더 쉽게 이해할 수 있습니다.', 
    #             'speed': '위험 말의 속도가 굉장히 빠릅니다. 차분한 전달이 필요합니다.\n특히 \'예를 들어, 변수는\' ~ \'도구로 설명할 수 있습니다.\' 구간에서 점점 빨라지고 있습니다.3', 
    #             'size': '일정', 
    #             'generated_question': '약속 장소를 옮기는 이유와 사람의 많고 적음이 어떤 영향을 미친다고 생각하시는지, 그리고 그 선택이 실제 상황에서 어떻게 도움이 되었는지 구체적으로 설명해 주실 수 있나요?', 
    #             'pronunciation': '정확', 
    #             'break_time': '주의 녹음 시간인 32.088 초 동안, 휴지 구간이 꽤 자주 발생했으며 긴 무음 구간이 발견되었습니다. 답변을 더 간결하게 만드는 연습이 필요합니다.5', 
    #             'meaningless': '없음', 
    #             'similarity': '높음'}
    #not fear
    Feedback['emotion']='0' # analyze_emotion(file_path)
    Feedback['analysis']=analyze_sentence(Feedback['answer'])
    # Feedback['analysis']={'expression': '2.63', 
    #                       'structure': '2.21', 
    #                       'content': '2.15', 
    #                       'exp_feed': '답변의 표현: 표현력이 매우 뛰어나고 다양한 어휘와 적절한 표현을 사용하여 내용을 잘 전달하고 있습니다.0', 
    #                       'str_feed': '답변의 구성: 구성은 적절하지만 문장의 전개나 문단 연결에 약간의 개선이 필요합니다.9', 
    #                       'con_feed': '답변의 내용: 내용이 적절하며 주제에 대한 이해도가 보입니다. 그러나 세부적인 구체성이 약간 부족할 수 있습니다.4'} 
    print(Feedback)

    # 처리 후 결과 반환 (예: 업로드 성공 메시지)
    return jsonify({"message": "File uploaded successfully!", 
                    "redirect_url": "/result",
                    "feedback":Feedback})


if __name__ == "__main__":
    app.run(debug=True)
