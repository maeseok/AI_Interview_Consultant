import React, { useState } from 'react';
import './QuestionUpload.css';

function QuestionUpload({ selectedJob, question, onFileUpload, apiUrl, onPageChange}) {
  const [file, setFile] = useState(null);
  const [page, setPage] = useState("jobSelection");

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    setFile(selectedFile);
  };

  const handleUpload = async () => {
    //if (file) {
    //  onFileUpload(file);
    //}
    if (!file) return;

    const formData = new FormData();
    formData.append('audio_file', file);
    formData.append('question', question);

    try {
      const response = await fetch(`${apiUrl}/upload`, {
        method: 'POST',
        body: formData,
      });
      console.log(apiUrl);
      if (response.ok) {
        const result = await response.json();
        alert(`업로드 성공! 결과 페이지: ${result.redirect_url}`);
        onFileUpload(result); // 피드백 데이터를 상위로 전달
        onPageChange("feedbackResult"); // 페이지 변경
        // window.location.href = result.redirect_url; // 결과 페이지로 리다이렉트
      } else {
        alert('업로드 실패. 다시 시도해주세요.');
      }
    } catch (error) {
      console.error('Error uploading file:', error);
      alert('파일 업로드 중 문제가 발생했습니다.');
    }
  };

  return (
    <div className="question-upload">
      <h2>{selectedJob} 직군 질문</h2>
      <p>질문:  "{question || '질문이 없습니다.'}"</p>
      <label className="file-upload-label" htmlFor="file-upload">
        <br></br>
        {file ?  "🔊 답변 파일이 업로드 되었습니다 : "+file.name : "🔊 답변 파일을 업로드 해주세요."}
      </label>
      <input 
        id="file-upload" 
        type="file" 
        onChange={handleFileChange} 
      />
      <button onClick={handleUpload} disabled={!file}>
        업로드
      </button>
    </div>
  );
}

export default QuestionUpload;
