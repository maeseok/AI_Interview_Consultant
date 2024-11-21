import React, { useState } from 'react';
import './JobSelection.css';

function JobSelection({ onJobSelect, onQuestionUpdate }) {
  const [selectedJob, setSelectedJob] = useState(null);
  const jobs = ["공통질문", "경영관리", "영업마케팅", "공공서비스", "ICT", "연구개발", "디자인", "생산제조", "의료"];
  
  // 직군 선택 핸들러
  const handleJobSelect = (job) => {
    setSelectedJob(job); // 선택된 직군 값 업데이트
    onJobSelect(job);    // 부모 컴포넌트에 직군 선택 값 전달

    // 직군을 Flask API로 전송
    fetch('http://127.0.0.1:5000/main', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ job: job }),  // 선택된 직군을 서버로 전송
    })
    .then(response => response.json())
    .then(data => {
      onQuestionUpdate(data.question); // 서버에서 받은 질문을 상태로 설정
    })
    .catch(error => {
      console.error('There was a problem with the fetch operation:', error);
      onQuestionUpdate('서버와 연결할 수 없습니다.'); // 오류 발생 시 처리
    });
  };

  return (
    <div className="job-selection">
      <h1>직군을 선택하세요</h1>
      <ul>
        {jobs.map((job, index) => (
          <li key={index} onClick={() =>  handleJobSelect(job)}>
            {job}
          </li>
        ))}
      </ul>
    </div>
  );
}

export default JobSelection;
