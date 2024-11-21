import React, { useState } from 'react';
import Navbar from './Navbar';
import JobSelection from './JobSelection';
import QuestionUpload from './QuestionUpload';
import FeedbackResult from './FeedbackResult';

function App() {
  const [page, setPage] = useState("jobSelection");
  const [selectedJob, setSelectedJob] = useState(null);
  const [question, setQuestion] = useState(''); // 서버에서 받은 질문
  const [feedback, setFeedback] = useState(null);
  const [loading, setLoading] = useState(false); // 로딩 상태
  const [error, setError] = useState(null); // 에러 상태

  // 서버에서 질문 가져오기
  const fetchQuestion = async (job) => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`/get-question?job=${job}`);
      if (!response.ok) {
        throw new Error("질문을 불러오지 못했습니다.");
      }
      const data = await response.json();
      setQuestion(data.question);
    } catch (err) {
      console.error(err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleJobSelect = async (job) => {
    setSelectedJob(job);
    setPage("questionUpload");
    await fetchQuestion(job); // 직군 선택 후 질문 불러오기
  };

  const handleQuestionUpdate = (newQuestion) => {
    setQuestion(newQuestion); // 서버에서 받은 질문 저장
  };

  const handleFileUpload = (feedbackData) => {
    setFeedback(feedbackData);
    setPage("feedbackResult");
  };

  const handleNavigateHome = () => {
    setPage("jobSelection");
    setSelectedJob(null);
    setFeedback(null);
  };
  const API_BASE_URL = process.env.REACT_APP_API_URL || "http://127.0.0.1:5000";
  
  return (
    <div className="App">
      <Navbar onNavigateHome={handleNavigateHome} />
      {page === "jobSelection" && (
        <JobSelection 
          onJobSelect={handleJobSelect} 
          onQuestionUpdate={handleQuestionUpdate} 
        />
      )}
      {page === "questionUpload" && (
        <QuestionUpload 
          selectedJob={selectedJob} 
          question={question} 
          onFileUpload={handleFileUpload}
          onPageChange={setPage} // 페이지 변경 함수 전달
          apiUrl={API_BASE_URL} // 전달 
        />
      )}
      {page === "feedbackResult" && (
        <FeedbackResult
          Question={question}
          feedback={feedback}
        />
      )}
    </div>
  );
}

export default App;
