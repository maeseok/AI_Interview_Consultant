import React, { useState } from 'react';
import Navbar from './Navbar';
import JobSelection from './JobSelection';
import QuestionUpload from './QuestionUpload';
import FeedbackResult from './FeedbackResult';
import LoadingSpinner from "./LoadingSpinner";
import DetailedEvaluationModal from "./DetailedEvaluationModal";

function App() {
  const [page, setPage] = useState("jobSelection");
  const [selectedJob, setSelectedJob] = useState(null);
  const [question, setQuestion] = useState(''); // 서버에서 받은 질문
  const [feedback, setFeedback] = useState(null);
  const [tailQuestionData, setTailQuestionData] = useState(null);
  const [loading, setLoading] = useState(false); // 로딩 상태
  const [error, setError] = useState(null); // 에러 상태
  const [isModalOpen, setIsModalOpen] = useState(false);

  // 서버에서 질문 가져오기
  const fetchQuestion = async (job) => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`/main`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ job: job }),
      });
      if (!response.ok) {
        throw new Error("질문을 불러오지 못했습니다.");
      }
      const data = await response.json();
      setQuestion(data.question); // 질문을 상태에 저장
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

  // const handleFileUpload = (feedbackData) => {
  //   setFeedback(feedbackData);
  //   setPage("feedbackResult");
  // };
  const handleFileUpload = async (feedbackData) => {
    setLoading(true); // 로딩 시작
    try {
      // 실제 서버 호출 없이 더미 데이터 설정하기
      setTimeout(() => {
        setFeedback(feedbackData);
        setPage("feedbackResult");
        setLoading(false);
      }, 50); // 3초 후에 피드백 페이지로 넘어가기
    } catch (error) {
      console.error("Error uploading file:", error);
      alert("파일 업로드 중 오류가 발생했습니다.");
    }
  };

  const handleNavigateHome = () => {
    setPage("jobSelection");
    setSelectedJob(null);
    setFeedback(null);
  };

  const handleNavigateTailQuestion = () => {
    setPage("tailQuestion");
  };

  const handleNavigateDetailedEvaluation = () => {
    setIsModalOpen(true);
  };

  const closeModal = () => {
    setIsModalOpen(false);
  };


  const handleTailQuestion = () => {
    setTailQuestionData(feedback.feedback?.generated_question);
    setQuestion(feedback.feedback?.generated_question)
    setPage("tailQuestion");
  };

  const handleEndQuestions = () => {
    alert("모든 질문을 종료합니다. 감사합니다!");
    setPage("jobSelection");
    setSelectedJob(null);
    setFeedback(null);
    setTailQuestionData(null);
  };

  const API_BASE_URL = process.env.REACT_APP_API_URL || "http://127.0.0.1:5000";
  
  return (
    
    <div className="App">
      {isModalOpen && <DetailedEvaluationModal onClose={closeModal} />}
      {!isModalOpen && <Navbar />}
      {loading && page !== "jobSelection" && <LoadingSpinner />}
      {!loading && page === "jobSelection" && (
        <JobSelection 
          onJobSelect={handleJobSelect} 
          onQuestionUpdate={handleQuestionUpdate} 
        />
      )}
      {!loading && page === "questionUpload" && (
        <QuestionUpload
          selectedJob={selectedJob} 
          Question={question} 
          onFileUpload={handleFileUpload}
          onPageChange={setPage} // 페이지 변경 함수 전달
          apiUrl={API_BASE_URL} // 전달 
          setLoading={setLoading} // pass setLoading as a prop
        />
      )}
      {!loading && page === "feedbackResult" && feedback && (
        <FeedbackResult
          feedback={feedback}
          Question={question}
          onTailQuestion={handleTailQuestion}
          onDetailedEvaluation={handleNavigateDetailedEvaluation}
          onEndQuestions={handleEndQuestions}
        />
        
      )}
      {!loading && page === "tailQuestion" && tailQuestionData && (
        <QuestionUpload
          selectedJob={selectedJob}
          onFileUpload={handleFileUpload}
          questionType="꼬리질문"
          onPageChange={setPage} // 페이지 변경 함수 전달
          Question={tailQuestionData}
          apiUrl={API_BASE_URL} // 전달 
          setLoading={setLoading} // pass setLoading as a prop
        />
      )}
    </div>
  );
}


//   return (
//     <div className="App">
//       <Navbar onNavigateHome={handleNavigateHome} />
//       {page === "jobSelection" && (
//         <JobSelection 
//           onJobSelect={handleJobSelect} 
//           onQuestionUpdate={handleQuestionUpdate} 
//         />
//       )}
//       {page === "questionUpload" && (
//         <QuestionUpload 
//           selectedJob={selectedJob} 
//           question={question} 
//           onFileUpload={handleFileUpload}
//           onPageChange={setPage} // 페이지 변경 함수 전달
//           apiUrl={API_BASE_URL} // 전달 
//         />
//       )}
//       {page === "feedbackResult" && (
//         <FeedbackResult
//           Question={question}
//           feedback={feedback}
//         />
//       )}
//     </div>
//   );
// }

export default App;
