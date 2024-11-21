import React from 'react';
import './FeedbackResult.css';

function FeedbackResult({ feedback: feedbackData, Question }) {
  // feedback에서 필요한 값을 구조 분해 할당
  console.log(feedbackData)
  console.log(Object.keys(feedbackData.feedback)); // 확인용
  console.log(feedbackData.feedback?.size)
  //const feedbackData = JSON.parse(response.feedbackData);
  // 구조 분해 할당
  const {
    answer,
    speed,
    size,
    generated_question,
    pronunciation,
    break_time,
    meaningless,
    similarity,
    emotion,
    analysis // 이 부분 구체화 해결 필요
  } = feedbackData.feedback;
  // 변수들을 배열로 변환
  const variables = [speed, size, pronunciation, similarity, meaningless, break_time];

  // 조건을 처리하고 숫자를 더하기
  const total = variables.reduce((sum, value) => {
    if (value && value.length >= 3) {
      const lastCharAsNumber = Number(value.slice(-1)); // 마지막 문자를 숫자로 변환
      return sum + (isNaN(lastCharAsNumber) ? 0 : lastCharAsNumber); // 숫자인 경우 더하기
    }
    return sum; // 조건 미충족 시 0 추가
  }, 0);

  const score  =100-(total+Number(analysis)+Number(emotion))// 점수 - 미완 

  return (
    <div className="feedbackData-result">
      <div className="feedbackData-container">
        <div className="top-section">
          <div className="overall-score">
            <h3>전체 피드백 점수</h3>
            <div className="score-chart">
              <div className="circle">
                <span className="percentage">{score || 'N/A'}</span>
              </div>
            </div>
          </div>
          <div className="emotion-status">
            <div className="status-header">
              <h3> {emotion === '0' ? '😊안정' :'😳불안' }</h3>
            </div>
            <div className="status-details">
              <div>
                <span>말의 속도</span> <strong className='speed'>{speed ? speed.slice(0, 2) : 'N/A'}</strong>
              </div>
              <div>
                <span>말의 크기</span> <strong className='size'>{size ? size.slice(0, 2) : 'N/A'}</strong>
              </div>
              <div>
                <span>발음</span> <strong className='pronunciation'>{pronunciation ? pronunciation.slice(0,2) : 'N/A'}</strong>
              </div>
              <div>
                <span>유사도</span> <strong className='similarity'>{similarity ? similarity.slice(0,2) : 'N/A'}</strong>
              </div>
              <div>
                <span>간투어</span> <strong className='meaningless'>{meaningless ? meaningless.slice(0,2) : 'N/A'}</strong>
              </div>
              <div>
                <span>휴지</span> <strong className='break'>{break_time ? break_time.slice(0,2) : 'N/A'}</strong>
              </div>
            </div>
          </div>
        </div>

        <div className="detailed-analysis">
          <table>
            <thead>
              <tr>
                <th>표현</th>
                <th>구성</th>
                <th>내용</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>{analysis?.expression || '2.23'}</td>
                <td>{analysis?.structure || '1.95'}</td>
                <td>{analysis?.content || '2.28'}</td>
              </tr>
            </tbody>
          </table>
        </div>

        <div className="ai-question">
          <h4>질문</h4>
          <p className="question">{Question || '질문이 없습니다.'}</p>
          <h4>사용자 답변</h4>
          <p className="answer">{answer || '답변이 없습니다.'}</p>
          <h4>주요 피드백</h4>
          <div className="main-feedbackData">
            <p>{speed && speed.length >= 3 ? speed.slice(3,-1) : null}</p>{/* 개선 조언 출력 */}
            <p>{size && size.length >= 3 ? size.slice(3,-1) : null}</p>
            <p>{break_time && break_time.length >= 3 ? break_time.slice(3,-1) : null}</p>
            <p>{meaningless && meaningless.length >= 3 ? meaningless.slice(3,-1) : null}</p>
            <p>{similarity && similarity.length >= 3 ? similarity.slice(3,-1) : null}</p>
            <p>{pronunciation && pronunciation.length >= 3 ? pronunciation.slice(3,-1) : null}</p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default FeedbackResult;