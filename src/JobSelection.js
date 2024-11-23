import React, { useState } from 'react';
import './JobSelection.css';

function JobSelection({ onJobSelect, onQuestionUpdate }) {
  const jobCategories = {
    "공통질문": [],
    "경영관리": ["경영기획", "웹기획PM", "DBA", "ERP"],
    "영업마케팅": ["마케팅", "영업고객상담", "웹마케팅", "서비스"],
    "공공서비스": ["건설", "관광레저", "미디어문화"],
    "연구개발": ["연구개발", "연구", "빅데이터AI", "소프트웨어", "하드웨어"],
    "ICT": [
      "UI",
      "IT강사",
      "QA",
      "게임개발자",
      "네트워크서버보안",
      "시스템프로그래머",
      "웹디자인",
      "웹프로그래머",
      "응용프로그래머",
      "웹운영자",
    ],
    "디자인": ["디자인", "영상제작편집"],
    "생산제조": ["무역유통"],
    "의료": ["의료질문"],
  };

  const [selectedCategory, setSelectedCategory] = useState(null);
  const [subcategories, setSubcategories] = useState([]);
  const [selectedSubcategory, setSelectedSubcategory] = useState(null); // 선택한 서브카테고리

  const handleCategorySelect = (category) => {
    setSelectedCategory(category);
    if (category === "공통질문") {
      onJobSelect(category);
      fetch('http://127.0.0.1:5000/main', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          job: category, // 상위 카테고리 (직군)
          subcategory: null, // Send the selected subcategory
        }),
      })
      .then((response) => response.json())
      .then((data) => {
        console.log('Server response:', data); // Log server response
        onQuestionUpdate(data.question); // Flask에서 반환된 질문 업데이트
      })
      .catch((error) => {
        console.error('Fetch 에러:', error);
        onQuestionUpdate('서버와 연결할 수 없습니다.');
      });
    }
    else {
      setSubcategories(jobCategories[category]);
    }
  };

  const handleSubcategorySelect = (subcategory) => {
    if (!selectedCategory) {
      console.error("Error: 상위 카테고리가 선택되지 않았습니다.");
      onQuestionUpdate("상위 카테고리를 먼저 선택하세요.");
      return;
    }

    setSelectedSubcategory(subcategory); // 서브카테고리 상태 업데이트

    fetch('http://127.0.0.1:5000/main', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        job: selectedCategory, // 상위 카테고리 (직군)
        subcategory: subcategory, // 선택한 서브카테고리
      }),
    })
      .then((response) => response.json())
      .then((data) => {
        console.log('Server response:', data); // Log server response
        onQuestionUpdate(data.question); // Flask에서 반환된 질문 업데이트
      })
      .catch((error) => {
        console.error('Fetch 에러:', error);
        onQuestionUpdate('서버와 연결할 수 없습니다.');
      });
  };
  // const [selectedJob, setSelectedJob] = useState(null);
  // const jobs = ["공통질문", "경영관리", "영업마케팅", "공공서비스", "ICT", "연구개발", "디자인", "생산제조", "의료"];
  
  // // 직군 선택 핸들러
  // const handleJobSelect = (job) => {
  //   setSelectedJob(job); // 선택된 직군 값 업데이트
  //   onJobSelect(job);    // 부모 컴포넌트에 직군 선택 값 전달

  //   // 직군을 Flask API로 전송
  //   fetch('http://127.0.0.1:5000/main', {
  //     method: 'POST',
  //     headers: {
  //       'Content-Type': 'application/json',
  //     },
  //     body: JSON.stringify({ job: job }),  // 선택된 직군을 서버로 전송
  //   })
  //   .then(response => response.json())
  //   .then(data => {
  //     onQuestionUpdate(data.question); // 서버에서 받은 질문을 상태로 설정
  //   })
  //   .catch(error => {
  //     console.error('There was a problem with the fetch operation:', error);
  //     onQuestionUpdate('서버와 연결할 수 없습니다.'); // 오류 발생 시 처리
  //   });
  // };
  return (
    <div className="job-selection">
      <h1>직군을 선택하세요</h1>

      {!selectedCategory && (
        <ul>
          {Object.keys(jobCategories).map((category, index) => (
            <li key={index} onClick={() => handleCategorySelect(category)}>
              {category}
            </li>
          ))}
        </ul>
      )}

      {selectedCategory && (
        <div style={{ textAlign: 'center' }}>
          <h2>{selectedCategory} 직군</h2>
          <ul>
            {subcategories.length > 0 ? (
              subcategories.map((subcategory, index) => (
              <li
                key={index}
                onClick={() => {
                  onJobSelect(subcategory); // 서브카테고리를 부모로 전달
                  fetch('http://127.0.0.1:5000/main', {
                    method: 'POST',
                    headers: {
                      'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                      job: selectedCategory, // Send the selected job category
                      subcategory: subcategory, // Send the selected subcategory
                    }),
                  })
                    .then((response) => response.json())
                    .then((data) => {
                      if (data.question) {
                        onQuestionUpdate(data.question); // 질문 업데이트
                      } else {
                        onQuestionUpdate('질문을 가져오는 데 실패했습니다.');
                      }
                    })
                    .catch((error) => {
                      console.error('Fetch 에러:', error);
                      onQuestionUpdate('서버와 연결할 수 없습니다.');
                    });
                }}
              >{subcategory}
                </li>
              ))
            ) : (
              <li onClick={() => onJobSelect(selectedCategory)}>
                {selectedCategory} 질문
              </li>
            )}
          </ul>
          <div style={{ display: 'flex', justifyContent: 'center', marginTop: '20px' }}>
          <button
              className="back-button"
              onClick={() => setSelectedCategory(null)}>
              뒤로가기
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
//   return (
//     <div className="job-selection">
//       <h1>직군을 선택하세요</h1>
//       <ul>
//         {jobs.map((job, index) => (
//           <li key={index} onClick={() =>  handleJobSelect(job)}>
//             {job}
//           </li>
//         ))}
//       </ul>
//     </div>
//   );
// }

export default JobSelection;
