import streamlit as st
import os
import json
import openai
import pandas as pd
from langchain.schema import Document, messages_from_dict
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema import HumanMessage, AIMessage, SystemMessage


## ---------- 데이터베이스 초기화 및 체크
db_path = './accomodation_data/'
db_initialized = os.path.exists(db_path)

if not db_initialized:
    # CSV 파일 읽기
    df = pd.read_csv('suksoDF.csv')

    # 텍스트 벡터화: OpenAI 임베딩 모델 사용
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002"
    )

    db = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings
    )

    # 벡터화 된 데이터를 DB에 저장
    documents = [Document(page_content=f"관광지명: {row['name']} 주소: {row['address']}, 소개: {row['overview']}, 숙소정보: {row['generalInfo']}, 객실정보: {row['roomInfo']}", metadata={"id": idx}) for idx, row in df.iterrows()]

    # 문서 배치를 나누는 함수
    def batch_documents(documents, batch_size):
        for i in range(0, len(documents), batch_size):
            yield documents[i:i + batch_size]

    # 배치 크기 설정
    batch_size = 100

    # 배치로 나누어 처리
    for batch in batch_documents(documents, batch_size):
        try:
            db.add_documents(batch)
            db.persist()
            print(f"Batch of size {batch_size} added successfully.")
        except Exception as e:
            print(f"An error occurred: {e}")
else:
    # 데이터베이스 로드
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002"
    )

    db = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings
    )

    # 데이터베이스가 올바르게 로드되었는지 확인
    if db is None:
        raise ValueError("데이터베이스 로드에 실패했습니다.")
    else:
        print("데이터베이스가 성공적으로 로드되었습니다.")

llm = ChatOpenAI(
    model='gpt-4o-2024-08-06'
)

# 5. RetrievalQA 객체를 사용하여 검색 및 답변 생성
qa_chain = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="map_reduce", 
    retriever=db.as_retriever()
)


## ---------- 프롬프트 설계

def get_answer_from_db(user_query, chat_history):
    results = db.similarity_search(user_query, k=5)

    if not results:
        return "데이터베이스에서 유사한 결과를 찾을 수 없습니다."

    context = "\n".join([result.page_content for result in results])

    messages = chat_history.copy()
    messages.append(HumanMessage(content=f"사용자 질문: {user_query}\n\n참고할 정보:\n{context}\n\n이 정보를 기반으로 정확하게 답변해 주세요.  {user_query}의 요구사항과 {context}\n\n의 정보가 일치하지 않으면 제외하고 답변해 주세요."))

    response = llm(messages)
    print(f"Response from ChatGPT: {response.content}")
    return response.content

# ---------- Streamlit 구현

st.set_page_config(page_title="상담 챗봇", layout="wide", initial_sidebar_state="collapsed")
st.title("🏨숙소봇🤖")

st.markdown(
    """
    <style>
    .chat-message {
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        display: inline-block;
    }
    .human-message {
        background-color: #F0F8FF; 
        text-align: left;
        color: black;
    }
    .ai-message {
        background-color: #E6E6FA; 
        text-align: left;
        color: black;
    }
    .stButton > button {
        background-color: yellow; 
        border: none;
        color: black;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 4px;
    }
    .stButton > button:hover {
        background-color: #ffcc00; 
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 대화 기록 저장
if 'messages' not in st.session_state:
    st.session_state['messages'] = [SystemMessage(content="너는 친절하고 사용자 요구사항에 맞는 숙소를 추천해주는 가이드야. 사용자의 질문에 세심하게 답변해줘.")]

with st.form("질문하세요"):
    text = st.text_area("질문 입력:", '')
    submitted = st.form_submit_button("보내기")

if submitted and text:
    chat_history = st.session_state['messages']
    response = get_answer_from_db(text, chat_history)
    st.session_state['messages'].append(HumanMessage(content=text))
    st.session_state['messages'].append(AIMessage(content=response))

# 대화 기록 출력
for message in st.session_state['messages']:
    if isinstance(message, HumanMessage):
        st.markdown(f"<div class='chat-message human-message'><strong>👩🏻:</strong> {message.content}</div>", unsafe_allow_html=True)
    elif isinstance(message, AIMessage):
        st.markdown(f"<div class='chat-message ai-message'><strong>🤖:</strong> {message.content}</div>", unsafe_allow_html=True)

if submitted and not text:
    st.warning("질문을 입력하세요.")
