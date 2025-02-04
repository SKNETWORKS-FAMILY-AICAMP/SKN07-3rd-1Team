import streamlit as st
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.chat_models import ChatOpenAI

chat = ChatOpenAI(
    model = 'gpt-4o-2024-08-06'
)

def generate_res(text):
    messages = [
        SystemMessage(content='너는 친절하고 도움이 되는 여행 가이드야. 사용자의 질문에 세심하게 답변해줘.'),
        HumanMessage(content=text)
    ]
    return chat(messages).content

st.set_page_config(page_title="상담 챗봇", layout="wide", initial_sidebar_state="collapsed")
st.title("✈️여행칭구🤖")

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
    st.session_state['messages'] = []

with st.form("질문하세요"):
    text = st.text_area("질문 입력:", '')
    submitted = st.form_submit_button("보내기")

if submitted and text:
    response = generate_res(text)
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
