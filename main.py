__import__("pysqlite3")
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
import streamlit as st
import tempfile
import os
#from dotenv import load_dotenv

# 환경 변수 로드
#load_dotenv()

# 제목
st.title("ChatAML")
st.write("---")

# 파일 업로드
uploaded_file = st.file_uploader("Choose a file", type=["pdf"])
st.write("---")

def pdf_to_document(uploaded_file):
    # 임시 디렉토리 생성
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    
    # 파일을 임시 디렉토리에 저장
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    
    # PyPDFLoader로 PDF 파일 로드 및 분할
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    
    # 임시 디렉토리 정리
    temp_dir.cleanup()
    
    return pages

# 파일이 업로드되었을 경우 처리하는 로직
if uploaded_file is not None:
    # 업로드된 PDF 파일을 문서로 변환
    pages = pdf_to_document(uploaded_file)
    
    # 텍스트를 일정한 크기로 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=20,
        length_function=len
    )
    texts = text_splitter.split_documents(pages)

    # Embedding 모델 초기화
    from langchain.embeddings import OpenAIEmbeddings
    embeddings_model = OpenAIEmbeddings()

    # Chroma에 문서 임베딩을 저장
    db = Chroma.from_documents(texts, embeddings_model)

    # 질문 입력 섹션
    st.header("AML PDF에게 질문해보세요!")
    question = st.text_input('질문을 입력하세요')

    if st.button("질문하기"):

        from langchain.chat_models import ChatOpenAI
        from langchain.chains import RetrievalQA

        with st.spinner('잠시만 기다려 주세요...'):
            
            # LLM 및 질의응답 체인 초기화
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
            qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())

        # 질문에 대한 답변 생성
        result = qa_chain.run({"query": question})
        st.write(result)
