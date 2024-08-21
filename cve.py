# import os
import streamlit as st
import os
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.document_loaders import WebBaseLoader
from kor.documents.html import MarkdownifyHTMLProcessor
from langchain.text_splitter import RecursiveCharacterTextSplitter


#GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
from dotenv import load_dotenv
load_dotenv()

#API 및 Reddit 정보 설정
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

@st.cache_resource
def initialize_llm():
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=GOOGLE_API_KEY)

llm = initialize_llm()

prompt_template = """Please summarize the sentence according to the following REQUEST.
REQUEST:
1. Each summarized sentence should start with an emoji that fits the meaning of the sentence.
2. use various emoticons to make the summary more interesting.
3. CVE ID: Unique identifier of the vulnerability.
4. Vulnerability Description: Describes the nature and impact of the vulnerability.
5. Commentary: Describe the vulnerability using examples and analogies to make it easier for everyone to understand. 
6. Vulnerability Severity: A severity rating based on the CVSS score.
7. Known Affected Software : Output the software and version affected by the vulnerability and the patched version information in a table. 
8. Vulnerability Discovery Date and Reported Date: When the vulnerability was discovered and reported.
9. Please write everything in Korean.


CONTEXT:
{context}

SUMMARY:"
"""
prompt = PromptTemplate.from_template(prompt_template)



llm_chain = LLMChain(llm=llm, prompt=prompt)

text_splitter = RecursiveCharacterTextSplitter(        
    chunk_size=1000,     # 쪼개는 글자수
    chunk_overlap=100  # 오버랩 글자수
)

def crawl_docs(cve_id):
    # 크롤링할 URL 목록을 설정합니다.
    prefix_host = "https://nvd.nist.gov/vuln/detail/"
    urls = [prefix_host + cve_id]

    loader = WebBaseLoader(urls)

    # 문서를 로드합니다.
    docs = loader.load()
    return docs

def get_markdown_docs(docs):
    markdown_docs = [MarkdownifyHTMLProcessor().process(doc) for doc in docs]
    return markdown_docs

def summarize_docs(docs):
    markdown_docs = get_markdown_docs(docs)#
    llm_chain = LLMChain(llm=llm, prompt=prompt)    
    # StuffDocumentsChain 정의
    stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="context")
    response = stuff_chain.invoke({"input_documents": markdown_docs})
    return response['output_text']

def main():
    st.title("CVE 취약점 분석기")
    cve_id = st.text_input("CVE ID를 입력하세요:", placeholder="CVE-2021-3999")

    st.caption('CVE 데이타는 https://nvd.nist.gov/ 에서 실시간으로 요청해서 제공합니다.')
    if st.button("분석하기"):
        if not re.match(r'^CVE-\d{4}-\d{4,7}$', cve_id):
            st.error("올바른 CVE ID를 입력하세요.")        
        else:
            with st.spinner('CVE 데이타 분석중...'):
                docs = crawl_docs(cve_id)
                markdown_docs = get_markdown_docs(docs)
                summary = summarize_docs(markdown_docs)
            st.markdown(summary)

if __name__ == "__main__":
    main()