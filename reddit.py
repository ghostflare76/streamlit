# import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage

# 환경 변수 로드
# from dotenv import load_dotenv
# load_dotenv()

# API 및 Reddit 정보 설정
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# REDDIT_USERNAME = os.getenv("REDDIT_USERNAME")
# REDDIT_PASSWORD = os.getenv("REDDIT_PASSWORD")
# REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
# REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")

GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
REDDIT_USERNAME = st.secrets["REDDIT_USERNAME"]
REDDIT_PASSWORD = st.secrets["REDDIT_PASSWORD"]
REDDIT_CLIENT_ID = st.secrets["REDDIT_CLIENT_ID"]
REDDIT_CLIENT_SECRET = st.secrets["REDDIT_CLIENT_SECRET"]

hide = """
<style>
div[data-testid="stConnectionStatus"] {
    display: none !important;
</style>
"""

st.markdown(hide, unsafe_allow_html=True)
                
# Reddit 토큰 생성 함수
@st.cache_data
def get_access_token():
    token_url = f"https://www.reddit.com/api/v1/access_token?grant_type=password&username={REDDIT_USERNAME}&password={REDDIT_PASSWORD}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    response = requests.post(token_url, auth=(REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET), headers=headers)
    if response.status_code != 200:
        st.error(f"Error acquiring access token: {response.status_code} - {response.text}")
        return None
    return response.json()["access_token"]

# Reddit 최신 리뷰 목록 가져오기 함수
def get_new_posts(subreddit, page_number):
    access_token = get_access_token()
    if not access_token:
        return None
    url = f"https://oauth.reddit.com/r/{subreddit}/new?limit={page_number}"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        st.error(f"Error fetching new posts: {response.status_code} - {response.text}")
        return None
    return response.json()

@st.cache_resource
def initialize_llm():
    return ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)

llm = initialize_llm()

def get_sentiment(text):
    prompt = HumanMessage(content=f"Analyze the sentiment of the following text: '{text}'. Return 'positive', 'negative', or 'neutral' based on the sentiment analysis.")
    response = llm([prompt])
    sentiment = response.content.strip().lower()
    # return sentiment
    if 'positive' in sentiment:
        return 'Positive'
    elif 'negative' in sentiment:
        return 'Negative'
    else:
        return 'Neutral'

def get_keyword(text):
    prompt = HumanMessage(content=f"Extract negative keywords that matter from the given text: '{text}'. Return list them separated by commas.")
    response = llm([prompt])
    return response.content.strip().lower()   

def process_post(post):
    text = post['selftext']
    sentiment = get_sentiment(text)
    if sentiment == 'Negative':
        keyword = get_keyword(text)
    else: 
        keyword = 'N/A'
    post_with_sentiment = {
        'title': post['title'],
        'sentiment': sentiment,
        'keyword' : keyword
    }
    return post_with_sentiment

# Streamlit 앱
def main():
    st.title("Reddit Sentiment Analysis App")
    # subreddit = st.text_input("Subreddit 이름을 입력하세요", "throneandliberty")

    subreddit = st.selectbox(
    "Subreddit 채널명을 선택하세요",
    ("throneandliberty", "BattleCrush", "stellarblade"))
    
    page_number = st.slider("리뷰 글 수를 입력하세요", 1, 100, 10)
    
    st.write("글 수 : ", page_number)
    st.caption('글 수가 많은 경우 분석시 시간이 오래 걸립니다.')

    if 'analysis_button' in st.session_state and st.session_state.analysis_button == True:
        st.session_state.running = True
    else:
        st.session_state.running = False 

    if st.button("실행", disabled=st.session_state.running, key='analysis_button'):

        data = get_new_posts(subreddit,page_number)
        if data:
            posts = data['data']['children']
            if posts: 
                titles = [post['data']['title'] for post in posts]
                selftexts = [post['data']['selftext'] for post in posts]
                commentcnts = [post['data']['num_comments'] for post in posts]
                scores = [post['data']['score'] for post in posts]
                urls = [post['data']['url'] for post in posts]
                df = pd.DataFrame({
                    'title': titles,
                    'selftext': selftexts,
                    'commentcnt': commentcnts,
                    'score': scores,
                    'url': urls
                })
                #st.write(df)
                posts_with_sentiments = []
                
                with st.spinner('리뷰글 감정 분석중...'):
                    for post in posts:
                        try:
                            post_with_sentiment = process_post(post['data'])
                        except Exception as e:
                            print (f"An error occured : {e}")
                            continue

                        posts_with_sentiments.append(post_with_sentiment)
                
                    processed_df = pd.DataFrame(posts_with_sentiments)

                st.success('분석 완료!')     
      
                st.session_state.running = False
                st.write(processed_df)
             
                sentiment_counts = processed_df['sentiment'].value_counts()

                fig, ax = plt.subplots()
                ax.bar(sentiment_counts.index, sentiment_counts.values, color=['blue', 'green', 'red'])
                st.pyplot(fig)

                # plt.title('Posts Sentiment Analysis')
                # plt.xlabel('Sentiment')
                # plt.ylabel('Number of Posts')
                # plt.xticks(rotation=0) 
                # plt.show()
   
                # # 워드 클라우드 생성
                # all_text = ' '.join(titles + selftexts)
                # wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
                # plt.figure(figsize=(10, 5))
                # plt.imshow(wordcloud, interpolation='bilinear')
                # plt.axis("off")
                # st.pyplot(plt)
                
            else:
                st.subheader('검색결과가 없습니다')
                st.session_state.running = False
        else:
            st.session_state.running = False  # 데이터를 가져오지 못한 경우에도 버튼을 다시 활성화



if __name__ == "__main__":
    main()