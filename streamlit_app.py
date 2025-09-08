import streamlit as st
import math
import pandas as pd
import matplotlib.pyplot as plt
from googleapiclient.discovery import build
from datetime import datetime, timedelta, timezone
from collections import Counter
import re
import time
import requests
import json
import numpy as np

# Streamlit 앱 설정
st.set_page_config(
    page_title="통합 키워드 분석기",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 한글 폰트 설정 (Streamlit 환경)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# --- 1. YouTube 분석 클래스 (기존 코드와 동일) ---
class YouTubeSpreadAnalyzer:
    def __init__(self, api_key):
        self.youtube = build('youtube', 'v3', developerKey=api_key)
        self.api_key = api_key
    
    def search_videos_by_keyword(self, keyword, max_results=100, days_back=30):
        """키워드로 비디오 검색 (페이지네이션 및 배치 처리)"""
        published_after = (datetime.now(timezone.utc) - timedelta(days=days_back)).isoformat()
        video_ids = []
        next_page_token = None
        remaining_results = max_results
        
        while remaining_results > 0:
            current_max = min(50, remaining_results)
            try:
                search_response = self.youtube.search().list(
                    q=keyword,
                    part="id,snippet",
                    maxResults=current_max,
                    pageToken=next_page_token,
                    order="viewCount",
                    publishedAfter=published_after,
                    type="video",
                    relevanceLanguage="ko"
                ).execute()
            except Exception as e:
                st.error(f"YouTube API 호출 오류: {e}")
                break

            for item in search_response['items']:
                if item['id']['kind'] == 'youtube#video':
                    video_ids.append(item['id']['videoId'])
            
            next_page_token = search_response.get('nextPageToken')
            if not next_page_token:
                break
            
            remaining_results -= current_max
            time.sleep(0.5)
        
        actual_count = len(video_ids)
        if actual_count == 0:
            return []
        
        videos = []
        batch_size = 50
        for i in range(0, actual_count, batch_size):
            batch_ids = video_ids[i:i + batch_size]
            try:
                video_response = self.youtube.videos().list(
                    part="statistics,snippet",
                    id=",".join(batch_ids)
                ).execute()
            except Exception as e:
                st.error(f"YouTube API 호출 오류: {e}")
                break

            for item in video_response['items']:
                view_count = int(item['statistics'].get('viewCount', 0))
                like_count = int(item['statistics'].get('likeCount', 0))
                comment_count = int(item['statistics'].get('commentCount', 0))
                
                videos.append({
                    'id': item['id'],
                    'title': item['snippet']['title'],
                    'channelTitle': item['snippet']['channelTitle'],
                    'publishedAt': item['snippet']['publishedAt'],
                    'viewCount': view_count,
                    'likeCount': like_count,
                    'commentCount': comment_count
                })
            
            time.sleep(0.5)
        
        return videos
    
    def calculate_spread_coefficient(self, videos):
        if not videos:
            return 0.0, 0.0, 0, 0.0
        
        total_views = 0
        total_weighted = 0
        video_count = len(videos)
        
        for video in videos:
            view_count = video['viewCount']
            like_count = video['likeCount']
            comment_count = video['commentCount']
            
            engagement = 0
            if view_count > 0:
                engagement = min(0.45, (like_count + comment_count) / view_count)
            
            weighted_views = view_count * (1 + engagement)
            total_weighted += weighted_views
            total_views += view_count
        
        avg_views = total_views / video_count
        avg_weighted = total_weighted / video_count
        
        if avg_weighted <= 1000:
            spread_coefficient = 0
        elif avg_weighted >= 5_000_000:
            spread_coefficient = 10.0
        else:
            spread_coefficient = (math.log10(avg_weighted) - 3) * (10 / (math.log10(5_000_000) - 3))
        
        return spread_coefficient, avg_weighted, total_views, avg_views
    
    def extract_common_keywords(self, titles, top_n=10):
        if not titles:
            return []
        
        stop_words = ["영상", "추천", "비디오", "Youtube", "YouTube", "보기", "최신", "인기", "급상승", "공개", "풀영상", "풀버전", "공식"]
        words = []
        for title in titles:
            clean_title = re.sub(r'[^\w\s#]', '', title)
            hashtags = re.findall(r'#(\w+)', clean_title)
            words.extend(hashtags)
            korean_words = re.findall(r'[\w#]*[가-힣]{2,}[\w#]*', clean_title)
            words.extend(korean_words)
        
        filtered_words = [word for word in words if word not in stop_words and len(word) > 1]
        word_counts = Counter(filtered_words)
        
        return [f"#{word}" for word, _ in word_counts.most_common(top_n)]
    
    def analyze_keyword_spread(self, keyword, days_back=30, max_results=100):
        videos = self.search_videos_by_keyword(keyword, max_results, days_back)
        
        if not videos:
            return {"error": "동영상을 찾을 수 없습니다"}, None
        
        spread_coeff, avg_weighted, total_views, avg_views = self.calculate_spread_coefficient(videos)
        
        top_videos = sorted(videos, key=lambda x: x['viewCount'], reverse=True)[:10]
        top_titles = [video['title'] for video in top_videos]
        common_keywords = self.extract_common_keywords(top_titles)
        
        result = {
            "keyword": keyword,
            "total_videos": len(videos),
            "total_views": total_views,
            "avg_views": avg_views,
            "avg_weighted_views": avg_weighted,
            "spread_coefficient": spread_coeff,
            "top_videos": top_videos,
            "common_keywords": common_keywords[:5],
            "days_back": days_back
        }
        
        return result, videos

# --- 2. Naver 분석 함수 (기존 코드와 동일) ---
def get_naver_search_index(keywords_dict, start_date, end_date):
    url = "https://openapi.naver.com/v1/datalab/search"
    headers = {
        "X-Naver-Client-Id": st.secrets["NAVER_CLIENT_ID"],
        "X-Naver-Client-Secret": st.secrets["NAVER_CLIENT_SECRET"],
        "Content-Type": "application/json"
    }
    
    keyword_groups = []
    for group_name, keywords in keywords_dict.items():
        keyword_groups.append({
            "groupName": group_name,
            "keywords": keywords
        })
    
    body = {
        "startDate": start_date,
        "endDate": end_date,
        "timeUnit": "date",
        "keywordGroups": keyword_groups
    }
    
    res = requests.post(url, json=body, headers=headers)
    res.raise_for_status()
    return res.json()

def calculate_absolute_index(main_data, ref_data_list):
    ref_max = 0
    for ref_df in ref_data_list:
        ref_max = max(ref_max, ref_df['ratio'].max())
    
    return (main_data['ratio'] / ref_max) * 100

def calculate_bti(naver_abs_index):
    return naver_abs_index / 100.0

# --- Streamlit UI 및 메인 로직 ---
st.title("통합 키워드 확산 분석기")
st.markdown("유튜브와 네이버 데이터를 활용하여 키워드의 확산력을 분석합니다.")

# 사이드바
with st.sidebar:
    st.header("🔑 키워드 및 설정")
    keyword = st.text_input("분석할 키워드", "창고형 약국")
    days_back = st.slider("분석 기간 (일)", 7, 365, 30)
    max_results = st.slider("YouTube 분석 동영상 수", 10, 200, 100)
    
    # 기준 키워드 (네이버 BTI용)
    st.subheader("네이버 BTI 기준 키워드")
    ref_keywords_str = st.text_input("콤마(,)로 구분하여 입력", "뉴스,날씨,유튜브")
    reference_keywords = [kw.strip() for kw in ref_keywords_str.split(',') if kw.strip()]
    
    run_button = st.button("🚀 분석 시작")

if run_button and keyword:
    try:
        # 탭 UI
        tab1, tab2 = st.tabs(["📊 유튜브 확산 분석", "📈 네이버 BTI 분석"])

        with tab1:
            st.header("📊 유튜브 확산 분석 결과")
            with st.spinner("YouTube 데이터를 불러오는 중..."):
                youtube_analyzer = YouTubeSpreadAnalyzer(st.secrets["YOUTUBE_API_KEY"])
                result, videos = youtube_analyzer.analyze_keyword_spread(
                    keyword,
                    days_back=days_back,
                    max_results=max_results
                )
            
            if "error" in result:
                st.error(f"오류 발생: {result['error']}")
            else:
                st.subheader(f"'{result['keyword']}' 키워드 확산 분석 (최근 {result['days_back']}일 기준)")
                st.metric("확산 계수 (SC)", f"{result['spread_coefficient']:.2f} / 10.00")
                
                st.info(f"**총 조회수**: {result['total_views']:,}회 | **평균 조회수**: {result['avg_views']:,.1f}회 | **평균 가중 조회수**: {result['avg_weighted_views']:,.1f}회")
                
                # SC 해석
                sc = result['spread_coefficient']
                sc_guide = ""
                if sc < 2.0: sc_guide = "미미한 영향 (여론 변동 없음)"
                elif sc < 4.0: sc_guide = "주목 요망 (이슈화 가능성)"
                elif sc < 6.0: sc_guide = "유의미한 영향 (공론화 시작)"
                elif sc < 8.0: sc_guide = "심각한 영향 (여론 악화)"
                elif sc < 10.0: sc_guide = "위기 수준 (국정 운영 영향)"
                else: sc_guide = "최고 위기 수준 (선거 판도 변경 가능성)"
                st.markdown(f"**해석**: {sc_guide}")

                st.markdown(f"**추천 해시태그**: `{'`, `'.join(result['common_keywords'])}`")
                
                st.subheader("상위 동영상 목록")
                top_videos_df = pd.DataFrame(result['top_videos'])
                top_videos_df['publishedAt'] = pd.to_datetime(top_videos_df['publishedAt']).dt.strftime('%Y-%m-%d')
                top_videos_df = top_videos_df[['channelTitle', 'title', 'viewCount', 'likeCount', 'commentCount', 'publishedAt']]
                st.dataframe(top_videos_df, use_container_width=True)
                
                st.subheader("주요 시각화")
                # 시각화 부분 (Streamlit에 맞게 수정)
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                
                # 1. 조회수 분포
                df_videos = pd.DataFrame(videos).sort_values('viewCount', ascending=False)
                ax1.bar(range(len(df_videos)), df_videos['viewCount'], color='skyblue')
                ax1.set_title('비디오별 조회수 분포')
                ax1.set_xlabel('비디오 순서 (조회수 기준)')
                ax1.set_ylabel('조회수')
                
                # 2. 확산 지표 비교
                metrics = ['총 조회수', '평균 조회수', '평균 가중 조회수']
                values = [
                    result['total_views'],
                    result['avg_views'],
                    result['avg_weighted_views']
                ]
                colors = ['blue', 'green', 'orange']
                ax2.bar(metrics, values, color=colors)
                ax2.set_title('확산 지표 비교')
                ax2.set_ylabel('값')
                ax2.ticklabel_format(style='plain', axis='y')
                
                plt.tight_layout()
                st.pyplot(fig)

        with tab2:
            st.header("📈 네이버 BTI 분석 결과")
            with st.spinner("Naver 데이터를 불러오는 중..."):
                end_date = datetime.now().strftime("%Y-%m-%d")
                start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
                
                keywords_dict = {"main": [keyword]}
                for i, ref_kw in enumerate(reference_keywords):
                    keywords_dict[f"ref_{i}"] = [ref_kw]
                
                try:
                    naver_raw = get_naver_search_index(keywords_dict, start_date, end_date)
                    
                    results = {}
                    for res in naver_raw['results']:
                        group_name = res['title']
                        df = pd.DataFrame(res['data'])
                        df['date'] = pd.to_datetime(df['period'])
                        results[group_name] = df
                    
                    main_df = results['main']
                    ref_dfs = [df for key, df in results.items() if key.startswith('ref_')]
                    
                    main_df['abs_index'] = calculate_absolute_index(main_df, ref_dfs)
                    main_df['bti'] = calculate_bti(main_df['abs_index'])
                    
                    st.subheader(f"'{keyword}' 키워드 BTI 분석 (최근 {days_back}일 기준)")
                    st.metric("최근 30일 평균 BTI", f"{main_df['bti'].tail(30).mean():.2f}")
                    st.markdown("최근 7일 BTI 추이:")
                    st.dataframe(main_df[['date', 'bti']].tail(7).set_index('date'), use_container_width=True)

                    st.subheader("BTI 지수 추이 그래프")
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(main_df['date'], main_df['bti'], 'b-', linewidth=2, label='BTI 지수')
                    
                    main_df['30d_ma'] = main_df['bti'].rolling(window=30).mean()
                    ax.plot(main_df['date'], main_df['30d_ma'], 'r--', linewidth=2, label='30일 이동평균')
                    
                    ax.set_title(f'BTI Index Trend: {keyword} (Naver)')
                    ax.set_xlabel('날짜')
                    ax.set_ylabel('BTI 지수')
                    ax.grid(True, linestyle='--', alpha=0.7)
                    ax.legend()
                    st.pyplot(fig)
                
                except requests.exceptions.HTTPError as err:
                    if err.response.status_code == 401:
                        st.error("네이버 API 인증 오류: Client ID 또는 Client Secret이 올바르지 않습니다.")
                    else:
                        st.error(f"네이버 API 호출 오류: {err}")
                except Exception as e:
                    st.error(f"데이터 처리 중 오류 발생: {e}")
            
    except Exception as e:
        st.error(f"분석 중 예상치 못한 오류가 발생했습니다: {e}")
        st.warning("API 키를 올바르게 설정했는지 확인해 주세요.")