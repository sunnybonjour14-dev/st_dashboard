import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from glob import glob
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# 페이지 설정
st.set_page_config(
    page_title="네이버 쇼핑 트렌드 대시보드",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS (프리미엄 디자인)
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    h1, h2, h3 {
        color: #1e1e1e;
        font-family: 'Inter', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# 데이터 로드 함수
@st.cache_data
def load_all_data():
    DATA_DIR = "data"
    shopping_files = glob(os.path.join(DATA_DIR, "*_네이버쇼핑_*.csv"))
    blog_files = glob(os.path.join(DATA_DIR, "*_블로그_*.csv"))
    trend_files = glob(os.path.join(DATA_DIR, "*_쇼핑트랜드_*.csv"))
    
    if not (shopping_files and blog_files and trend_files):
        return None, None, None

    df_shop = pd.concat([pd.read_csv(f).assign(keyword=os.path.basename(f).split('_')[0]) for f in shopping_files], ignore_index=True)
    df_blog = pd.concat([pd.read_csv(f).assign(keyword=os.path.basename(f).split('_')[0]) for f in blog_files], ignore_index=True)
    df_trend = pd.concat([pd.read_csv(f).assign(keyword=os.path.basename(f).split('_')[0]) for f in trend_files], ignore_index=True)
    
    # 데이터 정제
    df_trend['period'] = pd.to_datetime(df_trend['period'])
    
    return df_shop, df_blog, df_trend

def clean_text(text):
    if pd.isna(text): return ""
    text = re.sub(r'<[^>]*>', '', text)
    text = re.sub(r'[^가-힣a-zA-Z\s]', '', text)
    return text

# 메인 실행
def main():
    st.sidebar.title("🔍 검색 및 설정")
    df_shop, df_blog, df_trend = load_all_data()

    if df_shop is None:
        st.error("데이터를 찾을 수 없습니다. 수집 스크립트를 먼저 실행해 주세요.")
        return

    # 사이드바 키워드 필터
    all_keywords = df_shop['keyword'].unique().tolist()
    selected_keywords = st.sidebar.multiselect("분석 키워드 선택", all_keywords, default=all_keywords)

    if not selected_keywords:
        st.warning("분석할 키워드를 하나 이상 선택해 주세요.")
        return

    # 데이터 필터링
    filtered_shop = df_shop[df_shop['keyword'].isin(selected_keywords)]
    filtered_blog = df_blog[df_blog['keyword'].isin(selected_keywords)]
    filtered_trend = df_trend[df_trend['keyword'].isin(selected_keywords)]

    st.title("📊 네이버 쇼핑 인사이트 & EDA 대시보드")
    st.markdown(f"**현재 분석 대상:** {', '.join(selected_keywords)}")

    # 탭 구성
    tab1, tab2, tab3, tab4 = st.tabs(["🏠 홈", "📈 트렌드 분석", "🛒 쇼핑 분석", "📝 콘텐츠 분석"])

    # --- Tab 1: 홈 ---
    with tab1:
        st.subheader("📌 데이터 요약")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("총 상품 수", f"{len(filtered_shop)}개")
        col2.metric("평균 최저가", f"{int(filtered_shop['lprice'].mean()):,}원")
        col3.metric("최고 트렌드 지수", f"{filtered_trend['ratio'].max()}%")
        col4.metric("블로그 포스팅 수", f"{len(filtered_blog)}개")

        st.divider()
        st.subheader("📋 수집 데이터 미리보기")
        st.dataframe(filtered_shop.head(10), use_container_width=True)
        
        # [표 1] 데이터 기본 정보 요약표
        st.write("**[표 1] 데이터 요약 정보**")
        summary_table = pd.DataFrame({
            "항목": ["총 데이터 로우", "고유 키워드 수", "중복 행 수", "결측치 합계"],
            "값": [len(filtered_shop), len(selected_keywords), filtered_shop.duplicated().sum(), filtered_shop.isnull().sum().sum()]
        })
        st.table(summary_table)

    # --- Tab 2: 트렌드 분석 ---
    with tab2:
        st.subheader("📈 키워드별 쇼핑 클릭 트렌드")
        
        # [그래프 1] 트렌드 비교 선 그래프 (Plotly)
        fig_trend = px.line(filtered_trend, x='period', y='ratio', color='keyword',
                            title="일자별 클릭량 상대 지수 변화",
                            labels={'ratio': '클릭 지수', 'period': '날짜'},
                            line_shape='spline', render_mode='svg')
        fig_trend.update_layout(hovermode="x unified")
        st.plotly_chart(fig_trend, use_container_width=True)

        # [표 2] 키워드별 트렌드 기술 통계
        st.write("**[표 2] 키워드별 트렌드 통계**")
        trend_stat = filtered_trend.groupby('keyword')['ratio'].agg(['mean', 'max', 'min', 'std']).reset_index()
        st.dataframe(trend_stat, use_container_width=True)

    # --- Tab 3: 쇼핑 분석 ---
    with tab3:
        st.subheader("🛒 가격 및 카테고리 심층 분석")
        
        c1, c2 = st.columns(2)
        
        with c1:
            # [그래프 2] 가격 분포 히스토그램
            fig_price = px.histogram(filtered_shop, x='lprice', color='keyword',
                                     title="상품 최저가 분포",
                                     labels={'lprice': '가격(원)', 'count': '빈도'},
                                     marginal='box', barmode='overlay')
            st.plotly_chart(fig_price, use_container_width=True)
            
            # [표 3] 카테고리별 상품 수
            st.write("**[표 3] 3차 카테고리 구성**")
            cat3_count = filtered_shop.groupby(['keyword', 'category3']).size().reset_index(name='상품 수')
            st.dataframe(cat3_count, use_container_width=True)

        with c2:
            # [그래프 3] 판매처 점유율 도넛 차트
            top_malls = filtered_shop['mallName'].value_counts().head(10).reset_index()
            top_malls.columns = ['mallName', 'count']
            fig_mall = px.pie(top_malls, values='count', names='mallName', hole=.4,
                             title="상위 10개 판매처 점유율")
            st.plotly_chart(fig_mall, use_container_width=True)
            
            # [표 4] 판매처별 상세 가격 통계
            st.write("**[표 4] 주요 판매처별 가격 요약**")
            mall_price_stat = filtered_shop[filtered_shop['mallName'].isin(top_malls['mallName'])].groupby('mallName')['lprice'].agg(['mean', 'median', 'std']).reset_index()
            st.dataframe(mall_price_stat, use_container_width=True)

        # [그래프 4] 키워드별 가격 박스플롯
        fig_box = px.box(filtered_shop, x='keyword', y='lprice', color='keyword',
                         points="all", title="키워드별 가격 분포 상세 (Box Plot)")
        st.plotly_chart(fig_box, use_container_width=True)

    # --- Tab 4: 콘텐츠 분석 ---
    with tab4:
        st.subheader("📝 블로그 이슈 및 키워드 분석")
        
        # TF-IDF 분석
        filtered_blog['clean_text'] = (filtered_blog['title'] + " " + filtered_blog['description']).apply(clean_text)
        
        vectorizer = TfidfVectorizer(max_features=20)
        tfidf_matrix = vectorizer.fit_transform(filtered_blog['clean_text'])
        words = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.sum(axis=0).A1
        word_scores = pd.DataFrame({'keyword': words, 'score': scores}).sort_values('score', ascending=True)

        col1, col2 = st.columns([2, 1])
        
        with col1:
            # [그래프 5] TF-IDF 키워드 바 차트
            fig_word = px.bar(word_scores, x='score', y='keyword', orientation='h',
                              title="블로그 핵심 키워드 중요도 (TF-IDF)",
                              color='score', color_continuous_scale='Viridis')
            st.plotly_chart(fig_word, use_container_width=True)

        with col2:
            # [표 5] 활동 블로거 빈도 테이블
            st.write("**[표 5] 상위 블로거 목록**")
            top_bloggers = filtered_blog['bloggername'].value_counts().head(15).reset_index()
            top_bloggers.columns = ['블로거', '포스팅 수']
            st.table(top_bloggers)

        st.divider()
        st.subheader("🔗 블로그 검색 결과 확인")
        # 동적 테이블 (Plotly Table은 아니지만 Streamlit DataFrame으로 대체하여 인터랙티브성 확보)
        st.dataframe(filtered_blog[['title', 'bloggername', 'postdate', 'link']].sort_values('postdate', ascending=False), use_container_width=True)

if __name__ == "__main__":
    main()
