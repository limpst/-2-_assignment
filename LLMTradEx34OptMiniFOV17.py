import os
import json
import re
import time
import warnings
from pydantic import BaseModel, Field
from typing import Literal
from datetime import datetime, timedelta, date

import cloudscraper
import requests
import numpy as np
import mysql.connector
from bs4 import BeautifulSoup
from matplotlib import pyplot as plt
from mysql.connector import pooling
from scipy.optimize import minimize
from typing import TypedDict, List, Dict, Optional, Any
import urllib3

# --- [RAG 관련 라이브러리 통합] ---
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from CustomParentDocumentRetriever import CustomParentDocumentRetriever as ParentDocumentRetriever
from langchain_core.stores import InMemoryStore
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# ------------------------------

from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

import yfinance as yf
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import norm
from scipy.interpolate import interp1d

# 외부 모듈 (기존 유지)
from LLMTradEx34ScenarioScore import analyze_market_scenario, MarketScenario, MarketTrend, RiskLevel

# 이메일/스케줄
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import schedule

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------
# 1. 설정 및 초기화 (Configuration & Init)
# ---------------------------------------------------------
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

load_dotenv(override=True)

# 일반적인 Python 경고 숨기기
warnings.filterwarnings("ignore")

# sklearn 및 하위 라이브러리의 경고 숨기기
os.environ['PYTHONWARNINGS'] = 'ignore'

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 상수 설정
MULTIPLIER = 50000
API_BASE_URL = "https://openapi.ls-sec.co.kr:8080"
ACCESS_TOKEN = os.getenv("LS_ACCESS_TOKEN")
APP_KEY = os.getenv('APP_KEY')
APP_SECRET = os.getenv('APP_SECRET')
TOKEN_URL = f"{API_BASE_URL}/oauth2/token"
ACCESS_TOKEN_EXPIRES_AT = 0
TOTAL_CAPITAL = int(os.environ.get('TOTAL_CAPITAL', 100000000)) * 2

# --- NEW: Mini Future settings ---
MINI_FUTURE_FOCODE = os.getenv("MINI_FUTURE_FOCODE", "").strip()
MINI_FUTURE_INIT_MARGIN = float(os.getenv("MINI_FUTURE_INIT_MARGIN", "2500000"))  # KRW, 보수적 기본값

# LLM 및 RAG 구성 요소 초기화
# --- 수정 전 ---
# OPEN_AI_KEY = os.getenv('OPEN_AI_KEY')
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, api_key=OPEN_AI_KEY)

LLM_BASE_URL = "http://localhost:8080/v1"
EMBED_BASE_URL = "http://localhost:8081/v1"

llm = ChatOpenAI(
    model="local-llama",
    base_url=LLM_BASE_URL,
    api_key="no-key-needed",
    temperature=0.0,
    timeout=600,
    streaming=False
)

embeddings = OpenAIEmbeddings(
    model="local-model",
    base_url=EMBED_BASE_URL,
    api_key="no-key-needed"
)

scenario_title = ''


def clean_model_output(text: str) -> str:
    if not text:
        return ""

    text = str(text).strip()

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL | re.IGNORECASE).strip()

    bad_patterns = [
        r"Thinking Process:.*",
        r"Analyze the Request:.*",
        r"Analyze the Input Text:.*",
    ]
    for pattern in bad_patterns:
        text = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE).strip()

    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^```\s*", "", text).strip()
    text = re.sub(r"\s*```$", "", text).strip()

    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def safe_llm_text(prompt, fallback: str = "") -> str:
    try:
        start_t = time.time()

        if isinstance(prompt, str):
            print(f"🔍 [LLM DEBUG] prompt type=str len={len(prompt)}")
            prompt = [HumanMessage(content=prompt)]
        else:
            print(f"🔍 [LLM DEBUG] prompt type={type(prompt)}")

        # Qwen3 thinking 모드 비활성화: <think> 블록이 토큰을 소진하는 문제 방지
        if not isinstance(prompt[0], SystemMessage):
            prompt = [SystemMessage(content="/no_think")] + list(prompt)

        print("⏳ [LLM] invoke 시작")
        response = llm.invoke(prompt)
        print(f"⏳ [LLM] invoke 완료 ({time.time() - start_t:.2f}s)")

        raw = getattr(response, "content", "")
        # print(f"🔍 [LLM DEBUG] raw response={raw}")

        if isinstance(raw, list):
            raw = " ".join(
                x.get("text", str(x)) if isinstance(x, dict) else str(x)
                for x in raw
            )

        cleaned = clean_model_output(raw)
        print(f"🔍 [LLM DEBUG] cleaned={cleaned}")

        if not cleaned:
            print("⚠️ [LLM] 빈 응답 감지")
            return fallback

        return cleaned

    except Exception as e:
        print(f"⚠️ [LLM] 호출 실패: {e}")
        return fallback


def extract_json_block(text: str) -> dict:
    text = clean_model_output(text)

    if not text:
        return {}

    try:
        return json.loads(text)
    except Exception:
        pass

    if "```json" in text:
        try:
            inner = text.split("```json", 1)[1].split("```", 1)[0].strip()
            return json.loads(inner)
        except Exception:
            pass

    if "{" in text and "}" in text:
        try:
            start = text.find("{")
            end = text.rfind("}") + 1
            return json.loads(text[start:end].strip())
        except Exception:
            pass

    return {}


def normalize_sentiment_label(text: str) -> str:
    t = clean_model_output(text).strip().capitalize()

    if t in ["Positive", "Negative", "Neutral"]:
        return t

    low = t.lower()
    if "positive" in low:
        return "Positive"
    if "negative" in low:
        return "Negative"
    if "neutral" in low:
        return "Neutral"

    return "Neutral"


# DB 연결 설정
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER", "admin"),
    "password": os.getenv("DB_PASSWORD"),
    "database": os.getenv("DB_DATABASE", "LLM"),
}

db_pool = None
try:
    db_pool = pooling.MySQLConnectionPool(
        pool_name="db_pool",
        pool_size=3,
        pool_reset_session=True,
        **DB_CONFIG
    )
    print("✅ [System] DB Connection Pool 생성 완료")
except Exception as e:
    print(f"❌ [System] DB Pool 생성 실패: {e}")
    # raise SystemExit(1) # 로컬 테스트 시 DB 없으면 주석 처리 가능

# 공통 안전 장치 (코드 값 유지)
limits = [
    (0, min, 1025.0, "Deep OTM Call"),
    (1, min, 1022.5, "OTM Call"),
    (2, max, 375.0, "Deep OTM Put"),
    (3, max, 377.5, "OTM Put")
]

# [수정] 자산 5개로 확장 (옵션4 + 선물1)
TARGET_ASSETS = [
    {"name": "Call Strategy (Long)", "type": "Call"},  # 0
    {"name": "Call Hedge (Short)", "type": "Call"},  # 1
    {"name": "Put Strategy (Long)", "type": "Put"},  # 2
    {"name": "Put Hedge (Short)", "type": "Put"},  # 3
    {"name": "Mini Future (Delta)", "type": "Future"}  # 4  <-- NEW
]


# ---------------------------------------------------------
# 3. 만기/잔존일 계산
# ---------------------------------------------------------
def get_expiry_days(year, month, current_date=None):
    """
    2026년 n월 옵션 만기일(두 번째 목요일)까지 남은 일수 계산
    """
    if current_date is None:
        current_date = datetime.now().date()

    first_day = date(year, month, 1)

    # 첫 번째 목요일 찾기

    wday = first_day.weekday()
    days_to_thursday = (3 - wday + 7) % 7
    first_thursday = first_day + timedelta(days=days_to_thursday)

    # 두 번째 목요일
    second_thursday = first_thursday + timedelta(days=7)

    # 잔존일수
    days_remaining = (second_thursday - current_date).days

    return max(0, days_remaining), second_thursday


# ---------------------------------------------------------
# 2. 보조 함수 (Calculations & Helpers)
# ---------------------------------------------------------

# [수정] 현재 시스템 시간을 기준으로 잔존 시간 비율(Intraday Fraction) 계산 함수
def get_intraday_fraction():
    now = datetime.now()

    # 장 시작(08:45) 및 종료(15:45) 시간 설정
    market_open = now.replace(hour=8, minute=45, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=45, second=0, microsecond=0)

    # 1. 장 마감 후
    if now >= market_close:
        return 0.0

    # 2. 장 시작 전
    if now <= market_open:
        return 1.0

    # 3. 장 중
    total_seconds = (market_close - market_open).total_seconds()
    remaining_seconds = (market_close - now).total_seconds()

    return remaining_seconds / total_seconds


# get_expiry_days: 2026년 2월물
today = datetime.now().date()
year = today.year
month = today.month
days_left, expiry_date = get_expiry_days(year, month)
if days_left == 0:
    next_month = month + 1 if month < 12 else 1
    next_year = year if month < 12 else year + 1
    days_left, expiry_date = get_expiry_days(next_year, next_month)

# days_left, expiry_date = get_expiry_days(2026, 4)

# [적용] 현재 시간을 반영한 잔존일수 계산
current_time_ratio = get_intraday_fraction()

days_to_expiry = float(days_left) + current_time_ratio

print(f"🕒 [System Time] 현재 시간 비율: {current_time_ratio:.4f}")
print(f"⏳ [Expiry Info] 남은 일수(DTE): {days_to_expiry:.4f} days")


def fetch_latest_news_all(limit: int = 30):
    if db_pool is None:
        print("⚠️ [System] DB Pool 없음: 뉴스 조회 스킵")
        return []

    conn = None
    rows = []
    try:
        conn = db_pool.get_connection()
        cursor = conn.cursor(dictionary=True)
        query = """
                SELECT date, time, title, body, category
                FROM news_data
                WHERE category IN ('거시경제_LLM', '거시경제_KEY', '해외 증시_LLM', '해외 증시_KEY', '주도 섹터_LLM', '주도 섹터_KEY')
                ORDER BY date DESC, time DESC
                    LIMIT %s;
                """
        cursor.execute(query, (limit,))
        rows = cursor.fetchall()
    except mysql.connector.Error as err:
        print(f"❌ DB 조회 에러: {err}")
    except Exception as e:
        print(f"❌ DB 조회 예외: {e}")
    finally:
        try:
            if conn and conn.is_connected():
                conn.close()
        except Exception:
            pass
    return rows


def fetch_latest_news(limit: int = 30):
    if db_pool is None:
        print("⚠️ [System] DB Pool 없음: 뉴스 조회 스킵")
        return []

    conn = None
    rows = []
    try:
        conn = db_pool.get_connection()
        cursor = conn.cursor(dictionary=True)
        query = """
                SELECT date, time, title, body, category
                FROM news_data
                WHERE category LIKE '거시경제%'
                ORDER BY date DESC, time DESC
                    LIMIT %s;
                """
        cursor.execute(query, (limit,))
        rows = cursor.fetchall()
    except mysql.connector.Error as err:
        print(f"❌ DB 조회 에러: {err}")
    except Exception as e:
        print(f"❌ DB 조회 예외: {e}")
    finally:
        try:
            if conn and conn.is_connected():
                conn.close()
        except Exception:
            pass
    return rows


# 1. 출력 스키마 정의 (Type Safety 확보)
# class JudgeOutput(BaseModel):
#     final_consensus: str = Field(description="상승/하락 의견을 종합한 최종 합의문 (한국어). [Divergence] 섹션을 반드시 포함해야 함.")
#     market_trend: Literal["Bullish", "Bearish", "Volatile", "Neutral"] = Field(description="최종 시장 방향성")
#     risk_score: float = Field(ge=1.0, le=10.0, description="리스크 점수 (1: 안전, 10: 매우 위험)")
#     news_sentiment: Literal["Positive", "Negative", "Neutral"] = Field(description="뉴스 데이터의 전반적 심리")
#     divergence_note: str = Field(description="뉴스 심리와 가격 액션 간의 괴리에 대한 기술적 요약")


# --- [RAG 엔진 클래스: 3가지 핵심 기술 구현] ---
class FinancialRAGEngine:
    def __init__(self):
        # ② 청킹 및 계층적 인덱싱 (Parent Document Retrieval)
        # 검색용 작은 조각(Child)과 문맥 파악용 큰 조각(Parent)을 분리
        self.child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

        # Fixed: Ensure FAISS is initialized with the embedding model.
        # If the error was caused by the specific 'placeholder' text or metadata,
        # we ensure a standard initialization.
        self.vectorstore = FAISS.from_documents(
            [Document(page_content="system initialization", metadata={"id": "init"})],
            embeddings
        )
        self.docstore = InMemoryStore()

        self.retriever = ParentDocumentRetriever(
            vectorstore=self.vectorstore,
            docstore=self.docstore,
            child_splitter=self.child_splitter,
        )
        # 2. 데이터 로드 및 주입 실행
        self.ingest_knowledge_base()

        # self.retriever.add_documents(docs)

    def ingest_knowledge_base(self):
        # DB에서 뉴스 500개 정도 가져오기 (함수 재활용)
        # fetch_latest_news는 전역 함수이므로 바로 호출 가능하다고 가정
        past_news = fetch_latest_news_all(limit=100)

        docs = []
        for news in past_news:
            content = f"날짜: {news['date']}\n제목: {news['title']}\n내용: {news['body']}"
            docs.append(Document(page_content=content, metadata={"source": "db_news"}))

        if docs:
            self.retriever.add_documents(docs)
            print(f"\n✅ [RAG] DB 뉴스 {len(docs)}개 벡터화 완료\n")

    # ③ 쿼리 재작성 (Query Rewriting)
    def rewrite_query(self, market_context: str) -> str:
        # prompt = ChatPromptTemplate.from_template(
        #     "당신은 전문 퀀트 트레이더입니다. 아래의 시장 상황/Manager View를 분석하여, "
        #     "과거 유사한 금융 위기나 장세 사례를 찾기 위한 검색 최적화 쿼리(금융 전문 용어 포함)로 재작성하세요.\n"
        #     "현재 상황: {context}\n"
        #     "최적화된 검색어:"
        # )
        prompt = ChatPromptTemplate.from_template(
            "당신은 전문 퀀트 트레이더입니다. 다음 '시장 상황'에서 핵심 금융 변수(매크로, 지표, 심리 등)를 추출하여, "
            "과거 유사 사례(Historical Analogues)를 찾기 위한 검색 쿼리를 작성하세요.\n\n"
            "### 지침:\n"
            "1. 불필요한 설명 없이 '검색어 리스트'와 '최적화된 쿼리 조합'만 출력하세요.\n"
            "2. 금융 전문 용어(예: Bear steepening, Liquidity crunch 등)를 활용하세요.\n"
            "3. 입력 내용이 길 경우, 핵심적인 시장 충격 요소에만 집중하세요.\n\n"
            "현재 상황: {context}\n\n"
            "최적화된 검색어 및 쿼리:"
        )
        chain = prompt | llm | StrOutputParser()
        try:
            result = chain.invoke({"context": market_context})
            result = clean_model_output(result)
            return result if result else market_context[:300]
        except Exception as e:
            print(f"⚠️ [RAG Rewrite] 쿼리 재작성 실패: {e}")
            return market_context[:300]

    # ① 하이브리드 검색 (Hybrid Search)
    def hybrid_search(self, original_view: str, sql_news_list: List[Dict]) -> str:
        # 1. 쿼리 재작성 수행
        optimized_query = self.rewrite_query(original_view)

        # debug
        print(f"\n[Original view]\n{original_view}\n")
        print(f"\n[Optimized query]\n{optimized_query}\n\n")

        # 2. 벡터 검색 (의미적 검색)
        vector_docs = self.retriever.get_relevant_documents(optimized_query)
        vector_context = "\n".join([doc.page_content for doc in vector_docs])

        # 3. 키워드 기반 필터링 (SQL DB 뉴스 결합)
        keywords = optimized_query.split()
        keyword_hits = []
        for news in sql_news_list:
            if any(kw in news.get('title', '') or kw in news.get('body', '') for kw in keywords):
                keyword_hits.append(f"{news['title']}: {news['body'][:1500]}")

        keyword_context = "\n".join(keyword_hits[:5])
        # debug
        print(f"[유사 사례 및 지식]\n{vector_context}\n\n[실시간 관련 뉴스]\n{keyword_context}")

        return f"[유사 사례 및 지식]\n{vector_context}\n\n[실시간 관련 뉴스]\n{keyword_context}"


rag_engine = None
try:
    rag_engine = FinancialRAGEngine()
except Exception as e:
    print(f"⚠️ [RAG] 초기화 실패: {e}")
    rag_engine = None


# ---------------------------------------------------------
# 2. State 정의 (RAG 데이터 필드 추가)
# ---------------------------------------------------------
class QuantState(TypedDict, total=False):
    kospi_index: float
    market_iv: float
    manager_view: str
    rag_context: str  # <--- NEW: RAG 검색 결과 저장
    risk_aversion: float
    total_capital: float
    market_trend: str
    days_to_expiry: float
    target_month_code: str
    strikes: List[float]
    asset_codes: List[str]
    market_data: Dict[str, List[float]]

    expected_returns: List[float]
    vol_vector: List[float]
    covariance_matrix: List[List[float]]
    correlation_matrix: List[List[float]]

    optimal_weights: List[float]
    hedge_indices: List[int]
    final_report: str
    console_report: str

    margin_usage: float
    target_profit_pct: float
    stop_loss_pct: float
    futures_signal: Dict[str, Any]
    forex_news: str
    news_sentiment: str
    is_price_rising: bool
    divergence_note: str
    macro_pred: Dict[str, Any]
    raw_news_data: List[Dict]  # 하이브리드 검색용 원본 소스

    expected_return_pct: float
    anchor_name: str


class DebateState(TypedDict, total=False):
    news_context: str
    bull_opinion: str
    bear_opinion: str
    final_consensus: str
    market_trend: str
    risk_score: float
    news_sentiment: str
    divergence_note: str

    # self-improvement fields
    eval_score: int
    eval_critique: str
    is_sufficient: bool
    retry_count: int

class EvalOutput(BaseModel):
    score: int = Field(description="합의문의 논리적 완결성 및 뉴스 반영 점수 (1~10)")
    is_sufficient: bool = Field(description="추가 개선 없이 채택 가능한가")
    critique: str = Field(description="불충분하거나 보완이 필요한 부분에 대한 구체적 피드백")



class ScenarioManager:
    def __init__(self, pool, embeddings_model):
        self.pool = pool
        self.embeddings = embeddings_model
        self.vectorstore = None
        self.scenarios = {}

    def load_and_index_scenarios(self):
        """DB에서 시나리오를 읽어 FAISS 인덱스를 구축합니다."""
        if self.pool is None:
            print("⚠️ [Scenario] DB Pool이 없어 시나리오를 로드할 수 없습니다.")
            return

        conn = self.pool.get_connection()
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT id, scenario_name, market_description, mu, vol, corr FROM macro_scenarios")
            rows = cursor.fetchall()

            if not rows:
                print("⚠️ [Scenario] DB에 등록된 시나리오가 없습니다. 샘플 데이터를 먼저 입력하세요.")
                return

            docs = []
            for row in rows:
                # 1. 벡터 검색용 문서 생성
                content = f"장세명: {row['scenario_name']}\n상황설명: {row['market_description']}"
                # 2. 메타데이터에 파라미터 저장
                metadata = {
                    "id": row['id'],
                    "name": row['scenario_name'],
                    "mu": json.loads(row['mu']) if isinstance(row['mu'], str) else row['mu'],
                    "vol": json.loads(row['vol']) if isinstance(row['vol'], str) else row['vol'],
                    "corr": json.loads(row['corr']) if isinstance(row['corr'], str) else row['corr']
                }
                docs.append(Document(page_content=content, metadata=metadata))

            # FAISS 인덱싱
            self.vectorstore = FAISS.from_documents(docs, self.embeddings)
            print(f"✅ [Scenario] {len(docs)}개의 전략 시나리오 벡터화 완료")

        except Exception as e:
            print(f"❌ [Scenario] 로드 중 오류: {e}")
        finally:
            conn.close()

    def find_nearest_scenario(self, view_text: str):
        """현재 뷰와 가장 유사한 시나리오 파라미터를 찾습니다."""
        if self.vectorstore is None:
            return None

        # 유사도 기반 상위 1개 추출
        results = self.vectorstore.similarity_search(view_text, k=1)
        return results[0].metadata if results else None

    def update_successful_scenario(self, view_text, mu, vol, corr, expected_ret, anchor_name):
        """수익률이 높은 파라미터를 지식 베이스에 추가/업데이트"""
        if self.pool is None:
            return

        parts = view_text
        if "Judge Consensus" in view_text:
            parts = view_text.split("⚖️ [Judge Consensus]")[1].strip()

        nearest = self.find_nearest_scenario(parts)
        if nearest:
            print("\nNearest scenario: \n")
            print(nearest)

        conn = self.pool.get_connection()
        try:
            cursor = conn.cursor()

            summary_query = (
                "당신은 금융 요약기입니다. 아래의 긴 시장 뷰를 100자 이내의 핵심 상황 설명으로 요약하세요.\n"
                f"내용: {view_text}"
            )
            # summary = safe_llm_text(summary_query, fallback="시장 시나리오 요약 실패")[:100]
            summary = safe_llm_text(
                [HumanMessage(content=summary_query)],
                fallback="시장 시나리오 요약 실패"
            )

            insert_sql = """
                         INSERT INTO macro_scenarios (scenario_name, market_description, mu, vol, corr, created_at)
                         VALUES (%s,
                                 %s,
                                 %s,
                                 %s,
                                 %s,
                                 now())
                         """

            scenario_name = f"Success_{anchor_name}_{datetime.now().strftime('%Y%m%d')}_{expected_ret:.1f}%"

            cursor.execute(insert_sql, (
                scenario_name,
                summary,
                json.dumps(mu),
                json.dumps(vol),
                json.dumps(corr)
            ))
            conn.commit()
            print(f"🌟 [Self-Learning] 고수익 시나리오 저장 완료: {scenario_name}\n")

            self.load_and_index_scenarios()

        except Exception as e:
            print(f"❌ [Self-Learning] 저장 실패: {e}")
        finally:
            conn.close()


# 전역 변수 선언
scenario_manager = None


# ---------------------------------------------------------
# 3. 보조 함수들 (블랙숄즈, Greeks, API 호출 등 - 원본 유지)
# ---------------------------------------------------------
def issue_ls_access_token(force: bool = False) -> str:
    global ACCESS_TOKEN, ACCESS_TOKEN_EXPIRES_AT
    now = int(time.time())
    if (not force) and ACCESS_TOKEN and now < (ACCESS_TOKEN_EXPIRES_AT - 60):
        return ACCESS_TOKEN
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {"grant_type": "client_credentials", "appkey": APP_KEY, "appsecretkey": APP_SECRET, "scope": "oob"}
    resp = requests.post(TOKEN_URL, headers=headers, data=data, verify=False, timeout=50)
    j = resp.json()
    ACCESS_TOKEN = j.get("access_token")
    ACCESS_TOKEN_EXPIRES_AT = now + int(j.get("expires_in", 1800))
    return ACCESS_TOKEN


def get_headers(tr_cd: str, tr_cont: str = "N") -> Dict[str, str]:
    token = issue_ls_access_token()
    return {"Content-Type": "application/json; charset=UTF-8", "Authorization": f"Bearer {token}", "tr_cd": tr_cd,
            "tr_cont": tr_cont, "mac_address": "00:11:22:33:44:55"}


def ls_post_market_data(tr_cd: str, body: dict, timeout: int = 30) -> dict:
    url = f"{API_BASE_URL}/futureoption/market-data"
    headers = get_headers(tr_cd)
    resp = requests.post(url, headers=headers, data=json.dumps(body), verify=False, timeout=timeout)
    if resp.status_code == 401:
        issue_ls_access_token(force=True)
        resp = requests.post(url, headers=get_headers(tr_cd), data=json.dumps(body), verify=False, timeout=timeout)
    return resp.json()


def calculate_bs_all(S, K, T, r, sigma, option_type="Call"):
    if T <= 1e-5:
        intrinsic = max(0, S - K) if option_type == "Call" else max(0, K - S)
        return {"price": intrinsic, "delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "Call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        theta = (- (S * sigma * np.exp(-d1 ** 2 / 2) / (2 * np.sqrt(2 * np.pi * T))) - r * K * np.exp(
            -r * T) * norm.cdf(d2))
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
        theta = (- (S * sigma * np.exp(-d1 ** 2 / 2) / (2 * np.sqrt(2 * np.pi * T))) + r * K * np.exp(
            -r * T) * norm.cdf(-d2))
    gamma = np.exp(-d1 ** 2 / 2) / (S * sigma * np.sqrt(2 * np.pi * T))
    vega = S * np.sqrt(T) * np.exp(-d1 ** 2 / 2) / np.sqrt(2 * np.pi) / 100.0
    return {"price": price, "delta": delta, "gamma": gamma, "vega": vega, "theta": theta / 365.0}


# (중략: get_expiry_days, calculate_strikes_new, fetch_latest_news 등 모든 원본 함수 유지)
# ... [원본 코드의 모든 유틸리티 함수 포함됨] ...

# ---------------------------------------------------------
# 4. LangGraph 노드 (RAG 로직 통합)
# ---------------------------------------------------------

def rag_engine_node(state: QuantState):
    """[RAG Node] 하이브리드 검색을 통해 지식 베이스 추출"""
    print("📡 [RAG Engine] 시장 상황/Manager View에 맞는 News 검색 중...")

    if rag_engine is None:
        print("⚠️ [RAG Engine] RAG 사용 불가 - 빈 컨텍스트로 진행")
        return {"rag_context": ""}

    view = state.get('manager_view', '')
    raw_news = state.get('raw_news_data', [])
    consensus_only = view

    if "Judge Consensus" in view:
        consensus_only = view.split("⚖️ [Judge Consensus]")[1].strip()
        consensus_only = consensus_only.split("- NOTE")[0].strip()

    print("\n[DEBUG] consensus_only: ")
    print(consensus_only)

    context = rag_engine.hybrid_search(consensus_only, raw_news)
    return {"rag_context": context}


def quant_engine(state: QuantState):
    """[Modified] RAG 데이터를 참고하여 mu/vol 추정"""
    view = state['manager_view']
    rag_data = state.get('rag_context', '참고할 과거 데이터 없음')

    trend = state.get('market_trend', 'neutral').lower()
    macro_ret = state.get('macro_pred', {}).get('pred_pct', 0.0)

    iv = state['market_iv']

    consensus_only = view
    if "Judge Consensus" in view:
        consensus_only = view.split("⚖️ [Judge Consensus]")[1].strip()

    print("\n[DEBUG] consensus_only: ")
    print(consensus_only)

    # anchor = scenario_manager.find_nearest_scenario(consensus_only)
    anchor = scenario_manager.find_nearest_scenario(consensus_only) if scenario_manager else None

    anchor_info = ""
    anchor_mu_text = "[0.03, -0.01, 0.02, -0.01, 0.01]"

    anchor_name = "General_Market"
    if anchor:
        anchor_name = anchor.get('name', 'Scenario_Anchor')
        anchor_mu_text = str(anchor.get('mu', anchor_mu_text))
        anchor_info = (
            f"\n[과거 유사 시나리오 기준값: {anchor.get('name', 'Scenario_Anchor')}]\n"
            f"- 기준 mu: {anchor.get('mu')}\n"
            f"- 기준 vol: {anchor.get('vol')}\n"
            f"- 기준 corr: {anchor.get('corr')}\n"
        )
    else:
        anchor_info = "\n[참조할 과거 시나리오 없음: 기본 파라미터 사용]\n"

    print(anchor_info)

    prompt = (
        "SYSTEM: You are a quantitative risk management engine. "
        "Output MUST be a strictly valid JSON object and NOTHING ELSE. "
        "No markdown, no headers, no conversational text.\n\n"

        f"당신은 금융 분석 전문가입니다. 아래 [현재 상황]과 RAG로 추출된 [참고 Anchor]을 결합 분석하세요.\n"
        f"작성 규칙: 과거 시나리오(Anchor)를 기준으로 하되, 현재 IV와 뉴스 뉘앙스에 따라 mu, vol와 corr을 미세 조정(Fine-tune)하세요.\n"
        f"현재 상황: {consensus_only}\n\n"
        f"참고 RAG 데이터: {rag_data}\n\n"
        f"DECISION TREND: {trend.upper()}\n"
        f"IV: {iv}%\n\n"
        f"[참고 Anchor]: {anchor_info}\n\n"
        f"CRITICAL RULE: Your 'mu' and 'vol' MUST be centered around the provided [참고 Anchor] values.\n"
        f"Macro Predictor says {macro_ret}%. If this contradicts the sentiment, favor the Macro Predictor's direction for the 'mu' vector.\n"
        f"Anchor mu was: {anchor_mu_text}. Adjust it by no more than 20% based on current IV.\n"
        f"Analyze the market view and estimate parameters for 5 Assets.\n"
        f"Assets: [Call_Long, Call_Short, Put_Long, Put_Short, Mini_Future]\n"
        f"1. Deep OTM Call Long (Bull/Convexity)\n"
        f"2. OTM Call Short (Bear/Income/Hedge against Bull)\n"
        f"3. Deep OTM Put Long (Bear/Convexity)\n"
        f"4. OTM Put Short (Bull/Income/Hedge against Bear)\n"
        f"5. Mini KOSPI200 Future (Delta Hedge; Linear)\n\n"
        f"Return JSON with 'mu' (5 items), 'vol' (5 items), 'corr' (5x5 matrix).\n"
        "LOGIC RULE:\n"
        "1. 'Price Action' takes precedence over news sentiment.\n"
        "2. If price is rising despite negative news, it is a 'Bullish Climber'.\n"
        "3. For 'Bullish Climber', give bonus mu to Long assets.\n\n"
        "ASSET LIST (Indices 0-4):\n"
        "0: Deep OTM Call Long, 1: OTM Call Short, 2: Deep OTM Put Long, 3: OTM Put Short, 4: Mini KOSPI200 Future\n\n"
        "REQUIRED OUTPUT FORMAT (JSON ONLY):\n"
        "{\n"
        "  \"mu\": [float, float, float, float, float],\n"
        "  \"vol\": [0.2, 0.2, 0.3, 0.3, 0.15],\n"
        "  \"corr\": [[5x5 matrix of floats]]\n"
        "}\n"
    )

    print(prompt)

    try:
        # raw_response = safe_llm_text(prompt, fallback="")
        raw_response = safe_llm_text([HumanMessage(content=prompt)], fallback="")
        print(f"DEBUG [LLM Response]:\n{raw_response}")

        data = extract_json_block(raw_response)
        if not data:
            raise ValueError("LLM JSON 파싱 실패")

        mu = list(data.get('mu', []))
        vol = list(data.get('vol', []))
        corr = np.array(data.get('corr', []), dtype=float)

        target_n = 5
        while len(mu) < target_n:
            mu.append(0.0)
        while len(vol) < target_n:
            vol.append(0.15)

        mu = mu[:target_n]
        vol = vol[:target_n]

        if corr.shape != (target_n, target_n):
            new_corr = np.eye(target_n)
            if corr.ndim == 2 and corr.shape[0] > 0 and corr.shape[1] > 0:
                min_dim = min(corr.shape[0], corr.shape[1], target_n)
                new_corr[:min_dim, :min_dim] = corr[:min_dim, :min_dim]
            corr = new_corr

        sigma = np.zeros((target_n, target_n))
        for i in range(target_n):
            for j in range(target_n):
                sigma[i][j] = corr[i][j] * vol[i] * vol[j]

        return {
            "expected_returns": mu,
            "vol_vector": vol,
            "covariance_matrix": sigma.tolist(),
            "correlation_matrix": corr.tolist(),
            "anchor_name": anchor_name
        }

    except Exception as e:
        print(f"⚠️ [QuantEngine] Error: {e}")
        n = 5
        return {
            "expected_returns": [0.01] * n,
            "vol_vector": [0.15] * n,
            "covariance_matrix": (np.eye(n) * 0.04).tolist(),
            "correlation_matrix": np.eye(n).tolist(),
            "anchor_name": anchor_name
        }


# (중략: market_data_fetcher, portfolio_optimizer, reporter, notifier 등 원본 노드 유지)
# ... [원본 코드의 노드들 모두 포함됨] ...

def compute_dynamic_risk_score(
    llm_score: float,
    macro_pred: dict,
    news_sentiment: str,
    is_price_rising: bool,
    market_trend: str,
    divergence_note: str,
    market_iv: float = 0.0,
) -> float:
    """
    LLM 베이스 점수에 실제 시장 데이터를 결합한 동적 리스크 점수 계산.
    결과: 1.0 ~ 10.0 클램핑 후 반환.
    """
    score = llm_score

    # 1. 매크로 예측 pred_pct (하락 전망일수록 최대 +2.5)
    pred_pct = float(macro_pred.get("pred_pct", 0.0))
    if pred_pct < 0:
        score += min(2.5, abs(pred_pct) * 0.5)

    # 2. 방향성 정확도 (불확실성 높을수록 최대 +1.5)
    dir_acc = float(macro_pred.get("directional_acc", 1.0))
    score += max(0.0, (1.0 - dir_acc) * 1.5)

    # 3. 뉴스 센티먼트
    sentiment = str(news_sentiment).strip().capitalize()
    if sentiment == "Negative":
        score += 1.5
    elif sentiment == "Positive":
        score -= 0.8

    # 4. 가격 방향
    if not is_price_rising:
        score += 0.6
    else:
        score -= 0.4

    # 5. 시장 트렌드
    trend = str(market_trend).strip().lower()
    if trend == "bearish":
        score += 1.2
    elif trend == "volatile":
        score += 0.6
    elif trend == "bullish":
        score -= 0.6

    # 6. 다이버전스 감지
    div = str(divergence_note).lower()
    if "divergence" in div or "괴리" in div:
        score += 0.5

    # 7. VKOSPI/IV (옵션)
    if market_iv >= 30:
        score += 2.0
    elif market_iv >= 25:
        score += 1.2
    elif market_iv >= 20:
        score += 0.6

    result = round(max(1.0, min(10.0, score)), 2)
    print(f"🎯 [DynamicRisk] LLM={llm_score:.1f} -> Dynamic={result:.2f} "
          f"(pred={pred_pct:+.2f}%, iv={market_iv:.1f}, trend={trend}, sentiment={sentiment})")
    return result


def divergence_checker_node(state: QuantState) -> Dict[str, Any]:
    macro_val = state.get('macro_pred', {}).get('pred_pct', 0.0)

    if macro_val < -0.5:
        new_trend = "bearish"
        correction_note = f"\n⚠️ [Macro Override] 매크로 예측치({macro_val}%)가 심각한 하락을 예고함. 뷰를 BEARISH로 전환."
        return {
            "market_trend": new_trend,
            "manager_view": state['manager_view'] + correction_note,
            "risk_aversion": min(10.0, state['risk_aversion'] + 1.0)
        }

    manager_view = state['manager_view']
    current_trend = state.get('market_trend', 'neutral').lower()

    news_sentiment = state.get("news_sentiment", "").strip().capitalize()
    if news_sentiment not in ["Positive", "Negative", "Neutral"]:
        sentiment_prompt = (
            f"Analyze the sentiment of the following market view: \"{manager_view}\"\n"
            f"Classification: Return ONLY one word from ['Positive', 'Negative', 'Neutral']."
        )
        news_sentiment = normalize_sentiment_label(
            safe_llm_text([HumanMessage(content=sentiment_prompt)], fallback="Neutral")
        )

    is_price_rising = bool(state.get("is_price_rising", True))

    correction_note = ""
    new_trend = current_trend

    if news_sentiment == "Negative" and is_price_rising:
        new_trend = "bullish"
        correction_note = "\n⚠️ [Divergence Alert] 뉴스는 부정적이나 시장의 회복력이 강력함. '상승 추세'로 강제 전환."

    elif news_sentiment == "Positive" and not is_price_rising:
        new_trend = "neutral"
        correction_note = "\n⚠️ [Divergence Alert] 호재에도 가격 반응 미약. 추세 판단 '중립/관망'으로 유보."

    if correction_note:
        print(f"🔄 correction_note: {correction_note}")
        return {
            "market_trend": new_trend,
            "manager_view": manager_view + correction_note,
            "risk_aversion": max(2.0, state['risk_aversion'] - 1.5) if new_trend == "bullish" else state['risk_aversion']
        }

    return {"market_trend": current_trend}



# ---------------------------------------------------------
# [Helper] Volatility Skew Curve Generator
# ---------------------------------------------------------
def get_iv_curve(atm_iv: float, strikes: List[float], atm_price: float) -> dict:
    """
    간략화된 변동성 스마일 곡선 생성 (Skew 반영)
    - Moneyness(log(K/S))를 기준으로 2차 함수 형태의 Skew를 적용합니다.
    - 일반적으로 주가지수 옵션은 OTM Put(낮은 행사가)의 IV가 높습니다.
    """
    iv_map = {}
    for k in strikes:
        # 0으로 나누기 방지
        if atm_price == 0:
            iv_map[k] = atm_iv
            continue

        moneyness = np.log(k / atm_price)

        # Skew Modeling:
        # -0.15 * moneyness : 낮은 행사가일수록 IV 상승 (Put Skew)
        # +0.5 * moneyness^2 : 양쪽 끝(Deep OTM)으로 갈수록 IV 상승 (Smile)
        skew_adjust = -0.15 * moneyness + 0.5 * (moneyness ** 2)

        adjusted_iv = atm_iv * (1 + skew_adjust)

        # IV가 음수가 되지 않도록 최소값 보정
        iv_map[k] = max(0.01, adjusted_iv)

    return iv_map


# ---------------------------------------------------------
# [Modified] Skew Aware Black-Scholes Calculator
# ---------------------------------------------------------
def calculate_bs_skew_aware(S, K, T, r, base_iv, option_type="Call"):
    """
    get_iv_curve를 사용하여 해당 행사가(K)에 맞는 보정된 IV를 구한 뒤,
    calculate_bs_all 함수를 호출하여 이론가 및 Greeks를 계산합니다.
    """
    # 1. get_iv_curve를 호출하여 현재 행사가(K)에 대한 Skew IV를 산출
    #    (get_iv_curve는 리스트를 입력받으므로 [K] 형태로 전달)
    iv_map = get_iv_curve(base_iv, [K], S)
    local_iv = iv_map.get(K, base_iv)

    # 2. 보정된 IV(local_iv)를 사용하여 블랙-숄즈 전체 계산 수행
    #    (calculate_bs_all 함수는 기존 코드에 정의되어 있다고 가정)

    return calculate_bs_all(S, K, T, r, local_iv, option_type)


# ---------------------------------------------------------
# [금리] 네이버 CD 91일물 금리 조회 (기존 코드 유지)
# ---------------------------------------------------------
def get_cd91_rate_final():
    url = "https://finance.naver.com/marketindex/interestDailyQuote.naver?marketindexCd=IRr_CD91"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    try:
        response = requests.get(url, headers=headers)
        response.encoding = 'euc-kr'

        soup = BeautifulSoup(response.text, 'html.parser')
        rows = soup.select('table.tbl_exchange > tbody > tr')

        for row in rows:
            cols = row.select('td')
            if len(cols) >= 2:
                date_txt = cols[0].text.strip()
                rate_txt = cols[1].text.strip()
                if date_txt and rate_txt:
                    return date_txt, rate_txt

        return None, "데이터 테이블에서 값을 추출하지 못했습니다."
    except Exception as e:
        return None, f"에러 발생: {e}"


# ---------------------------------------------------------
# 5. 행사가 계산 로직 (원본 유지)
# ---------------------------------------------------------
def calculate_strikes_new(atm: float, risk_aversion: float, iv: float, market_trend: str) -> List[float]:
    """
    [Tuning v3] 승률 향상을 위한 행사가 전진 배치
    - Long 포지션을 ATM(0.0)에 가깝게 붙여 델타 감도를 높임
    - Short 포지션을 적당히 벌려(Spread) 손익비 개선
    """
    dte = max(days_to_expiry, 0.1)
    time_factor = np.sqrt(dte / 252.0)
    sigma_move = (iv / 100.0) * time_factor
    trend = str(market_trend).lower()

    # [핵심] Z-Score를 0.0(ATM) 근처로 당겨서, 지수가 조금만 움직여도 수익이 나게 변경
    if trend == "bullish":
        # Call Long: ATM(-0.05), Call Short: OTM(0.4) -> Bull Call Spread
        # Put Long: Deep OTM(-0.8) 보험용
        z_scores = [-0.05, 0.40, -0.80, -0.60]

    elif trend == "bearish":
        # Put Long: ATM(0.05), Put Short: OTM(-0.4) -> Bear Put Spread
        # Call Long: Deep OTM(0.8) 보험용
        z_scores = [0.80, 0.60, 0.05, -0.40]

    elif trend == "volatile":
        # 양매수 성향 강화: 둘 다 ATM에 가깝게 붙임
        # Call Long(0.05) / Put Long(-0.05) -> Straddle에 가까운 Strangle
        # Hedge용 Short는 멀리 보냄
        z_scores = [0.05, 0.50, -0.05, -0.50]

    else:
        # Neutral: Iron Condor (기존 유지)
        z_scores = [0.6, 0.2, -0.6, -0.2]

    # 만기 임박 시 보정
    if dte < 5.0:
        z_scores = [z * 0.5 for z in z_scores]

    strikes = []
    for z in z_scores:
        val = atm * np.exp(z * sigma_move)
        strikes.append(val)

    # 2.5 단위 반올림 및 겹침 방지
    strikes = [max(float(round(s / 2.5) * 2.5), 0.0) for s in strikes]

    # 스프레드 간격 강제 확보 (최소 2.5pt)
    if strikes[1] <= strikes[0]: strikes[1] = strikes[0] + 2.5
    if strikes[2] <= strikes[3]: strikes[2] = strikes[
                                                  3] + 2.5  # Put Long > Put Short (Bearish/Volatile) logic check needed

    # Put Side 정렬 보정 (전략에 따라 다름)
    # Volatile/Bearish일 경우 Put Long(idx 2)이 Put Short(idx 3)보다 높아야 함(Debit)
    if trend in ["bearish", "volatile"]:
        if strikes[2] <= strikes[3]: strikes[2] = strikes[3] + 2.5
    # Bullish/Neutral일 경우 Put Long(idx 2)이 Put Short(idx 3)보다 낮아야 함(Credit)
    else:
        if strikes[3] <= strikes[2]: strikes[3] = strikes[2] + 2.5

    print(f"🔧 [Strike Tuned v3] Trend={trend.upper()} | Z={z_scores} | K={strikes}")
    return strikes


# ---------------------------------------------------------
# [신규] 미니 선물 전략 생성 노드 (코드 유지)
# ---------------------------------------------------------
def futures_strategy_engine(state: QuantState):
    kospi = state['kospi_index']
    iv = state['market_iv']
    trend = str(state['market_trend']).lower()
    risk_score = state['risk_aversion']

    daily_volatility = (iv / 100.0) / np.sqrt(252)
    expected_move = kospi * daily_volatility

    if trend == "bullish":
        action = "LONG (매수)"
        entry = kospi

        tp = entry + (expected_move * 1.5)
        sl = entry - (expected_move * 0.8)
        confidence = "High" if risk_score < 6.0 else "Medium"
        desc = "상승 모멘텀 추종. 눌림목 매수 유효."

    elif trend == "bearish":
        action = "SHORT (매도)"
        entry = kospi

        tp = entry - (expected_move * 2.0)
        sl = entry + (expected_move * 1.0)
        confidence = "High" if risk_score > 7.0 else "Medium"
        desc = "하락 압력 지속. 반등 시 매도 대응."

    elif trend == "volatile":
        action = "NEUTRAL (관망/단타)"
        entry = kospi

        tp = entry + (expected_move * 0.8)
        sl = entry - (expected_move * 0.8)
        confidence = "Low"
        desc = "방향성 부재. 박스권 매매 또는 옵션 양매수 유리."

    else:
        action = "WAIT (관망)"
        entry = 0.0
        tp = 0.0
        sl = 0.0
        confidence = "None"
        desc = "뚜렷한 시그널 대기 중."

    signal = {
        "action": action,
        "entry_price": entry,
        "target_price": tp,
        "stop_loss": sl,
        "expected_move": expected_move,
        "confidence": confidence,
        "description": desc
    }

    if action != "WAIT (관망)":
        print(f"\n🔮 [Futures Signal] {action} | Entry: {entry:.2f} | TP: {tp:.2f} | SL: {sl:.2f}")
    else:
        print(f"\n🔮 [Futures Signal] {action} - {desc}")

    return {"futures_signal": signal}


# 코드 기준: 4월물 코드
def _generate_option_code(strike: float, asset_type: str) -> str:
    k_int = int(strike)
    if asset_type.lower() == "call":
        return f"B0564{k_int}"
    else:
        return f"C0564{k_int}"


def fetch_option_data_from_api(focode: str):
    body = {"t2101InBlock": {"focode": focode}}
    DUMMY_DATA = {"price": 1.0, "delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}

    try:
        time.sleep(0.5)
        data = ls_post_market_data("t2101", body, timeout=30)
        if data.get("rsp_cd") == "00000" and "t2101OutBlock" in data:
            block = data["t2101OutBlock"]
            return {
                "price": float(block.get("price", 0.0) or 0.0),
                "delta": float(block.get("delt", 0.0) or 0.0),
                "gamma": float(block.get("gama", 0.0) or 0.0),
                "vega": float(block.get("vega", 0.0) or 0.0),
                "theta": float(block.get("ceta", 0.0) or 0.0)
            }
        return DUMMY_DATA
    except Exception:
        return DUMMY_DATA


def fetch_mini_future_data_from_api(focode: str, fallback_price: float) -> Dict[str, float]:
    """
    미니선물 데이터 조회 (완벽 보정 버전)
    - 가격이 0.0으로 들어오면 즉시 KOSPI200 지수로 폴백하여 P&L 왜곡 방지
    """
    data = {"t2101InBlock": {"focode": focode}}

    DUMMY = {"price": float(fallback_price), "delta": 1.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}

    if not focode or focode == "MINI_FUTURE_DUMMY":
        return DUMMY

    try:
        res_json = ls_post_market_data("t2101", data, timeout=30)

        block = res_json.get("t2101OutBlock", {})
        api_price = float(block.get("price", 0.0) or 0.0)

        final_price = api_price if api_price > 0 else float(fallback_price)

        return {"price": final_price, "delta": 1.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}

    except Exception as e:
        print(f"⚠️ [API] MINI KOSPI 200 조회 실패 ({e}). 지수({fallback_price})로 폴백합니다.")
        return DUMMY


def market_data_fetcher(state: QuantState):
    kospi = state['kospi_index']
    iv = state['market_iv'] / 100.0  # % -> 소수
    trend = state['market_trend']
    ra = state['risk_aversion']

    # 1. 만기 및 이자율 설정
    days_left_local = state.get('days_to_expiry', 28.0)
    T_years = max(days_left_local, 0.5) / 365.0  # 최소 0.5일 보정

    print("네이버 금융에서 CD 91일물 금리를 조회합니다...")
    standard_date, rate = get_cd91_rate_final()

    if standard_date:
        print("-" * 30)
        print(f"기준일: {standard_date}")
        print(f"CD 91일물 금리: {rate}%")
        print("-" * 30)
    else:
        print(rate)
        rate = "2.75"  # CD금리 조회 실패 시 기본값

    risk_free_rate = float(rate) / 100.0  # CD 91일물 금리

    # 2. 행사가 선정

    atm = round(kospi / 2.5) * 2.5
    strikes = calculate_strikes_new(atm, ra, iv * 100, trend)  # 함수엔 % 단위 전달

    # Safety Limits 적용
    for i, func, limit_val, name in limits:
        if i < len(strikes):
            strikes[i] = func(strikes[i], limit_val)

    codes = []
    raw_data_list = []

    print(f"\n📡 [MarketData] Fetching & Validating (ATM: {atm}, r: {risk_free_rate * 100:.2f}%)")

    # 3. API 데이터 수집

    for i, strike in enumerate(strikes):
        asset_type = TARGET_ASSETS[i]['type']
        focode = _generate_option_code(strike, asset_type)
        codes.append(focode)

        api_data = fetch_option_data_from_api(focode)

        mult = -1.0 if "Short" in TARGET_ASSETS[i]['name'] else 1.0

        raw_data_list.append({
            "idx": i,
            "strike": strike,
            "type": asset_type,
            "mult": mult,
            "data": api_data,
            "source": "API"
        })

    # 4. 정합성 검사 + Fallback
    final_data = {'price': [], 'delta': [], 'gamma': [], 'vega': [], 'theta': []}

    for i in range(len(raw_data_list)):
        curr = raw_data_list[i]
        use_fallback = False
        reason = ""

        # (1) 0원 체크
        if curr['data']['price'] <= 0.01:
            use_fallback = True
            reason = "Zero Price"

        # (2) 이론가 대비 괴리율 체크
        if not use_fallback:
            ref_bs = calculate_bs_skew_aware(kospi, curr['strike'], T_years, risk_free_rate, iv, curr['type'])
            ref_price = ref_bs['price']
            api_price = curr['data']['price']

            if abs(api_price - ref_price) > 0.5 and (api_price < ref_price * 0.5 or api_price > ref_price * 1.5):
                use_fallback = True
                reason = f"Price Deviation (API:{api_price} vs BS:{ref_price:.2f})"

        # (3) 확정
        if use_fallback:
            bs_ret = calculate_bs_skew_aware(kospi, curr['strike'], T_years, risk_free_rate, iv, curr['type'])

            p_price = bs_ret['price']
            p_delta = bs_ret['delta'] * curr['mult']
            p_gamma = bs_ret['gamma'] * curr['mult']
            p_vega = bs_ret['vega'] * curr['mult']
            p_theta = bs_ret['theta'] * curr['mult']

            print(
                f"   ⚠️ [Fallback] {TARGET_ASSETS[i]['name']} (K={curr['strike']}): {reason} -> Used BS Model (P={p_price:.2f})")
        else:
            d = curr['data']
            p_price = d['price']
            p_delta = d['delta'] * curr['mult']
            p_gamma = d['gamma'] * curr['mult']
            p_vega = d['vega'] * curr['mult']
            p_theta = d['theta'] * curr['mult']

            print(f"   ✅ [API] {TARGET_ASSETS[i]['name']} (K={curr['strike']}): P={p_price}, Δ={d['delta']:.2f}")

        final_data['price'].append(p_price)
        final_data['delta'].append(p_delta)
        final_data['gamma'].append(p_gamma)
        final_data['vega'].append(p_vega)
        final_data['theta'].append(p_theta)

    # 5. 미니 선물 처리

    fut = fetch_mini_future_data_from_api(MINI_FUTURE_FOCODE, fallback_price=kospi)
    codes.append(MINI_FUTURE_FOCODE if MINI_FUTURE_FOCODE else "MINI_FUTURE_DUMMY")

    final_data['price'].append(fut['price'])
    final_data['delta'].append(fut['delta'])
    final_data['gamma'].append(0.0)
    final_data['vega'].append(0.0)
    final_data['theta'].append(0.0)

    print(f"   - {TARGET_ASSETS[4]['name']}: P={fut['price']}, Δ={fut['delta']:.2f}")

    return {"strikes": strikes, "asset_codes": codes, "market_data": final_data}


def _fallback_portfolio_weights(market_trend: str) -> List[float]:
    t = str(market_trend).lower()
    if t == "bullish":
        return [0.18, 0.15, 0.05, 0.15, 0.17, 0.30]
    elif t == "bearish":
        return [0.05, 0.20, 0.18, 0.12, 0.15, 0.30]
    elif t == "volatile":
        return [0.20, 0.10, 0.20, 0.10, 0.10, 0.30]
    else:
        return [0.10, 0.20, 0.10, 0.20, 0.10, 0.30]

def _get_objective_weights(market_trend: str, risk_aversion: float) -> Dict[str, float]:
    weights = {
        "return": 1.0,
        "risk": risk_aversion * 2.0,
        "delta": 15.0,
        "vega": 2.0,
        "theta": 5.0,
        "gamma": 2.0,
        "concentration_penalty": 10.0,
        "direction_penalty": 15.0
    }

    trend = str(market_trend).lower()

    if trend == "bullish":
        weights["return"] = 1.5
        weights["delta"] = 50.0
        weights["risk"] = risk_aversion * 1.2

    elif trend == "bearish":
        weights["return"] = 1.0
        weights["delta"] = 25.0
        weights["theta"] = 6.0
        weights["risk"] = risk_aversion * 2.2

    elif trend == "volatile":
        weights["return"] = 1.0
        weights["vega"] = 4.0
        weights["gamma"] = 3.0
        weights["theta"] = 4.0
        weights["delta"] = 5.0
        weights["risk"] = risk_aversion * 2.0

    else:
        weights["return"] = 1.0
        weights["theta"] = 6.0
        weights["delta"] = 10.0
        weights["risk"] = risk_aversion * 2.0

    return weights


def get_expiration_effects(state: QuantState):
    dte = max(state.get('days_to_expiry', 4.0), 0.1)
    theta_acceleration = 1.0 / np.sqrt(dte)
    gamma_risk_weight = 1.0 / (dte ** 2) if dte < 1.0 else 1.0 / dte

    if dte <= 0.5:
        mode = "EXPIRATION_SCALPING"
    elif dte <= 1.5:
        mode = "THETA_ACCELERATION"
    else:
        mode = "TREND_FOLLOWING"

    return {"theta_weight": theta_acceleration, "gamma_weight": gamma_risk_weight, "mode": mode}


def get_dynamic_risk_targets(state: QuantState):
    dte = state.get('days_to_expiry', 4.0)
    trend = str(state.get('market_trend', 'neutral')).lower()

    vega_scale = np.sqrt(dte / 5.0)
    gamma_limit = 0.05 * (dte / 4.0)

    if trend == "bullish":
        target_delta, target_vega = 2.35, 0.05 * vega_scale
    elif trend == "bearish":
        target_delta, target_vega = -3.50, 0.15 * vega_scale
    elif trend == "volatile":
        target_delta, target_vega = -0.10, 0.40 * vega_scale
    else:
        target_delta, target_vega = 0.15, -0.10 * vega_scale

    return {
        "target_delta": target_delta,
        "target_vega": target_vega,
        "gamma_limit": gamma_limit,
        "vega_scale": vega_scale
    }


def portfolio_optimizer_greeks_mo(state: QuantState):
    mu = np.array(state['expected_returns'])
    sigma = np.array(state['covariance_matrix'])
    # [FIX] 변수명을 current_iv로 통일하여 NameError 방지
    current_iv = float(state.get('market_iv', 15.0))

    greeks = state['market_data']
    deltas = np.array(greeks['delta'])
    gammas = np.array(greeks['gamma'])
    thetas = np.array(greeks['theta'])
    vegas = np.array(greeks['vega'])

    trend = str(state['market_trend']).lower()
    risk_aversion = float(state['risk_aversion'])

    # 2. 리스크 타겟 및 만기 효과 중앙 집중 호출 (핵심 변경 사항)
    risk_params = get_dynamic_risk_targets(state)

    # 3. 타켓 값 할당
    target_delta = risk_params['target_delta']
    target_vega = risk_params['target_vega']

    MAX_WEIGHT = 0.45
    MIN_CASH = 0.20

    # [튜닝] 최대 허용 Debit 비율 축소: 12% -> 3%
    # 봇이 "돈을 많이 내는 포트폴리오"를 구조적으로 못 만들게 강제
    # MAX_DEBIT_RATIO = 0.85 if trend == "bearish" else 0.50
    MAX_DEBIT_RATIO = 0.50

    # local_min_cash = 0.10 if trend == "bearish" else MIN_CASH
    local_min_cash = 0.15 if trend == "bearish" else MIN_CASH

    # [수정] volatile에서는 선물 방향 노출을 더 엄격히 제한
    FUTURE_MAX_WEIGHT = 0.10 if trend == "volatile" else 0.30
    base_cash = 0.30 if "불확실" in state.get('manager_view', '') else MIN_CASH

    if trend == "bullish":
        hedge_indices = [1, 2]
    elif trend == "bearish":
        hedge_indices = [0, 3]
    elif trend == "volatile":
        hedge_indices = [1, 3]
    else:
        hedge_indices = [0, 2]

    obj_weights = _get_objective_weights(trend, risk_aversion)
    expiry_effects = get_expiration_effects(state)
    dte = state.get('days_to_expiry', 3.0)

    put_short_limit = MAX_WEIGHT  # spread 제약으로 naked put 차단 (bearish/volatile 공통)

    def multi_objective_cost(w, iv_val):
        """
        통합 옵션 포트폴리오 다목적 비용 함수
        :param w: 가중치 벡터 (마지막 요소는 현금 또는 조절 변수)
        :param iv_val: 현재 시장의 내재 변동성(IV)
        """
        # ---------------------------------------------------------
        # [1] 데이터 언패킹 및 기초 지표 산출
        # ---------------------------------------------------------
        w_assets = w[:-1]
        w_opt = w_assets[:4]  # [Call_Long, Call_Short, Put_Long, Put_Short]
        w_fut = w_assets[4]  # 선물 가중치

        # 기초 지표 계산
        port_return = np.dot(w_assets, mu)
        port_var = np.dot(w_assets.T, np.dot(sigma, w_assets))
        port_vol = np.sqrt(max(port_var, 1e-9))

        # 포트폴리오 그리스(Greeks) 산출
        curr_delta = np.dot(w_assets, deltas)
        curr_vega = np.dot(w_assets, vegas)
        curr_theta = np.dot(w_assets, thetas)
        curr_gamma = np.dot(w_assets, gammas)

        # IV 스케일링 팩터 (기준 IV: 15.0)
        iv_ref = 15.0
        iv_factor = np.sqrt(max(iv_val, 5.0) / iv_ref)

        # ---------------------------------------------------------
        # [Tuning] Bearish 및 전 장세 공통 안정화 로직
        # ---------------------------------------------------------
        est_net_debit = (w_opt[0] + w_opt[2]) - (w_opt[1] + w_opt[3]) * 0.4
        dist_to_limit = MAX_DEBIT_RATIO - est_net_debit

        if dist_to_limit < 0.001:
            # 지수 성벽(Exponential Wall)의 강도를 1e4로 낮추고
            # abs(dist_to_limit)에 따른 선형 증가를 결합해 엔진이 '탈출' 방향을 찾게 함
            penalty_coef = 1e3 if trend == "bearish" else 1e4
            f_debit_penalty = penalty_coef * (1.0 + abs(dist_to_limit) * 100.0)
        else:
            # 로그 장벽(Log Barrier)의 계수를 100.0으로 설정하여
            # 한도에 가까워질수록 '부드러운 압박'을 가함
            f_debit_penalty = -np.log(max(dist_to_limit, 1e-6)) * 50.0

        # ---------------------------------------------------------
        # [3] 스프레드 구조적 균형 (Spread Balance)
        # ---------------------------------------------------------
        f_spread_balance = 0.0
        SPREAD_RATIO_MIN = 0.4  # 매수 대비 매도 비중 최소 40%

        if trend == "bullish":
            if w_opt[0] > 0.05 and w_opt[1] < w_opt[0] * SPREAD_RATIO_MIN:
                f_spread_balance += (w_opt[0] * SPREAD_RATIO_MIN - w_opt[1]) ** 2 * 2000.0
        elif trend == "bearish":
            if w_opt[2] > 0.05 and w_opt[3] < w_opt[2] * SPREAD_RATIO_MIN:
                f_spread_balance += (w_opt[2] * SPREAD_RATIO_MIN - w_opt[3]) ** 2 * 2000.0

        # ---------------------------------------------------------
        # [4] 그리스 및 리스크 정규화 (Normalization & Weighting)
        # ---------------------------------------------------------
        # 4-1. 리스크 대비 수익 (Sharpe 스타일)
        f_return = -port_return / (port_vol + 0.1)

        # [수정] obj_weights["delta"]를 실제로 사용 (V16_3 로직 복원)
        # bearish/volatile에서는 1.5배 강화하여 delta 중립 유도
        effective_delta_weight = (obj_weights["delta"] * 1.5) if trend in ["bearish", "volatile"] else obj_weights["delta"]
        f_delta = ((curr_delta - target_delta) / 1.0) ** 2 * 500.0

        # 4-3. 베가 및 세타 (추세 조건부 가중치)
        f_vega = ((curr_vega - risk_params['target_vega']) / max(iv_val / 100.0, 0.1)) ** 2
        f_theta = -curr_theta * 70.0 * expiry_effects.get('theta_weight', 1.0) if trend not in ["volatile",
                                                                                                "bearish"] else 0.0

        # 4-4. 감마 리스크 (만기 시간 dte에 따른 기하급수적 강화)
        gamma_penalty_weight = 150.0 / (dte + 0.05)
        f_gamma_risk = (max(0, abs(curr_gamma) - risk_params['gamma_limit'])) ** 2 * gamma_penalty_weight

        if trend == "volatile" and curr_gamma < 0:
            # [수정] 변동성 장에서 음의 감마는 추가 벌점
            f_gamma_risk += (abs(curr_gamma) ** 2) * 4000.0

        # ---------------------------------------------------------
        # [5] 방향성 가드레일 (IV-Adaptive Dynamic Threshold)
        # ---------------------------------------------------------
        dynamic_threshold = 0.10 * iv_factor
        dynamic_lambda = 1000.0 / iv_factor
        f_direction_penalty = 0.0

        if trend == "bullish":
            if curr_delta < 0:
                f_direction_penalty = (abs(curr_delta) + 0.5) ** 2 * 50000.0
            elif curr_delta < 0.1:
                f_direction_penalty = (0.1 - curr_delta) ** 2 * 5000.0

        elif trend == "bearish":
            if curr_delta > 0:
                f_direction_penalty = (curr_delta + 0.5) ** 2 * 50000.0
            elif curr_delta > -0.1:
                f_direction_penalty = (curr_delta + 0.1) ** 2 * 5000.0

        # ---------------------------------------------------------
        # [6] 기타 구조적 페널티 및 특수 모드
        # ---------------------------------------------------------
        f_concentration = np.sum(w_assets ** 2) * 50.0  # 자산 집중 방지
        f_future_overuse = (max(0.0, abs(w_fut) - FUTURE_MAX_WEIGHT)) ** 2 * 3000.0  # 선물 증거금 가드

        # 만기 스캘핑 모드 시 페널티 (특수 목적 제약)
        mode_penalty = np.sum(w_assets ** 2) * 150.0 if expiry_effects.get('mode') == "EXPIRATION_SCALPING" else 0.0

        # ---------------------------------------------------------
        # [7] 최종 가중 합산
        # ---------------------------------------------------------
        return (
                obj_weights["return"] * f_return +
                obj_weights["risk"] * port_var * 1500.0 +
                effective_delta_weight * f_delta +
                obj_weights["vega"] * f_vega +
                obj_weights["theta"] * f_theta +
                f_spread_balance +  # 스프레드 구조 유지
                f_debit_penalty +  # 강력한 비용 제어
                f_gamma_risk +  # 감마 리스크 관리
                f_direction_penalty +  # 뷰 일치 가드레일
                f_future_overuse +  # 선물 사용량 제약
                f_concentration +  # 분산 투자 유도
                mode_penalty  # 특수 실행 모드 반영
        )

    min_hedge_ratio = 0.15
    INSURANCE_LIMIT = 0.05

    bounds = []
    for i in range(4):
        if i == 0:
            bounds.append((0.0, INSURANCE_LIMIT if trend == "bearish" else MAX_WEIGHT))
        elif i == 1:
            # [수정] volatile에서는 Call Short 상한 축소
            bounds.append((0.0, 0.15 if trend == "volatile" else MAX_WEIGHT))
        elif i == 2:
            put_long_min = 0.10 if trend in ("bearish", "volatile") else 0.0
            bounds.append((put_long_min, 0.65 if trend == "bearish" else MAX_WEIGHT))
        elif i == 3:
            bounds.append((0.0, put_short_limit))

    bounds.append((-FUTURE_MAX_WEIGHT, +FUTURE_MAX_WEIGHT))
    # bounds.append((MIN_CASH, 1.0))  # cash
    bounds.append((local_min_cash, 1.0))  # cash

    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},
        {'type': 'ineq', 'fun': lambda x: x[-1] - base_cash},
        {'type': 'ineq', 'fun': lambda x: np.sum([x[i] for i in hedge_indices]) - min_hedge_ratio}
    ]

    if trend == "bullish":
        constraints.append({'type': 'ineq', 'fun': lambda x: 0.40 - np.sum([x[1], x[2]])})
    if trend in ("bearish", "volatile"):
        # Put Spread 강제: Put Long >= Put Short (naked put 차단)
        constraints.append({'type': 'ineq', 'fun': lambda x: x[2] - x[3]})

    init_w = np.array(_fallback_portfolio_weights(trend))
    for i, (l, h) in enumerate(bounds):
        init_w[i] = np.clip(init_w[i], l, h)
    init_w[-1] = 1.0 - np.sum(init_w[:-1])

    print(f"\n🧩 [Multi-Objective Optimizer Tuned] Trend: {trend}, Weights: {obj_weights}")
    print(f"   • MAX_DEBIT_RATIO: {MAX_DEBIT_RATIO:.2%} | ThetaCoef: 50 | CostPenalty: 5000")
    print(f"   • Bounds Applied: {bounds}")

    try:
        res = minimize(
            multi_objective_cost,
            init_w,
            args=(current_iv,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 2000, 'ftol': 1e-6, 'eps': 1e-3, 'disp': False}
        )
        if res.success:
            print(f"\n✅ Optimization Success! (Cost: {res.fun:.4f})")
            print(f"   • Weights: {[round(x, 3) for x in res.x]}")
            return {"optimal_weights": res.x.tolist(), "hedge_indices": hedge_indices}
    except Exception as e:
        print(f"❌ Optimization Error: {e}")

    print("❌ Using Fallback Weights.")
    return {"optimal_weights": _fallback_portfolio_weights(trend), "hedge_indices": hedge_indices}



def calculate_refined_futures_qty(state: QuantState, opt_delta: float, opt_gamma: float) -> int:
    kospi = state['kospi_index']
    iv = state['market_iv'] / 100.0
    capital = state['total_capital']
    dte = max(state.get('days_to_expiry', 3.0), 0.1)

    # 1. 중앙 리스크 타켓 호출 (동기화의 핵심)
    risk_params = get_dynamic_risk_targets(state)
    target_delta = risk_params['target_delta'] # Optimizer와 동일한 목표값 사용

    daily_vol = iv * np.sqrt(dte / 252)
    expected_move = kospi * daily_vol
    adjusted_delta = opt_delta + (opt_gamma * expected_move)

    delta_gap = target_delta - adjusted_delta

    # [수정] 기존 capital/notional 방식은 단위가 섞여 선물 수량이 과도해질 수 있음
    # [수정] 계약 델타 기준으로 직접 보정
    min_hedge_threshold = 0.25 if str(state.get('market_trend', '')).lower() == "volatile" else 0.15

    if abs(delta_gap) < min_hedge_threshold:
        print(f"     ℹ️ [Hedge Skip] Gap({delta_gap:.3f}) < Threshold({min_hedge_threshold:.3f})")
        return 0

    fut_qty = int(round(delta_gap))

    # [수정] volatile에서는 과도한 선물 방향 노출 방지
    if str(state.get('market_trend', '')).lower() == "volatile":
        fut_qty = int(np.clip(fut_qty, -2, 2))

    return fut_qty


def find_beps(positions, current_kospi):
    """
    현재 포지션의 손익분기점(BEP)을 수치적으로 탐색합니다.
    Future까지 포함 가능하도록 확장.
    """
    beps = []
    scan_range = np.arange(current_kospi * 0.8, current_kospi * 1.2, 0.1)

    prev_pnl = None
    for s in scan_range:
        current_pnl = 0.0
        for p in positions:
            # --- NEW: Future payoff ---
            if p.get('option_type') == "Future":
                # Future는 선형: (S - entry) * MULTIPLIER

                if p['type'] == "Long":
                    unit_pnl = (s - p['price']) * MULTIPLIER
                else:
                    unit_pnl = (p['price'] - s) * MULTIPLIER
                current_pnl += unit_pnl * p['qty']
                continue

            # 옵션 내재가치
            if p.get('option_type') == "Call":
                expiry_val = max(0, s - p['strike'])
            else:
                expiry_val = max(0, p['strike'] - s)

            if p['type'] == "Long":
                unit_pnl = (expiry_val - p['price']) * MULTIPLIER
            else:
                unit_pnl = (p['price'] - expiry_val) * MULTIPLIER
            current_pnl += unit_pnl * p['qty']

        if prev_pnl is not None and prev_pnl * current_pnl <= 0:
            beps.append(round(s, 2))
        prev_pnl = current_pnl

    return beps


def get_risk_management_params(state: QuantState):
    trend = str(state.get('market_trend', 'neutral')).lower()
    risk_score = float(state.get('risk_aversion', 5.0))

    if trend == "volatile":
        tp, sl = 25.0, 10.0
    elif trend == "bearish":
        tp, sl = 15.0, 7.0
    else:
        tp, sl = 10.0, 5.0

    sl = sl * (1.0 - (risk_score - 5.0) * 0.05)
    return tp, sl


def _html_escape(s):
    if s is None:
        return ""
    s = str(s)
    return (s.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;"))


def execution_reporter_greeks(state: QuantState):
    weights, strikes, market_data = state.get('optimal_weights'), state.get('strikes'), state.get('market_data')
    capital, kospi = state['total_capital'], state['kospi_index']
    trend = str(state['market_trend']).lower()
    iv, view = state['market_iv'], state['manager_view']
    hedge_indices = state.get("hedge_indices", [])
    fut_signal = state.get("futures_signal", {})

    macro_pred = state.get("macro_pred", {}) or {}

    if not weights or not strikes or not market_data:
        return {"final_report": "Optimization Failed: Missing Market Data", "console_report": "Optimization Failed: Missing Market Data"}

    w_options = weights[:4]
    w_future_weight = weights[4]

    temp_positions = []
    total_port_greeks = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}

    def get_naked_margin(strike, price):
        return (strike * MULTIPLIER * 0.15)

    def get_unit_cost(strike, price, is_short):
        return get_naked_margin(strike, price) if is_short else price * MULTIPLIER

    for i, func, limit_val, name in limits:
        if i < len(strikes):
            strikes[i] = func(strikes[i], limit_val)

    direction = trend.capitalize()
    mapping = {"bullish": " 📈", "bearish": " 📉", "volatile": " ⚡"}
    direction += mapping.get(trend, " ⚖️")

    # STEP 1: 옵션 기본 수량 산출
    # [ISSUE 해결 3] 헤지 포지션의 이산화 손실 방지 (Floor Logic)
    for i, w in enumerate(w_options):
        asset = TARGET_ASSETS[i]
        price = market_data['price'][i]
        strike = strikes[i]
        is_short = "Short" in asset['name']
        u_cost = (strike * MULTIPLIER * 0.15) if is_short else (price * MULTIPLIER)

        # 기본 수량 (버림 처리)
        raw_qty = int((capital * w) // u_cost) if price > 0 and u_cost > 0 else 0

        # [Guardrail] 비중이 0.2% 이상인데 수량이 0이면 최소 1계약 할당 (hedge/main 모두)
        if w > 0.002 and raw_qty == 0:
            raw_qty = 1
            role_tag = "🛡️ Hedge" if i in hedge_indices else "🚀 Main"
            print(f"   [{role_tag} Guardrail] {asset['name']} 최소 1계약 강제 할당 (Weight: {w:.1%})")

        temp_positions.append({
            "idx": i, "name": asset['name'], "type": asset['type'],  # [FIX] missing 'type' key added
            "pos_type": "Short" if is_short else "Long",
            "strike": strike, "price": price,
            "unit_margin_naked": get_naked_margin(strike, price),
            "qty": raw_qty,
            "delta_unit": market_data['delta'][i], "gamma_unit": market_data['gamma'][i],
            "theta_unit": market_data['theta'][i], "vega_unit": market_data['vega'][i],
            "effective_margin": 0.0
        })

    print("\n🔍 [Reporter: Smart Spread Balancer]")
    pairs = [(2, 3), (0, 1)] if trend == "bearish" else [(0, 1), (2, 3)]
    for l_idx, s_idx in pairs:
        lp, sp = temp_positions[l_idx], temp_positions[s_idx]
        # if lp['qty'] > 0:
        #     sp['qty'] = lp['qty']
        if lp['qty'] > 0 and sp['qty'] == 0:
            sp['qty'] = max(1, int(lp['qty'] * 0.5))
        if sp['qty'] > 0 and lp['qty'] == 0:
            lp['qty'] = 1

    # STEP 2: 선물 헤지 수량
    opt_delta_for_hedge = sum(p['qty'] * p['delta_unit'] for p in temp_positions)
    opt_gamma_for_hedge = sum(p['qty'] * p['gamma_unit'] for p in temp_positions)

    fut_qty_signed = calculate_refined_futures_qty(state, opt_delta_for_hedge, opt_gamma_for_hedge)

    max_fut_contracts = int(
        (capital * abs(w_future_weight)) // MINI_FUTURE_INIT_MARGIN) if MINI_FUTURE_INIT_MARGIN > 0 else 0
    fut_qty_signed = int(np.clip(fut_qty_signed, -max_fut_contracts, +max_fut_contracts))

    # STEP 3: 증거금 체크 및 스케일링
    def get_total_margin_locked(t_pos, f_qty):
        m_locked = 0.0
        for l_idx, s_idx in pairs:
            lp, sp = t_pos[l_idx], t_pos[s_idx]
            naked_q = max(0, sp['qty'] - lp['qty'])
            sp['effective_margin'] = naked_q * sp['unit_margin_naked']
            m_locked += sp['effective_margin']
        m_locked += abs(f_qty) * MINI_FUTURE_INIT_MARGIN
        return m_locked

    initial_total_margin = get_total_margin_locked(temp_positions, fut_qty_signed)
    MARGIN_THRESHOLD, SAFETY_BUFFER = 0.70, 0.05
    current_ratio = initial_total_margin / capital if capital > 0 else 0

    if current_ratio > MARGIN_THRESHOLD:
        scale_factor = (MARGIN_THRESHOLD - SAFETY_BUFFER) / current_ratio

        print(
            f"\n⚠️ [Margin Control] Alert: Usage {current_ratio * 100:.1f}% exceeds threshold {MARGIN_THRESHOLD * 100}%.")
        print(f"   👉 Auto-scaling all positions by factor: {scale_factor:.2f}")

        for p in temp_positions:
            p['qty'] = int(p['qty'] * scale_factor)
        fut_qty_signed = int(fut_qty_signed * scale_factor)

        final_margin_locked = get_total_margin_locked(temp_positions, fut_qty_signed)
        print(f"   ✅ Adjusted Margin Utilization: {(final_margin_locked / capital) * 100:.1f}%")
    else:
        final_margin_locked = initial_total_margin

    # STEP 4: 포지션 확정
    positions = []
    total_spent_on_assets, total_premium_received, total_premium_paid = 0.0, 0.0, 0.0
    hedge_assets_list = []

    for p in temp_positions:
        if p['qty'] <= 0:
            continue

        premium_val = p['qty'] * p['price'] * MULTIPLIER
        if p['pos_type'] == "Long":
            total_premium_paid += premium_val
            total_spent_on_assets += premium_val
            actual_w = premium_val / capital
        else:
            total_premium_received += premium_val
            actual_w = p['effective_margin'] / capital

        total_port_greeks['delta'] += p['qty'] * p['delta_unit']
        total_port_greeks['gamma'] += p['qty'] * p['gamma_unit']
        total_port_greeks['vega'] += p['qty'] * p['vega_unit']
        total_port_greeks['theta'] += p['qty'] * p['theta_unit']

        is_hedge = p['idx'] in hedge_indices
        role = "🛡️ Hedge" if is_hedge else "🚀 Main"
        if is_hedge:
            hedge_assets_list.append(p['name'])

        positions.append({
            "name": p['name'], "strike": p['strike'], "type": p['pos_type'], "option_type": p['type'],
            "delta": p['delta_unit'], "gamma": p['gamma_unit'],
            "vega": p['vega_unit'], "theta": p['theta_unit'],
            "weight": actual_w, "qty": p['qty'], "price": p['price'],
            "amount": premium_val, "role": role
        })

    if fut_qty_signed != 0:
        fut_data = fetch_mini_future_data_from_api(MINI_FUTURE_FOCODE, kospi)
        fut_price = fut_data['price']
        fut_delta_unit = 1.0

        total_port_greeks['delta'] += fut_qty_signed * fut_delta_unit
        fut_margin = abs(fut_qty_signed) * MINI_FUTURE_INIT_MARGIN
        hedge_assets_list.append("Mini Future (Delta)")

        print(f"🧩 [Reporter] Mini Future Delta Hedge: Scaled Qty={fut_qty_signed:+d}")

        positions.append({
            "name": TARGET_ASSETS[4]['name'], "strike": 0.0,
            "type": "Long" if fut_qty_signed > 0 else "Short", "option_type": "Future",
            "delta": fut_delta_unit if fut_qty_signed > 0 else -fut_delta_unit,
            "gamma": 0.0, "vega": 0.0, "theta": 0.0,
            "weight": fut_margin / capital, "qty": abs(fut_qty_signed), "price": fut_price,
            "amount": 0.0, "role": "🛡️ Hedge"
        })

    # [수정] Directional Bias Alert 추가
    net_delta_notional = total_port_greeks['delta'] * kospi * MULTIPLIER
    delta_capital_ratio = (abs(net_delta_notional) / capital) if capital > 0 else 0.0

    DELTA_ALERT_CONTRACTS = 2.0
    DELTA_ALERT_CAPITAL_RATIO = 0.30

    if abs(total_port_greeks['delta']) >= DELTA_ALERT_CONTRACTS or delta_capital_ratio >= DELTA_ALERT_CAPITAL_RATIO:
        bias_side = "LONG DELTA BIAS" if total_port_greeks['delta'] > 0 else "SHORT DELTA BIAS"
        print(
            f"⚠️ [Directional Bias Alert] {bias_side} | "
            f"Delta={total_port_greeks['delta']:.2f} | "
            f"Notional={int(net_delta_notional):+,} KRW | "
            f"CapitalRatio={delta_capital_ratio * 100:.1f}%"
        )

    def calculate_expiry_pnl(target_s):
        total_pnl = 0.0
        for p in positions:
            if p.get('option_type') == "Future":
                unit_pnl = (target_s - p['price']) * MULTIPLIER if p['type'] == "Long" else (p[
                                                                                                 'price'] - target_s) * MULTIPLIER
            else:
                ev = max(0, target_s - p['strike']) if p['option_type'] == "Call" else max(0, p['strike'] - target_s)
                unit_pnl = (ev - p['price']) * MULTIPLIER if p['type'] == "Long" else (p['price'] - ev) * MULTIPLIER
            total_pnl += unit_pnl * p['qty']
        return total_pnl

    current_cash = capital - total_spent_on_assets + total_premium_received
    buying_power = current_cash - final_margin_locked
    net_premium = total_premium_received - total_premium_paid
    net_premium_str = f"{int(net_premium):+,} KRW ({'Credit' if net_premium >= 0 else 'Debit'})"

    found_beps = find_beps(positions, kospi)
    bep_str = ", ".join([str(b) for b in found_beps]) if found_beps else "None"
    tp, sl = get_risk_management_params(state)
    hedge_text = " + ".join(hedge_assets_list) if hedge_assets_list else "None"
    formatted_view = view.strip().replace(". ", ".\n" + " " * 15)

    # -------------------- [REPLACE START: HTML REPORT - NO TABLES LIST VIEW] --------------------
    html_style = """
    <style>
      :root{
        --bg:#ffffff;
        --text:#1f2937;
        --muted:#6b7280;
        --line:#e5e7eb;
        --soft:#f9fafb;
        --soft2:#f3f4f6;
        --accent:#2563eb;
        --good:#16a34a;
        --bad:#dc2626;
        --warn:#b45309;
        --tpbg: rgba(22,163,74,0.08);
        --slbg: rgba(220,38,38,0.08);
        --neutralbg: rgba(37,99,235,0.06);
      }

      body{
        background:var(--bg);
        color:var(--text);
        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
        margin:0;
        padding:16px;
      }

      h2{ margin:0 0 8px 0; font-size:18px; line-height:1.3; }
      h3{ margin:16px 0 10px 0; font-size:15px; }

      .meta{
        margin: 10px 0 10px 0;
        padding: 12px;
        border:1px solid var(--line);
        border-radius:10px;
        background:var(--soft);
      }
      .meta ul{ list-style:none; margin:0; padding:0; }
      .meta li{ color:var(--muted); font-size:13px; padding:2px 0; }
      .meta b{ color:var(--text); font-weight:800; }

      .summarybar{
        border:1px solid var(--line);
        border-radius:10px;
        background:#fff;
        padding:10px 12px;
        margin: 10px 0 12px 0;
      }
      .summarybar .title{ font-weight:900; font-size:13px; margin-bottom:8px; }

      .pill{
        display:inline-block;
        padding:3px 9px;
        border-radius:999px;
        font-size:12px;
        border:1px solid var(--line);
        background:var(--soft);
        color:var(--muted);
        line-height:1.4;
        margin-right:6px;
        margin-bottom:6px;
        white-space:nowrap;
      }
      .pill.good{ border-color: rgba(22,163,74,0.35); background: rgba(22,163,74,0.08); color: var(--good); font-weight:900; }
      .pill.bad{ border-color: rgba(220,38,38,0.35); background: rgba(220,38,38,0.08); color: var(--bad); font-weight:900; }
      .pill.warn{ border-color: rgba(180,83,9,0.35); background: rgba(180,83,9,0.08); color: var(--warn); font-weight:900; }
      .pill.accent{ border-color: rgba(37,99,235,0.35); background: rgba(37,99,235,0.08); color: var(--accent); font-weight:900; }

      .card{
        border:1px solid var(--line);
        border-radius:12px;
        background:#fff;
        overflow:hidden;
        margin: 10px 0;
      }
      .card-h{
        padding:10px 12px;
        background:var(--soft2);
        border-bottom:1px solid var(--line);
        font-weight:900;
        font-size:13px;
      }
      .card-b{ padding:12px; }

      /* Key/Value rows without <table> (email-safe) */
      .kv{
        display:block;
        border-top:1px solid var(--line);
        margin-top:10px;
      }
      .kv .row{
        display:block;
        padding:8px 0;
        border-bottom:1px solid var(--line);
      }
      .kv .row:last-child{ border-bottom:none; }
      .kv .k{
        display:block;
        color:var(--muted);
        font-size:12.5px;
        margin-bottom:4px;
      }
      .kv .v{
        display:block;
        font-weight:800;
        font-size:13px;
        word-break:break-word;
      }

      .mono{
        white-space: pre-wrap;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        font-size:12px;
        color:var(--text);
      }
      .muted{ color:var(--muted); }

      /* List cards (Positions / Expiry) */
      .list{ margin: 8px 0 16px 0; padding:0; }
      .item{
        border:1px solid var(--line);
        border-radius:12px;
        background:#fff;
        padding:12px;
        margin:10px 0;
      }
      .item .top{
        display:block;
        margin-bottom:8px;
      }
      .item .title{
        font-weight:900;
        font-size:14px;
        margin-bottom:6px;
      }
      .item .sub{
        color:var(--muted);
        font-size:12.5px;
        line-height:1.5;
      }
      .item .grid2{
        display:block; /* email-safe: 2열 레이아웃 대신 줄바꿈 */
        border-top:1px dashed var(--line);
        padding-top:10px;
        margin-top:10px;
      }
      .item .grid2 .row{
        padding:6px 0;
      }

      .tag{
        display:inline-block;
        padding:2px 8px;
        border-radius:999px;
        font-size:12px;
        border:1px solid var(--line);
        background:var(--soft);
        color:var(--muted);
        font-weight:800;
        margin-right:6px;
        margin-bottom:6px;
      }
      .tag.buy{ color:#d9534f; border-color: rgba(217,83,79,0.35); background: rgba(217,83,79,0.08); }
      .tag.sell{ color:#428bca; border-color: rgba(66,139,202,0.35); background: rgba(66,139,202,0.08); }
      .tag.tp{ color:var(--good); border-color: rgba(22,163,74,0.35); background: rgba(22,163,74,0.08); }
      .tag.sl{ color:var(--bad); border-color: rgba(220,38,38,0.35); background: rgba(220,38,38,0.08); }
      .tag.active{ color:var(--accent); border-color: rgba(37,99,235,0.35); background: rgba(37,99,235,0.08); }

      .item.tp{ background: var(--tpbg); border-color: rgba(22,163,74,0.30); }
      .item.sl{ background: var(--slbg); border-color: rgba(220,38,38,0.30); }
      .item.neutral{ background: var(--neutralbg); border-color: rgba(37,99,235,0.25); }

      .tag.call{ color:#2563eb; border-color: rgba(37,99,235,0.35); background: rgba(37,99,235,0.08); }
      .tag.put{ color:#7c3aed; border-color: rgba(124,58,237,0.35); background: rgba(124,58,237,0.08); }
      .tag.future{ color:#0f766e; border-color: rgba(15,118,110,0.35); background: rgba(15,118,110,0.08); }
      .item.call{ border-color: rgba(37,99,235,0.25); }
      .item.put{ border-color: rgba(124,58,237,0.25); }
      .item.future{ border-color: rgba(15,118,110,0.25); }

      .item.hedge{
        border-color: rgba(15,118,110,0.45);
        position: relative;
        padding-left: 22px; /* 좌측 아이콘 바 공간 */
      }
      .item.hedge:before{
        content: "🛡️";
        position: absolute;
        left: 0;
        top: 0;
        height: 100%;
        width: 18px;
        display: flex;
        align-items: flex-start;
        justify-content: center;
        padding-top: 10px;
        border-top-left-radius: 12px;
        border-bottom-left-radius: 12px;
        background: rgba(15,118,110,0.14);
        border-right: 1px solid rgba(15,118,110,0.25);
        font-size: 12px;
      }
      .item.hedge .hedge-badge{
        display: inline-block;
        margin-left: 6px;
        padding: 2px 8px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 900;
        border: 1px solid rgba(15,118,110,0.35);
        background: rgba(15,118,110,0.10);
        color: #0f766e;
        vertical-align: middle;
        white-space: nowrap;
      }
      .item .title{ display: block; }

      .item .main-badge{
        display: inline-block;
        margin-left: 6px;
        padding: 2px 8px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 900;
        border: 1px solid rgba(37,99,235,0.35);
        background: rgba(37,99,235,0.10);
        color: #2563eb;
        vertical-align: middle;
        white-space: nowrap;
      }

    </style>
    """

    # helper: kv row builder
    def _kv_row(k, v):
        return f"<div class='row'><div class='k'>{_html_escape(k)}</div><div class='v'>{v}</div></div>"

    report = f"<html><head>{html_style}</head><body>"

    # ----- Header -----
    report += f"<h2>🚀 Scenario: {_html_escape(scenario_title)}</h2>"
    report += "<h3>📊 LLM OPTION STRATEGY REPORT</h3>"

    report += '<div class="meta"><ul>'
    report += f"<li><b>Date:</b> {_html_escape(time.strftime('%Y-%m-%d %H:%M'))}</li>"
    report += f"<li><b>Underlying:</b> (miniV13-2) KOSPI 200 (Current: {kospi:.2f})</li>"
    report += f"<li><b>Market IV:</b> {_html_escape(iv)}%</li>"
    report += "</ul></div>"

    # ✅ Summary badges
    trend_label = str(state.get("market_trend", "") or "").strip()
    risk_label = float(state.get("risk_aversion", 0.0) or 0.0)
    is_credit = "(Credit)" in str(net_premium_str)

    credit_pill = '<span class="pill good">Credit</span>' if is_credit else '<span class="pill bad">Debit</span>'
    t_low = trend_label.lower()
    if "bull" in t_low:
        trend_pill = f'<span class="pill good">Trend: {_html_escape(trend_label)}</span>'
    elif "bear" in t_low:
        trend_pill = f'<span class="pill bad">Trend: {_html_escape(trend_label)}</span>'
    elif "vol" in t_low:
        trend_pill = f'<span class="pill warn">Trend: {_html_escape(trend_label)}</span>'
    else:
        trend_pill = f'<span class="pill accent">Trend: {_html_escape(trend_label)}</span>'

    if risk_label >= 7.5:
        risk_pill = f'<span class="pill warn">Risk: {risk_label:.1f}</span>'
    elif risk_label <= 3.5:
        risk_pill = f'<span class="pill good">Risk: {risk_label:.1f}</span>'
    else:
        risk_pill = f'<span class="pill accent">Risk: {risk_label:.1f}</span>'

    report += '<div class="summarybar">'
    report += '<div class="title">📌 Summary</div>'
    report += trend_pill + risk_pill + credit_pill
    report += "</div>"

    # ----- Macro Predictor card (no tables) -----
    report += '<div class="card"><div class="card-h">📈 [Macro Predictor]</div><div class="card-b">'
    if macro_pred:
        try:
            pred_pct = float(macro_pred.get('pred_pct', 0.0))
            curr_k = float(macro_pred.get('current_kospi', 0.0))
            next_k = float(macro_pred.get('next_kospi', 0.0))
            rising = bool(state.get('is_price_rising', True))
            rmse = float(macro_pred.get('rmse', 0.0))
            diracc = float(macro_pred.get('directional_acc', 0.0)) * 100.0

            report += "<div class='kv'>"
            report += _kv_row("- Predicted Return", f"{pred_pct:+.2f}%")
            report += _kv_row("- Current KOSPI200", f"{curr_k:.2f}")
            report += _kv_row("- Next KOSPI200", f"{next_k:.2f}")
            report += _kv_row("- is_price_rising",
                              '<span class="pill good">True</span>' if rising else '<span class="pill bad">False</span>')
            report += _kv_row("- (RMSE, DirAcc)", f"(RMSE {rmse:.5f}, DirAcc {diracc:.2f}%)")
            report += "</div>"
        except Exception:
            report += "<div class='mono'>- Macro predictor data available but formatting failed.</div>"
    else:
        report += "<div class='mono'>- 데이터 없음 (예측 실패 또는 미주입)</div>"
    report += "</div></div>"

    # ----- Futures Signal card -----
    report += '<div class="card"><div class="card-h">🔮 [Futures Signal Analysis]</div><div class="card-b">'
    if fut_signal:
        act = fut_signal.get('action', '-')
        conf = fut_signal.get('confidence', '-')
        dsc = fut_signal.get('description', '-')
        report += "<div class='kv'>"
        report += _kv_row("• Action Strategy",
                          f"{_html_escape(act)} <span class='pill'>Confidence: {_html_escape(conf)}</span>")
        report += _kv_row("• Signal Logic", _html_escape(dsc))
        if act != "WAIT (관망)":
            entry = float(fut_signal.get('entry_price', 0.0))
            target = float(fut_signal.get('target_price', 0.0))
            stop = float(fut_signal.get('stop_loss', 0.0))
            report += _kv_row("• Setup Guide", f"Entry {entry:.2f} | Target {target:.2f} | Stop {stop:.2f}")
        report += "</div>"
    else:
        report += "<div class='mono muted'>-</div>"
    report += "</div></div>"

    # ----- Financial Summary card -----
    report += '<div class="card"><div class="card-h">💰 [Financial Summary]</div><div class="card-b">'
    report += "<div class='kv'>"
    report += _kv_row("• Total Capital", f"{int(capital):,} KRW")
    report += _kv_row("• Cash Balance", f"{int(current_cash):,} KRW (Bank)")
    report += _kv_row("• Margin Locked", f"{int(final_margin_locked):,} KRW")
    report += _kv_row("• Buying Power", f"{int(buying_power):,} KRW")
    report += _kv_row("• Net Premium", f"{_html_escape(net_premium_str)} {credit_pill}")
    report += "</div></div></div>"

    # ----- Strategy Analysis card -----
    delta_pnl_1pct = total_port_greeks['delta'] * (kospi * 0.01) * MULTIPLIER
    final_leverage = abs(net_delta_notional) / capital if capital > 0 else 0.0
    delta_capital_pct = (net_delta_notional / capital) * 100 if capital > 0 else 0.0

    report += '<div class="card"><div class="card-h">🎯 [Strategy Analysis]</div><div class="card-b">'
    report += "<div class='kv'>"
    report += _kv_row("• Direction", _html_escape(direction))
    report += _kv_row("• Portfolio Delta",
                      f'{total_port_greeks["delta"]:.2f} <span class="pill">Leverage: {final_leverage:.1f}x</span>')
    report += _kv_row("• Net Delta Notional", f'{int(net_delta_notional):+,} KRW')
    report += _kv_row("• Delta / Capital", f'{delta_capital_pct:+.1f}%')
    report += _kv_row("• Portfolio Gamma", f'{total_port_greeks["gamma"]:+.4f}')
    report += _kv_row("• Portfolio Theta",
                      f'{total_port_greeks["theta"]:+.4f} <span class="pill">Daily Decay: {int(total_port_greeks["theta"] * MULTIPLIER):,} KRW</span>')
    report += _kv_row("• Est. P&L (±1%)", f"{int(delta_pnl_1pct):>+12,} KRW (Instant Move)")
    report += _kv_row("• Hedge Assets", _html_escape(hedge_text))
    report += "</div></div></div>"

    # ----- Break-Even card -----
    report += '<div class="card"><div class="card-h">🎯 [Break-Even Analysis]</div><div class="card-b">'
    report += "<div class='kv'>"
    report += _kv_row(" • Found BEP(s)", _html_escape(bep_str))
    if found_beps:
        dist_pct = ((found_beps[0] / kospi) - 1) * 100
        report += _kv_row(" • Dist. to BEP", f"{dist_pct:>+6.2f}% (지수가 이 지점에 도달 시 원금 보전)")
    report += "</div></div></div>"

    # =========================
    # Positions
    # =========================
    report += "<h3>📋 Positions</h3>"
    report += '<table style="border-collapse:collapse;width:100%;font-size:13px;">'
    report += (
        '<thead><tr style="background:#2c3e50;color:white;text-align:center;">'
        '<th style="padding:6px 8px;text-align:left;">Name</th>'
        '<th style="padding:6px 8px;">Type</th>'
        '<th style="padding:6px 8px;">Option</th>'
        '<th style="padding:6px 8px;">Strike</th>'
        '<th style="padding:6px 8px;">Price</th>'
        '<th style="padding:6px 8px;">Delta</th>'
        '<th style="padding:6px 8px;">Gamma</th>'
        '<th style="padding:6px 8px;">Weight</th>'
        '<th style="padding:6px 8px;">Qty</th>'
        '<th style="padding:6px 8px;">Role</th>'
        '</tr></thead><tbody>'
    )

    for p in positions:
        opt_type = str(p.get("option_type", "") or "")
        opt_low = opt_type.strip().lower()
        strike_val = float(p.get("strike", 0.0) or 0.0)
        price = float(p.get("price", 0.0) or 0.0)
        qty = int(p.get("qty", 0) or 0)
        weight_pct = float(p.get("weight", 0.0) or 0.0) * 100.0
        delta = float(p.get("delta", 0.0) or 0.0)
        gamma = float(p.get("gamma", 0.0) or 0.0)
        role = str(p.get("role", "") or "")
        role_low = role.lower()
        name = str(p.get("name", "") or "")
        is_long = p.get("type") == "Long"
        is_hedge = ("hedge" in role_low) or ("🛡️" in role)
        strike_display = "—" if opt_low == "future" else f"{strike_val:.1f}"
        row_bg = "#fff8f0" if is_hedge else "#ffffff"
        type_color = "#1565C0" if is_long else "#C62828"
        if opt_low == "call":
            opt_label = "📈 Call"
            opt_color = "#2E7D32"
        elif opt_low == "put":
            opt_label = "📉 Put"
            opt_color = "#AD1457"
        elif opt_low == "future":
            opt_label = "🧩 Future"
            opt_color = "#E65100"
        else:
            opt_label = _html_escape(opt_type)
            opt_color = "#555"

        report += f'<tr style="background:{row_bg};text-align:center;">'
        report += f'<td style="padding:5px 8px;text-align:left;font-weight:bold;">{_html_escape(name)}</td>'
        report += f'<td style="padding:5px 8px;color:{type_color};font-weight:bold;">{"Long" if is_long else "Short"}</td>'
        report += f'<td style="padding:5px 8px;color:{opt_color};">{opt_label}</td>'
        report += f'<td style="padding:5px 8px;">{strike_display}</td>'
        report += f'<td style="padding:5px 8px;">{price:.2f}</td>'
        report += f'<td style="padding:5px 8px;">{delta:.2f}</td>'
        report += f'<td style="padding:5px 8px;">{gamma:.3f}</td>'
        report += f'<td style="padding:5px 8px;">{weight_pct:.1f}%</td>'
        report += f'<td style="padding:5px 8px;">{qty}</td>'
        report += f'<td style="padding:5px 8px;">{_html_escape(role)}</td>'
        report += '</tr>'

    report += '</tbody></table>'

    # =========================
    # Expiry P&L
    # =========================
    report += "<h3>📊 Expiry P&L Scenario - Intrinsic Value Based</h3>"
    report += '<div class="list">'

    for change in [-0.12, -0.09, -0.06, -0.03, 0.0, 0.03, 0.06, 0.09, 0.12]:
        target_idx = kospi * (1 + change)
        pnl = calculate_expiry_pnl(target_idx)
        ret = (pnl / capital) * 100 if capital > 0 else 0.0

        status = "🎯 TP" if ret >= tp else ("🛑 SL" if ret <= -sl else "Active")
        if "TP" in status:
            item_cls = "item tp"
            status_tag = '<span class="tag tp">🎯 TP</span>'
        elif "SL" in status:
            item_cls = "item sl"
            status_tag = '<span class="tag sl">🛑 SL</span>'
        else:
            item_cls = "item neutral" if change == 0.0 else "item"
            status_tag = '<span class="tag active">Active</span>'

        ret_tag = ""
        if ret >= tp:
            ret_tag = f'<span class="tag tp">{ret:+.2f}%</span>'
        elif ret <= -sl:
            ret_tag = f'<span class="tag sl">{ret:+.2f}%</span>'
        else:
            ret_tag = f'<span class="tag">{ret:+.2f}%</span>'

        report += f'<div class="{item_cls}">'
        report += '<div class="top">'
        report += f'<div class="title">Index Move {change * 100:+.1f}%</div>'
        report += f'<div class="sub">{status_tag}{ret_tag}<span class="tag">Stable</span></div>'
        report += '</div>'

        report += '<div class="grid2">'
        report += f"<div class='row'><span class='k'>KOSPI 200: </span><span class='v'>{target_idx:.2f}</span></div>"
        report += f"<div class='row'><span class='k'>Expected P&L (KRW): </span><span class='v'>{int(pnl):,}</span></div>"
        report += f"<div class='row'><span class='k'>Return: </span><span class='v'>{ret:+.2f}%</span></div>"
        report += f"<div class='row'><span class='k'>Strategy: </span><span class='v'>{_html_escape(status)}</span></div>"
        report += "</div>"
        report += "</div>"

    report += "</div>"
    report += '<div class="mono muted">※ 본 시나리오는 만기 시점의 내재가치를 기준으로 하며, 중도 청산 시 그리스 변동에 따른 오차가 있을 수 있습니다.</div>'

    # ----- Risk & Margin card -----
    margin_ratio = (final_margin_locked / capital) * 100 if capital > 0 else 0.0
    util_pill = '<span class="pill warn">⚠️ High</span>' if margin_ratio > 70 else '<span class="pill good">✅ Stable</span>'

    report += '<div class="card"><div class="card-h">🛡️ [Risk & Margin Management]</div><div class="card-b">'
    report += "<div class='kv'>"
    report += _kv_row(" • Total Margin Locked", f"{int(final_margin_locked):,} KRW")
    report += _kv_row(" • Margin Utilization", f"{margin_ratio:.1f}% {util_pill}")
    report += _kv_row(" • Target Profit (TP)", f"+{tp:.1f}%")
    report += _kv_row(" • Stop Loss (SL)", f"-{sl:.1f}%")
    report += "</div></div></div>"

    # ----- Expiration card -----
    expiry_info = get_expiration_effects(state)
    report += '<div class="card"><div class="card-h">⏳ [Expiration Analysis]</div><div class="card-b">'
    report += "<div class='kv'>"
    report += _kv_row(" • Days to Expiry", f"{state.get('days_to_expiry', 3.0):.2f} Days")
    report += _kv_row(" • Strategy Mode", _html_escape(expiry_info["mode"]))
    report += _kv_row(" • Theta Intensity", f"x{expiry_info['theta_weight']:.2f}")
    report += "</div></div></div>"

    # ----- Global Macro Event card -----
    forex_news_text = state.get('forex_news', "No Major High Impact News.")
    if forex_news_text and "없습니다" not in forex_news_text:
        report += '<div class="card"><div class="card-h">🌍 [Global Macro Event (High Impact)]</div>'
        report += f'<div class="card-b"><pre class="mono">{_html_escape(forex_news_text)}</pre></div></div>'
    else:
        report += '<div class="card"><div class="card-h">🌍 [Global Macro Event]</div>'
        report += '<div class="card-b"><div class="mono">• 특이사항 없음 (No High Impact News)</div></div></div>'

    # ----- Market View card -----
    report += '<div class="card"><div class="card-h">🧠 Market View</div>'
    report += f'<div class="card-b"><pre class="mono">{_html_escape(view)}</pre></div></div>'

    report += "</body></html>"
    # -------------------- [REPLACE END: HTML REPORT - NO TABLES LIST VIEW] --------------------

    consoleReport = f"\n{'=' * 0}\n🚀 [Scenario: {scenario_title}]\n\n"
    consoleReport += f"📊  LLM OPTION STRATEGY REPORT (Risk Controlled)\n{'=' * 0}\n"
    consoleReport += (f"• Date: {time.strftime('%Y-%m-%d %H:%M')}\n")
    consoleReport += f"• Underlying: (miniV13-2) KOSPI 200 (Current: {kospi:.2f})\n"
    consoleReport += f"• Market IV: {iv}%\n{'-' * 0}\n"

    # --- NEW: Macro Predictor section in report ---
    if macro_pred:
        consoleReport += f"📈 [Macro Predictor]\n\n"
        try:
            consoleReport += f" - Predicted Return: {float(macro_pred.get('pred_pct', 0.0)):+.2f}%\n"
            consoleReport += f" - Current KOSPI200: {float(macro_pred.get('current_kospi', 0.0)):.2f}\n"
            consoleReport += f" - Next KOSPI200   : {float(macro_pred.get('next_kospi', 0.0)):.2f}\n"
            consoleReport += f" - is_price_rising : {bool(state.get('is_price_rising', True))}\n"
            consoleReport += f" - (RMSE {float(macro_pred.get('rmse', 0.0)):.5f}, DirAcc {float(macro_pred.get('directional_acc', 0.0)) * 100:.2f}%)\n\n"
        except Exception:
            consoleReport += f" - Macro predictor data available but formatting failed.\n\n"
    else:
        consoleReport += f"📈 [Macro Predictor]\n\n - 데이터 없음 (예측 실패 또는 미주입)\n\n"

    if fut_signal:
        act, conf, dsc = fut_signal.get('action', '-'), fut_signal.get('confidence', '-'), fut_signal.get('description',
                                                                                                          '-')

        consoleReport += f"🔮 [Futures Signal Analysis]\n\n• Action Strategy  : {act} (Confidence: {conf})\n• Signal Logic     : {dsc}\n"
        if act != "WAIT (관망)":
            consoleReport += f"• Setup Guide      : Entry {fut_signal.get('entry_price', 0.0):.2f} | Target {fut_signal.get('target_price', 0.0):.2f} | Stop {fut_signal.get('stop_loss', 0.0):.2f}\n"

    consoleReport += f"\n💰 [Financial Summary]\n\n• Total Capital    : {int(capital):,} KRW\n"
    consoleReport += f"• Cash Balance     : {int(current_cash):,} KRW (Bank)\n"
    consoleReport += f"• Margin Locked    : {int(final_margin_locked):,} KRW\n"
    consoleReport += f"• Buying Power     : {int(buying_power):,} KRW\n"
    consoleReport += f"• Net Premium      : {net_premium_str}\n\n"

    delta_pnl_1pct = total_port_greeks['delta'] * (kospi * 0.01) * MULTIPLIER
    final_leverage = abs(net_delta_notional) / capital if capital > 0 else 0.0
    delta_capital_pct = (net_delta_notional / capital) * 100 if capital > 0 else 0.0

    consoleReport += f"🎯 [Strategy Analysis]\n\n• Direction        : {direction}\n"
    consoleReport += f"• Portfolio Delta  : {total_port_greeks['delta']:.2f} (Leverage: {final_leverage:.1f}x)\n"
    consoleReport += f"• Net Delta Notional : {int(net_delta_notional):+,} KRW\n"
    consoleReport += f"• Delta / Capital  : {delta_capital_pct:+.1f}%\n"
    consoleReport += f"• Portfolio Gamma  : {total_port_greeks['gamma']:+.4f}\n"
    consoleReport += f"• Portfolio Theta  : {total_port_greeks['theta']:+.4f} (Daily Decay: {int(total_port_greeks['theta'] * MULTIPLIER):,} KRW)\n"
    consoleReport += f"• Est. P&L (±1%)   : {int(delta_pnl_1pct):>+12,} KRW (Instant Move)\n"
    consoleReport += f"• Hedge Assets     : {hedge_text}\n\n"

    consoleReport += f"🎯 [Break-Even Analysis]\n\n • Found BEP(s)     : {bep_str}\n"
    if found_beps:
        dist_pct = ((found_beps[0] / kospi) - 1) * 100
        consoleReport += f" • Dist. to BEP     : {dist_pct:>+6.2f}% (지수가 이 지점에 도달 시 원금 보전)\n"
    consoleReport += f"{'-' * 45}\n"

    consoleReport += f"📋 [Position]\n{'-' * 45}\n| {'Asset Name':<18} | {'Strike':<6} | {'Type':<5} | {'Delta':<6} | {'Gamma':<6} | {'Weight':<6} | {'Qty':<3} | {'Price':<6} | {'Role':<8} |\n{'-' * 45}\n"
    for p in positions:
        consoleReport += (
            f"| {p['name']:<18} | {p['strike']:<6.1f} | {p['type']:<5} | {p['delta']:>6.2f} | {p['gamma']:>6.3f} | {p['weight'] * 100:>5.1f}% | {p['qty']:>3} | {p['price']:>6.2f} | {p['role']:<8} |\n"
        )
    consoleReport += f"{'-' * 45}\n"

    consoleReport += f"\n📊 [Expiry P&L Scenario - Intrinsic Value Based]\n\n{'-' * 45}\n| Index Move | KOSPI 200 | Expected P&L (KRW) | Return |  Margin Status  | Strategy |\n{'-' * 45}\n"
    for change in [-0.12, -0.09, -0.06, -0.03, 0.0, 0.03, 0.06, 0.09, 0.12]:
        target_idx = kospi * (1 + change)
        pnl = calculate_expiry_pnl(target_idx)
        ret = (pnl / capital) * 100 if capital > 0 else 0.0
        status = "🎯 TP" if ret >= tp else ("🛑 SL" if ret <= -sl else "Active")
        margin_warning = "⚠️ Margin Rise" if (trend == "bearish" and change > 0.02) or (
                trend == "bullish" and change < -0.02) else "Stable"

        consoleReport += f"| {change * 100:>+9.1f}% | {target_idx:<9.2f} | {int(pnl):>18,} | {ret:>+6.2f}% | {margin_warning:<14} | {status:<10} |\n"

    margin_ratio = (final_margin_locked / capital) * 100 if capital > 0 else 0.0
    consoleReport += f"{'-' * 45}\n※ 본 시나리오는 만기 시점의 내재가치를 기준으로 하며, 중도 청산 시 그리스 변동에 따른 오차가 있을 수 있습니다.\n"
    consoleReport += f"\n 🛡️ [Risk & Margin Management]\n\n • Total Margin Locked : {int(final_margin_locked):,} KRW\n"
    consoleReport += f" • Margin Utilization  : {margin_ratio:.1f}% ({'⚠️ High' if margin_ratio > 70 else '✅ Stable'})\n"
    consoleReport += f" • Target Profit (TP)  : +{tp:.1f}%\n • Stop Loss (SL)      : -{sl:.1f}%\n"

    expiry_info = get_expiration_effects(state)
    consoleReport += f"\n ⏳ [Expiration Analysis]\n\n • Days to Expiry   : {state.get('days_to_expiry', 3.0):.2f} Days\n"
    consoleReport += f" • Strategy Mode    : {expiry_info['mode']}\n • Theta Intensity  : x{expiry_info['theta_weight']:.2f}\n"

    forex_news_text = state.get('forex_news', "No Major High Impact News.")

    if forex_news_text and "없습니다" not in forex_news_text:
        consoleReport += f"\n 🌍 [Global Macro Event (High Impact)]\n\n"
        consoleReport += f" {forex_news_text}\n"
        consoleReport += f" {'-' * 0}\n\n"
    else:
        consoleReport += f"\n 🌍 [Global Macro Event]\n\n• 특이사항 없음 (No High Impact News)\n{'-' * 0}\n\n"

    consoleReport += (f"{formatted_view}\n\n")

    return {"final_report": report, "console_report": consoleReport}


def send_email_message(subject: str, body: str):
    sender = os.getenv("EMAIL_SENDER")
    password = os.getenv("EMAIL_PASSWORD")
    receiver_a = os.getenv("EMAIL_RECEIVER_A")
    receiver_b = os.getenv("EMAIL_RECEIVER_B")

    if not sender or not password or not receiver_a:
        print("⚠️ [Notification] 이메일 설정이 누락되었습니다. (EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECEIVER_A 필수)")
        return

    # 수신자 목록 구성 (receiver_b는 선택)
    recipients = [r for r in [receiver_a] if r]  # , receiver_b

    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = ", ".join(recipients)
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'html'))

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender, password)
            server.sendmail(sender, recipients, msg.as_string())
        print(f"✅ [Notification] 이메일 전송 완료 -> {recipients}")
    except Exception as e:
        print(f"❌ [Notification] 이메일 전송 실패: {e}")


def notifier_node(state: QuantState):
    report = state.get('final_report', "No Report Generated")

    title = f"📢 [LLM Quant] {time.strftime('%Y-%m-%d')} 전략 리포트"

    send_email_message(title, report)
    return state


def learning_node(state: QuantState):
    """결과를 분석하여 성공적인 파라미터를 DB에 피드백"""

    target_profit = 3.0
    current_perf = state.get('expected_return_pct', 0.0)

    if current_perf >= target_profit:
        if scenario_manager is None:
            print("⚠️ [Learning] scenario_manager 없음 - 저장 스킵")
            return state

        print(f"🎯 [Learning] 수익률 {current_perf:.2f}% 확인. 우수 파라미터 학습 중...")

        scenario_manager.update_successful_scenario(
            view_text=state['manager_view'],
            mu=state['expected_returns'],
            vol=state.get('vol_vector', [0.2] * 5),
            corr=state.get('correlation_matrix', []),
            expected_ret=current_perf,
            anchor_name=state.get('anchor_name', 'Unknown_Anchor')
        )
    else:
        print(f"ℹ️ [Learning] 수익률 {current_perf:.2f}% - 학습 임계치 미달로 스킵.")

    return state


# ---------------------------------------------------------
# 5. Workflow 구성 (RAG 노드 삽입)
# ---------------------------------------------------------

workflow = StateGraph(QuantState)

# 노드 등록
workflow.add_node("RAG_Search", rag_engine_node)  # <--- RAG 노드 추가
workflow.add_node("Engine", quant_engine)
workflow.add_node("DivergenceChecker", divergence_checker_node)  # 원본 함수 사용
workflow.add_node("MarketData", market_data_fetcher)  # 원본 함수 사용
workflow.add_node("FuturesStrategy", futures_strategy_engine)  # 원본 함수 사용
workflow.add_node("Optimizer", portfolio_optimizer_greeks_mo)  # 원본 함수 사용
workflow.add_node("Reporter", execution_reporter_greeks)  # 원본 함수 사용
workflow.add_node("Notifier", notifier_node)  # 원본 함수 사용
workflow.add_node("Learning", learning_node)

# 엣지 연결 수정 (DivergenceChecker를 Engine 앞으로 이동)
workflow.set_entry_point("RAG_Search")

# 1. RAG로 과거 지식 추출
workflow.add_edge("RAG_Search", "DivergenceChecker")

# 2. 뉴스 심리 vs 가격 액션 검증 (Trend 및 View 확정)
# 여기서 결정된 market_trend와 수정된 manager_view가 Engine으로 전달됨
workflow.add_edge("DivergenceChecker", "Engine")

# 3. 확정된 View를 바탕으로 mu, vol, corr 추정
workflow.add_edge("Engine", "MarketData")

# workflow.set_entry_point("RAG_Search")
# workflow.add_edge("RAG_Search", "Engine")
# workflow.add_edge("Engine", "DivergenceChecker")
workflow.add_edge("MarketData", "FuturesStrategy")
workflow.add_edge("FuturesStrategy", "Optimizer")
workflow.add_edge("Optimizer", "Reporter")
workflow.add_edge("Reporter", "Notifier")
workflow.add_edge("Notifier", "Learning")
workflow.add_edge("Learning", END)

app = workflow.compile()


# ---------------------------------------------------------
# 10. Forex Factory 주요 뉴스 크롤링 (코드 유지)
# ---------------------------------------------------------
def get_forex_news():
    print("\n--- Forex Factory 뉴스 추출 시작 ---")

    try:
        scraper = cloudscraper.create_scraper()
        response = scraper.get("https://www.forexfactory.com/calendar?day=today")

        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', class_='calendar__table')

        if not table:
            print("캘린더 테이블을 찾을 수 없습니다. (HTML 구조 변경 또는 봇 차단 의심)")
            return None

        news_list = []
        rows = table.find_all('tr', class_='calendar__row')

        latest_time_str = "Tentative"

        for row in rows:

            time_ele = row.find('td', class_='calendar__time')
            if time_ele:
                time_text = time_ele.text.strip()

                if time_text:
                    latest_time_str = time_text

            impact_ele = row.find('span', class_='icon--ff-impact-red')

            if impact_ele:
                currency_ele = row.find('td', class_='calendar__currency')
                event_ele = row.find('td', class_='calendar__event')

                currency = currency_ele.text.strip() if currency_ele else "N/A"
                event = event_ele.text.strip() if event_ele else "N/A"

                news_item = f"[{latest_time_str}] {currency} - {event} (High Impact)"
                news_list.append(news_item)
                print(f"추출됨: {news_item}")

        if not news_list:
            msg = "오늘은 예정된 High Impact 뉴스가 없습니다."
            print(msg)
            return msg

        return "\n".join(news_list)

    except Exception as e:
        print(f"크롤링 중 오류 발생: {e}")
        return None


def get_kospi200_index() -> float:
    MANUAL_KOSPI_INDEX = 575.05
    data = {"t2101InBlock": {"focode": "A0166000"}}

    try:
        res_json = ls_post_market_data("t2101", data, timeout=60)
        return float(res_json["t2101OutBlock"]["kospijisu"])
    except Exception as e:
        print(f"⚠️ KOSPI 200 지수 조회 실패: {e}")
        print(f"👉 [System] API 오류로 인해 KOSPI 지수 ({MANUAL_KOSPI_INDEX})을 사용하여 진행합니다.")
        return MANUAL_KOSPI_INDEX


# ---------------------------------------------------------
# 9. Execution Functions
# ---------------------------------------------------------

def get_vkospi_data():
    # 타겟 URL
    url = "https://www.investing.com/indices/kospi-volatility"

    # 봇 탐지를 피하기 위한 헤더 설정 (일반 크롬 브라우저인 척 위장)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,ko;q=0.8",
        "Referer": "https://www.google.com/"
    }

    try:
        # 요청 보내기
        response = requests.get(url, headers=headers, timeout=50)

        # 응답 코드 확인 (200이 아니면 차단되었거나 에러 발생)
        if response.status_code != 200:
            print(f"Error: 페이지를 불러오지 못했습니다. 상태 코드: {response.status_code}")
            return None

        # HTML 파싱
        soup = BeautifulSoup(response.content, "html.parser")

        # 데이터 추출 (Investing.com의 최신 구조 반영: data-test 속성 사용)
        # 1. 현재 지수 가격
        price_tag = soup.find("div", {"data-test": "instrument-price-last"})
        current_price = price_tag.text.strip() if price_tag else "N/A"

        # 2. 전일 대비 변동폭
        change_point_tag = soup.find("span", {"data-test": "instrument-price-change"})
        change_point = change_point_tag.text.strip() if change_point_tag else "N/A"

        # 3. 전일 대비 변동률
        change_percent_tag = soup.find("span", {"data-test": "instrument-price-change-percent"})
        change_percent = change_percent_tag.text.strip() if change_percent_tag else "N/A"

        # 4. 장 상태 (장중/장마감 등)
        market_status = soup.find("div", {"class": "instrument-metadata_instrument-metadata__1kAkV"})
        # 클래스명은 자주 바뀌므로 data-test가 없는 경우 텍스트로 보조 추출 시도
        status_text = "N/A"
        if market_status:
            # 시간 정보가 포함된 하위 요소 찾기
            time_tag = market_status.find("time")
            if time_tag:
                status_text = time_tag.text.strip()

        # 결과 딕셔너리 생성
        data = {
            "지수명": "KOSPI 200 Volatility (VKOSPI)",
            "현재가": current_price,
            "변동폭": change_point,
            "변동률": change_percent,
            "기준시간": status_text
        }

        return data

    except Exception as e:
        print(f"크롤링 중 에러 발생: {e}")
        return None


def run_simulation(view_text, trend, risk_level, news_data, test_news,
                   news_sentiment="Neutral", is_price_rising=True, divergence_note="",
                   macro_pred: Optional[Dict[str, Any]] = None):
    kospi_realtime = get_kospi200_index()

    if kospi_realtime == 0:
        print("⚠️ KOSPI 200 지수를 가져오지 못해 시뮬레이션을 건너뜁니다.")
        return

    print("\nVKOSPI 데이터 조회 중...")
    result = get_vkospi_data()

    if result:
        print("-" * 30)
        for key, value in result.items():
            print(f"{key}: {value}")
        print("-" * 30)

        # MARKET_IV = float(result['현재가'])
        MARKET_IV = float(str(result['현재가']).replace(",", "").strip())

    else:
        print("데이터를 가져오는데 실패했습니다. (Investing.com의 보안 정책에 의해 차단되었을 수 있습니다.)")

        MARKET_IV = float(os.environ.get('MARKET_IV', 12.0))

    inputs: QuantState = {
        "kospi_index": kospi_realtime,
        "market_iv": MARKET_IV,
        "total_capital": TOTAL_CAPITAL,
        "manager_view": view_text,
        "raw_news_data": test_news,  # 하이브리드 검색을 위해 전달
        "risk_aversion": float(risk_level),
        "market_trend": str(trend),
        "days_to_expiry": float(days_to_expiry),
        "forex_news": news_data,

        # [NEW] Debate 재활용 필드
        "news_sentiment": news_sentiment,
        "is_price_rising": bool(is_price_rising),
        "divergence_note": divergence_note,

        "macro_pred": macro_pred or {},  # <--- NEW
    }

    try:
        # LangGraph 실행
        result = app.invoke(inputs)
        print(result['console_report'])
    except Exception as e:
        print(f"❌ Simulation Error: {e}")


# ---------------------------------------------------------
# DB 저장: MarketScenario 객체 + dict 모두 지원 (Debate 결과 저장)
# ---------------------------------------------------------
def insert_market_scenario(market_scenario: Any):
    if db_pool is None:
        print("⚠️ [System] DB Pool 없음: 시나리오 저장 스킵")
        return

    conn = None
    try:
        conn = db_pool.get_connection()
        cursor = conn.cursor()
        insert_query = """
                       INSERT INTO MarketScenario (title, summary_for_scenarios, risk_aversion_score, score_desc, trend,
                                                   risk, driver, key_factors, strategy)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
                       """

        if isinstance(market_scenario, dict):
            title = market_scenario.get("title", "Untitled")
            summary = market_scenario.get("summary_for_scenarios", market_scenario.get("summary", ""))
            score = float(market_scenario.get("risk_aversion_score", 5.0))
            score_desc = str(market_scenario.get("score_desc", "AI_DEBATE"))
            trend = str(market_scenario.get("trend", "neutral"))
            risk = str(market_scenario.get("risk", "AI_DEBATE"))
            driver = str(market_scenario.get("driver", "AI_DEBATE"))
            key_factors = market_scenario.get("key_factors", [])
            if isinstance(key_factors, list):
                sKeyFactors = "\n".join(f"- {x}" for x in key_factors)
            else:
                sKeyFactors = str(key_factors)
            strategy = str(market_scenario.get("strategy", "Dynamic Allocation"))
        else:
            # 기존 호환
            sKeyFactors = "\n".join(f"- {factor}" for factor in market_scenario.key_factors)
            title = market_scenario.title
            summary = market_scenario.summary
            score = float(market_scenario.risk_aversion_score)
            score_desc = market_scenario.score_desc.value
            trend = market_scenario.trend.value
            risk = market_scenario.risk.value
            driver = market_scenario.driver.value
            strategy = market_scenario.strategy

        cursor.execute(insert_query, (title, summary, score, score_desc, trend, risk, driver, sKeyFactors, strategy))
        conn.commit()
    except Exception as e:
        print(f"❌ DB 에러: {e}")
    finally:
        try:
            if conn:
                conn.close()
        except Exception:
            pass


# ---------------------------------------------------------
# [NEW] Debate 결과를 “첫 번째 코드 베이스”에 합치는 formatter
# ---------------------------------------------------------
def build_manager_view_from_debate(
        bull_opinion: str,
        bear_opinion: str,
        final_consensus: str,
        news_sentiment: str,
        divergence_note: str,
        is_price_rising: bool
) -> str:
    """
    첫 번째 코드의 quant_engine / divergence_checker가 읽기 좋게 manager_view를 구조화.

    핵심:
    - [Divergence] 섹션을 '명시적으로' 포함
    - "뉴스는 부정적이나 상승" 같은 문구를 divergence_note에 강제
    - price action 우선 규칙을 문장으로 박아 LLM이 mu/vol 반영하기 쉽게
    """
    price_action_text = "상승" if is_price_rising else "부진/하락"
    ns = news_sentiment.strip().capitalize()
    if ns not in ["Positive", "Negative", "Neutral"]:
        ns = "Neutral"

    if not divergence_note:
        if ns == "Negative" and is_price_rising:
            divergence_note = "뉴스는 부정적이나 가격은 상승(=Bullish Climber) → 추세는 상승 우선"
        elif ns == "Positive" and (not is_price_rising):
            divergence_note = "뉴스는 긍정적이나 가격 반응 미약 → 하락 전환 경계"
        else:
            divergence_note = "뉴스 심리와 가격 액션 간 뚜렷한 괴리 없음"

    view = ""
    view += "📌 [AI Debate Market View]\n"
    view += f"- News Sentiment: {ns}\n"
    view += f"- Price Action: {price_action_text}\n\n"

    view += "🟢 [Bull Case]\n"
    view += bull_opinion.strip() + "\n\n"

    view += "🔴 [Bear Case]\n"
    view += bear_opinion.strip() + "\n\n"

    view += "⚖️ [Judge Consensus]\n"
    view += final_consensus.strip() + "\n\n"

    # 중요: 만약 final_consensus 안에 이미 [Divergence]가 있다면
    # divergence_note를 또 붙이지 않습니다.
    if "[Divergence]" not in final_consensus:
        view += f"[Divergence]\n- {divergence_note}\n"
    view += "- NOTE: Price Action takes precedence over news sentiment.\n"

    return view


# ---------------------------------------------------------
# 1. 데이터 수집 (Data Fetching) - 기존과 동일
# ---------------------------------------------------------

def get_macro_data(api_key=None, start_date="2020-01-01"):
    if api_key is None:
        api_key = os.getenv("ALPHA_VANTAGE_KEY")

    base_url = "https://www.alphavantage.co/query"
    df_macro = pd.DataFrame()

    print(f"\n📡 [Alpha Vantage] 데이터 수집 시작... (Key: {api_key[:5] if api_key else 'N/A'}***)")

    economic_functions = {
        'FED_RATE': 'FEDERAL_FUNDS_RATE',
        'CPI': 'CPI',
        'UNEMPLOYMENT': 'UNEMPLOYMENT',
        'GDP': 'REAL_GDP',
        'RETAIL_SALES': 'RETAIL_SALES',
        'DURABLES': 'DURABLES'
    }

    for name, func in economic_functions.items():
        success = False
        retry_count = 0

        while not success and retry_count < 2:
            try:
                params = {"function": func, "apikey": api_key, "datatype": "json"}
                response = requests.get(base_url, params=params, timeout=15)
                data = response.json()

                if "Note" in data:
                    print(f"⚠️ [API Limit] {name}: 15초 대기 후 재시도...")
                    time.sleep(15)
                    retry_count += 1
                    continue

                if "data" in data and len(data["data"]) > 0:
                    temp_df = pd.DataFrame(data["data"])[['date', 'value']].copy()
                    temp_df['date'] = pd.to_datetime(temp_df['date'])
                    temp_df['value'] = pd.to_numeric(temp_df['value'], errors='coerce')
                    temp_df = temp_df.dropna(subset=['date'])
                    temp_df = temp_df.drop_duplicates(subset=['date'], keep='first')
                    temp_df = temp_df.sort_values('date').set_index('date')
                    temp_df = temp_df.rename(columns={'value': name})

                    if df_macro.empty:
                        df_macro = temp_df
                    else:
                        df_macro = df_macro.join(temp_df, how='outer')

                    print(f"✅ [Success] {name}: {len(temp_df)} rows 수집.")
                    success = True
                else:
                    print(f"❌ [Fail] {name}: 응답 데이터 없음. (스킵하고 계속 진행)")
                    break

            except Exception as e:
                print(f"⚠️ [Error] {name}: {e}")
                break

            time.sleep(12)

    for name in economic_functions.keys():
        if name not in df_macro.columns:
            df_macro[name] = np.nan

    tickers = {
        'KOSPI200': '^KS200',
        'S&P500': '^GSPC',
        'NASDAQ': '^IXIC',
        'SOX': '^SOX',
        'Nikkei225': '^N225',
        'USD_KRW': 'KRW=X',
        'US_10Y': '^TNX',
        'WTI_Oil': 'CL=F',
        'VIX': '^VIX'
    }

    print("📡 yfinance 데이터 수집 중...")
    for name, ticker in tickers.items():
        try:
            data = yf.download(ticker, start=start_date, progress=False, auto_adjust=False)

            if not data.empty:
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)

                val = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
                temp_yf = val.to_frame(name=name)
                temp_yf.index = pd.to_datetime(temp_yf.index)
                temp_yf = temp_yf[~temp_yf.index.duplicated(keep='first')]

                if df_macro.empty:
                    df_macro = temp_yf
                else:
                    df_macro = df_macro.join(temp_yf, how='outer')

                print(f"✅ [Success] {name} 수집 완료.")
        except Exception as e:
            print(f"⚠️ {name} 수집 오류: {e}")

    if df_macro.empty:
        return pd.DataFrame()

    df_macro = df_macro.sort_index()
    df_macro = df_macro[~df_macro.index.duplicated(keep='first')]

    # 월간/분기 지표를 거래일로 확장
    df_macro = df_macro.ffill()

    if 'KOSPI200' in df_macro.columns:
        kospi_idx = df_macro['KOSPI200'].dropna().index
        df_macro = df_macro.reindex(kospi_idx).ffill().bfill()

    print(f"🎯 [Data Sync] 최종 가용 데이터: {len(df_macro)} 거래일 확보")
    return df_macro


# 1. 컬럼 분류 (박사님의 설정을 그대로 유지)
price_cols = ['KOSPI200', 'S&P500', 'NASDAQ', 'SOX', 'Nikkei225', 'USD_KRW', 'WTI_Oil']
level_cols = ['US_10Y', 'VIX', 'FED_RATE', 'CPI', 'UNEMPLOYMENT', 'GDP', 'RETAIL_SALES', 'DURABLES']


# ---------------------------------------------------------
# 2. 발표 시차(Publication Lag) 반영 전처리 함수 (기존 유지)
# ---------------------------------------------------------
def process_data_with_sync(df):
    macro_lags = {
        'CPI': 20,
        'UNEMPLOYMENT': 10,
        'GDP': 45,
        'RETAIL_SALES': 15,
        'DURABLES': 20
    }

    processed_df = pd.DataFrame(index=df.index)

    # 1. 가격 지수 -> 로그 수익률
    price_indices = ['KOSPI200', 'S&P500', 'NASDAQ', 'SOX', 'Nikkei225', 'USD_KRW', 'WTI_Oil']
    for col in price_indices:
        if col in df.columns:
            processed_df[col] = np.log(df[col] / df[col].shift(1))

    # 2. 수준값 지표
    level_indices = ['VIX', 'US_10Y', 'FED_RATE']
    for col in level_indices:
        if col in df.columns:
            processed_df[col] = df[col]

    # 3. 거시 지표 -> Lag 반영 후 다시 ffill()
    for col, lag in macro_lags.items():
        if col in df.columns:
            processed_df[col] = df[col].shift(lag).ffill()

    # 4. 다음 거래일 KOSPI 로그수익률을 타깃으로 설정
    processed_df['TARGET_NEXT_KOSPI'] = processed_df['KOSPI200'].shift(-1)

    # 5. 피처 생성
    feature_df = pd.DataFrame(index=processed_df.index)

    # 해외 가격 변수는 T-1
    lagged_price_cols = ['S&P500', 'NASDAQ', 'SOX', 'USD_KRW', 'WTI_Oil']
    for col in lagged_price_cols:
        if col in processed_df.columns:
            feature_df[f"{col}_Lag1"] = processed_df[col].shift(1)

    # 수준값/거시값도 T-1
    lagged_level_cols = ['VIX', 'US_10Y', 'FED_RATE', 'CPI', 'UNEMPLOYMENT', 'GDP', 'RETAIL_SALES', 'DURABLES']
    for col in lagged_level_cols:
        if col in processed_df.columns:
            feature_df[f"{col}_Lag1"] = processed_df[col].shift(1)

    # 니케이 동조화 처리
    if 'Nikkei225' in processed_df.columns:
        feature_df['Nikkei225_Sync'] = processed_df['Nikkei225']

    # 6. 최종 병합 및 유실 방지
    model_data = pd.concat([processed_df[['TARGET_NEXT_KOSPI']], feature_df], axis=1).dropna()

    print(f"📉 [Pre-process] 최종 Ridge 학습 가용 데이터: {len(model_data)}행")

    feature_cols = feature_df.columns.tolist()
    return model_data, feature_cols


# ---------------------------------------------------------
# 2. 데이터 전처리 (Precise Lag Alignment) - [핵심 수정]
# ---------------------------------------------------------
def process_data(df):
    """
    KOSPI(T) ~ Macro_Variables(T-1) 관계를 형성하도록 데이터 구조화
    """
    # 1) 전체 데이터 로그 수익률 계산 (오늘/어제)
    #    이 시점에서는 모든 변수가 T 시점의 등락률임
    log_returns = np.log(df / df.shift(1))

    # 2) 타겟(Y)과 피처(X) 분리

    target_col = 'KOSPI200'

    # Y: KOSPI 200 (T 시점 그대로 유지)
    y = log_returns[[target_col]]

    # X: 나머지 변수들 (아직 T 시점)
    X = log_returns.drop(columns=[target_col])

    # 3) 피처(X)에만 시차 적용 (T -> T-1)
    #    shift(1)을 하면 '어제' 데이터가 '오늘' 행으로 내려옴
    #    즉, 같은 행(Row)에 [오늘 KOSPI]와 [어제 미국지수]가 위치하게 됨
    X_shifted = X.shift(1)

    # 컬럼명에 Lag 표시 추가 (혼동 방지)
    X_shifted.columns = [f"{col}_Lag1" for col in X_shifted.columns]

    # 4) 데이터 병합 (Concat)
    #    axis=1 (옆으로 붙이기)
    model_data = pd.concat([y, X_shifted], axis=1)

    # 5) 결측치 제거
    #    - 첫 번째 행: 수익률 계산으로 인한 NaN
    #    - 두 번째 행: shift(1)로 인한 NaN
    #    최소 2개 행이 삭제됨
    model_data = model_data.dropna()

    # 피처 컬럼 목록 리스트 (KOSPI200 제외한 나머지)
    feature_cols = X_shifted.columns.tolist()

    return model_data, feature_cols


# ---------------------------------------------------------
# 3. 상관관계 분석
# ---------------------------------------------------------
def analyze_correlation(model_data):
    plt.figure(figsize=(10, 8))
    corr_matrix = model_data.corr()

    print("\n[KOSPI 200(T)과 선행지표(T-1) 간 상관계수]")
    print(corr_matrix['KOSPI200'].sort_values(ascending=False))

    # 시각화
    #
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation: KOSPI(T) vs Variables(T-1)")
    plt.show()


# ---------------------------------------------------------
# 1. 지표별 발표 지연(Publication Lag) 반영 전처리
# ---------------------------------------------------------
def process_data_with_publication_lag(df):
    """
    Look-ahead Bias 제거 및 하이브리드 피처 생성:
    - 가격 데이터: 로그 수익률 (Log Return)
    - 금리/변동성: 수준값 (Level)
    - 거시 지표: 수준값 + 실제 발표 지연(Publication Lag) 반영
    """
    # [설정] 거시 지표별 평균적 발표 지연 일수 (보수적 기준)
    # 실제로는 공표일 캘린더를 연동하는 것이 좋으나, 고정 Lag로도 편향을 크게 줄임
    macro_lags = {
        'CPI': 20,  # 전월 데이터가 약 20일 뒤 발표
        'UNEMPLOYMENT': 10,  # 약 10일 뒤 발표
        'GDP': 50,  # 분기 종료 후 약 50일 뒤 (속보치)
        'RETAIL_SALES': 15,
        'DURABLES': 25
        # 'SENTIMENT': 1  # 심리지수는 보통 월말 발표되므로 짧게 설정
    }

    # 지표 분류
    price_indices = ['KOSPI200', 'S&P500', 'NASDAQ', 'SOX', 'Nikkei225', 'USD_KRW', 'WTI_Oil']
    realtime_levels = ['VIX', 'US_10Y', 'FED_RATE']

    processed_df = pd.DataFrame(index=df.index)

    # 1. 가격 지수 -> 로그 수익률
    for col in price_indices:
        if col in df.columns:
            processed_df[col] = np.log(df[col] / df[col].shift(1))

    # 2. 실시간 금리/변동성 -> 수준값(Level) 그대로
    for col in realtime_levels:
        if col in df.columns:
            processed_df[col] = df[col]

    # 3. [핵심] 거시 경제 지표 -> 발표 지연 반영 (Look-ahead Bias 제거)
    for col, lag in macro_lags.items():
        if col in df.columns:
            # 해당 날짜에 실제로 가용했던 '과거 값'을 매핑
            processed_df[col] = df[col].shift(lag)

    # 4. 타겟(KOSPI) 및 피처 정렬
    target_col = 'KOSPI200'
    y = processed_df[[target_col]]
    X_raw = processed_df.drop(columns=[target_col])

    # (A) Nikkei: 동시간대 동조화 (T 시점)
    X_sync = X_raw[['Nikkei225']].rename(columns={'Nikkei225': 'Nikkei225_Sync'})

    # (B) 나머지 지표들: 전일 가용 데이터 (T-1 시점)
    # 이미 macro_lags로 밀린 데이터들도 T-1 시점에 알고 있어야 하므로 한번 더 shift
    X_lagged = X_raw.drop(columns=['Nikkei225']).shift(1)
    X_lagged.columns = [f"{col}_Lag1" for col in X_lagged.columns]

    model_data = pd.concat([y, X_lagged, X_sync], axis=1).dropna()
    feature_cols = X_lagged.columns.tolist() + X_sync.columns.tolist()

    return model_data, feature_cols


# ---------------------------------------------------------
# 3. Ridge 분석 실행 함수 (Inference 부분 수정)
# ---------------------------------------------------------
def run_analysis_return(current_realtime_kospi, start_date="2020-01-01", alpha=1.0, train_ratio=0.85, show_corr=False):
    raw_df = get_macro_data(api_key=os.getenv("ALPHA_VANTAGE_KEY"), start_date=start_date)
    raw_df = raw_df[['KOSPI200', 'S&P500', 'NASDAQ', 'SOX', 'Nikkei225', 'USD_KRW', 'US_10Y',
                     'WTI_Oil', 'VIX']]

    if raw_df is None or raw_df.empty:
        raise ValueError("raw_df가 비어 있습니다.")

    model_data, feature_cols = process_data_with_sync(raw_df)

    if len(model_data) < 200:
        raise ValueError(f"학습 데이터가 너무 적습니다: {len(model_data)} rows")

    if show_corr:
        corr_data = model_data.rename(columns={'TARGET_NEXT_KOSPI': 'KOSPI200'})
        analyze_correlation(corr_data)

    # 마지막 row를 inference input으로 사용
    # 그 이전 row들로만 학습/검증
    trainable_data = model_data.iloc[:-1].copy()
    inference_row = model_data.iloc[[-1]].copy()

    if len(trainable_data) < 200:
        raise ValueError(f"학습용 데이터가 너무 적습니다: {len(trainable_data)} rows")

    X_all = trainable_data[feature_cols]
    y_all = trainable_data['TARGET_NEXT_KOSPI']

    split = int(len(trainable_data) * train_ratio)
    X_train, X_test = X_all.iloc[:split], X_all.iloc[split:]
    y_train, y_test = y_all.iloc[:split], y_all.iloc[split:]

    if len(X_test) == 0:
        raise ValueError("테스트 데이터가 비어 있습니다. train_ratio를 조정하세요.")

    model_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=alpha))
    ])

    model_pipeline.fit(X_train, y_train)

    preds = model_pipeline.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    acc = float(np.mean(np.sign(preds) == np.sign(y_test.values)))

    # 마지막 feature row를 그대로 추론 입력으로 사용
    input_df = inference_row[feature_cols]
    pred_log_ret = float(model_pipeline.predict(input_df)[0])

    pred_pct = float((np.exp(pred_log_ret) - 1.0) * 100.0)
    next_kospi = float(current_realtime_kospi * np.exp(pred_log_ret))

    return {
        "pred_log_ret": pred_log_ret,
        "pred_pct": pred_pct,
        "current_kospi": float(current_realtime_kospi),
        "next_kospi": next_kospi,
        "rmse": rmse,
        "directional_acc": acc,
        "feature_date": str(inference_row.index[0].date()) if hasattr(inference_row.index[0], "date") else str(
            inference_row.index[0]),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_total_model_data": int(len(model_data)),
    }


# ---------------------------------------------------------
# [NEW] Multi-Agent Debate Engine (Bull/Bear/Judge)
#   합의문에 Divergence 문구를 '명시적으로' 포함시켜
#   DivergenceChecker가 manager_view에서 근거를 더 잘 읽도록 구성
# ---------------------------------------------------------
def bull_agent_node(state: DebateState):
    retry_feedback = ""
    if state.get("retry_count", 0) > 0 and state.get("eval_critique"):
        retry_feedback = f"\n[이전 피드백]\n{state.get('eval_critique', '')}\n"

    prompt = f"""
        You are a Bullish Market Strategist.

        Task:
        - Read the news context carefully.
        - Build a bullish case ONLY from facts explicitly contained in the news context.
        - Do NOT invent events, prices, yields, wars, oil levels, policy actions, or technical indicators not mentioned in the news.
        - If bullish evidence is weak, say so clearly.
        - Output in Korean.
        - Keep it short.
        {retry_feedback}

        Required format:
        [핵심 상승 논거]
        - ...
        - ...
        [한계]
        - ...

        News context:
        {state.get("news_context", "")}
        """.strip()

    response_text = safe_llm_text([HumanMessage(content=prompt)], fallback="상승 논거 생성 실패")
    return {"bull_opinion": response_text}



def bear_agent_node(state: DebateState):
    retry_feedback = ""
    if state.get("retry_count", 0) > 0 and state.get("eval_critique"):
        retry_feedback = f"\n[이전 피드백]\n{state.get('eval_critique', '')}\n"

    prompt = f"""
        You are a Bearish Risk Analyst.

        Task:
        - Read the news context carefully.
        - Build a bearish case ONLY from facts explicitly contained in the news context.
        - Do NOT invent events, prices, yields, wars, oil levels, policy actions, or technical indicators not mentioned in the news.
        - If bearish evidence is weak, say so clearly.
        - Output in Korean.
        - Keep it short.
        {retry_feedback}

        Required format:
        [핵심 하락 논거]
        - ...
        - ...
        [한계]
        - ...

        News context:
        {state.get("news_context", "")}
        """.strip()

    response_text = safe_llm_text([HumanMessage(content=prompt)], fallback="하락 논거 생성 실패")
    return {"bear_opinion": response_text}


def consensus_judge_node(state: DebateState):
    print("⚖️ [Judge] 심판 노드 시작")

    bull_text = (state.get("bull_opinion", "") or "")[:1200]
    bear_text = (state.get("bear_opinion", "") or "")[:1200]

    example_json = """
    {
      "final_consensus": "한국어 결론",
      "market_trend": "Neutral",
      "risk_score": 5.5,
      "news_sentiment": "Neutral",
      "divergence_note": "한국어 괴리 설명"
    }
    """.strip()

    prompt = (
        "You are the Chief Investment Officer for the OptiQ systematic trading system.\n"
        "Use ONLY the evidence contained in the arguments below.\n"
        "Do NOT introduce any new macro facts, events, price levels, yields, wars, oil prices, or technical indicators.\n\n"
        f"[Bullish Argument]\n{bull_text}\n\n"
        f"[Bearish Argument]\n{bear_text}\n\n"
        "Return ONLY valid JSON with keys:\n"
        "final_consensus, market_trend, risk_score, news_sentiment, divergence_note\n\n"
        f"Example:\n{example_json}\n\n"
        "Rules:\n"
        "- market_trend must be one of: Bullish, Bearish, Volatile, Neutral\n"
        "- news_sentiment must be one of: Positive, Negative, Neutral\n"
        "- risk_score must be a number between 1 and 10\n"
        "- final_consensus must be in Korean\n"
        "- divergence_note must be in Korean\n"
        "- no markdown\n"
        "- no explanation\n"
        "- JSON only"
    )

    try:
        print("⚖️ [Judge] LLM 호출 직전")
        raw = safe_llm_text([HumanMessage(content=prompt)], fallback="")
        print(f"🔍 [Judge RAW]\n{raw}")

        data = extract_json_block(raw)
        if not data:
            raise ValueError("Judge JSON 파싱 실패")

        final_consensus = str(
            data.get("final_consensus", "데이터 분석 오류로 인한 보수적 관망 유지.")
        ).strip()

        market_trend = str(data.get("market_trend", "Neutral")).strip().capitalize()

        try:
            risk_score = float(data.get("risk_score", 5.5))
        except Exception:
            risk_score = 5.5

        news_sentiment = normalize_sentiment_label(data.get("news_sentiment", "Neutral"))
        divergence_note = str(
            data.get("divergence_note", "LLM 응답 파싱 실패로 인한 폴백 데이터 생성")
        ).strip()

        clean_consensus = final_consensus.split("[Divergence]")[0].strip()
        clean_note = divergence_note.replace("[Divergence]", "").strip()
        final_text = f"{clean_consensus}\n\n[Divergence]\n{clean_note}"

        return {
            "final_consensus": final_text,
            "market_trend": market_trend,
            "risk_score": risk_score,
            "news_sentiment": news_sentiment,
            "divergence_note": clean_note,
        }

    except Exception as e:
        print(f"❌ [Critical] Judge Generation Failed: {e}")
        return {
            "final_consensus": "데이터 분석 오류로 인한 보수적 관망 유지.\n\n[Divergence]\n판단 근거 부족",
            "market_trend": "Neutral",
            "risk_score": 5.5,
            "news_sentiment": "Neutral",
            "divergence_note": "판단 근거 부족"
        }

def evaluator_node(state: DebateState):
    prompt = f"""
        당신은 금융 리포트 감사관입니다.
        아래 [원본 뉴스]와 [최종 합의문]을 비교해서 품질을 평가하세요.

        반드시 JSON만 반환하세요.
        키는 score, is_sufficient, critique 입니다.

        조건:
        - score: 1~10 정수
        - is_sufficient: true 또는 false
        - critique: 한국어
        - JSON only
        - no markdown

        [원본 뉴스]
        {state.get("news_context", "")}

        [최종 합의문]
        {state.get("final_consensus", "")}

        예시:
        {{
          "score": 8,
          "is_sufficient": true,
          "critique": "핵심 뉴스 반영이 충분하며 추가 보완이 크지 않습니다."
        }}
        """.strip()

    retry_cnt = state.get("retry_count", 0)

    try:
        raw = safe_llm_text([HumanMessage(content=prompt)], fallback="")
        data = extract_json_block(raw)
        if not data:
            raise ValueError("Evaluator JSON 파싱 실패")

        score = int(float(data.get("score", 6)))

        raw_sufficient = data.get("is_sufficient", False)
        if isinstance(raw_sufficient, bool):
            is_sufficient = raw_sufficient
        else:
            is_sufficient = str(raw_sufficient).strip().lower() == "true"

        critique = str(data.get("critique", "보완 의견 없음")).strip()

    except Exception as e:
        print(f"⚠️ [Evaluator] Error: {e}")
        score = 6
        is_sufficient = True if retry_cnt >= 2 else False
        critique = "평가기 응답 파싱 실패"

    if retry_cnt >= 2:
        is_sufficient = True

    result = {
        "eval_score": score,
        "is_sufficient": is_sufficient,
        "eval_critique": critique,
        "retry_count": retry_cnt + 1
    }

    print("\n[DEBUG] evaluator result:")
    print(result)

    return result


def decide_refinement(state: DebateState):
    if state.get("is_sufficient"):
        return "approved"

    print(f"🔄 [Self-Improvement] 품질 미달(점수:{state.get('eval_score')}). 재토론 진입...")
    return "refine"

def _normalize_trend(trend: str) -> str:
    t = str(trend).strip().lower()
    if t in ["bullish", "bearish", "volatile", "neutral", "reversal"]:
        return t

    if "bull" in t:
        return "bullish"
    if "bear" in t:
        return "bearish"
    if "vol" in t:
        return "volatile"
    return "neutral"


debate_workflow = StateGraph(DebateState)
debate_workflow.add_node("Bull", bull_agent_node)
debate_workflow.add_node("Bear", bear_agent_node)
debate_workflow.add_node("Judge", consensus_judge_node)
debate_workflow.add_node("Evaluate", evaluator_node)

debate_workflow.set_entry_point("Bull")
debate_workflow.add_edge("Bull", "Bear")
debate_workflow.add_edge("Bear", "Judge")
debate_workflow.add_edge("Judge", "Evaluate")
debate_workflow.add_conditional_edges(
    "Evaluate",
    decide_refinement,
    {
        "refine": "Bull",
        "approved": END
    }
)

debate_app = debate_workflow.compile()


# ---------------------------------------------------------
# 6. 메인 실행 Job (하이브리드 데이터 준비)
# ---------------------------------------------------------

def job():
    print(f"\n⏰ [Scheduler] RAG 기반 자동 매매 시스템 가동: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # 1. DB 및 뉴스 데이터 가져오기
        test_news = fetch_latest_news()

        # 2. Forex Factory 뉴스 크롤링
        news_data = get_forex_news()
        if not news_data:
            news_data = "뉴스 데이터를 가져올 수 없습니다."

        # 3. [NEW] Multi-Agent Debate Engine 실행
        if test_news:
            print("\n🤖 [Debate Engine] 상승/하락 토론 시작...")

            news_context = "\n".join(
                [f"- {row.get('date', '')} {row.get('time', '')} | {row.get('title', '')}"
                 for row in test_news]
            )

            print("\n📰 [DEBUG news_context]")
            print(news_context[:3000])

            # debate_inputs: DebateState = {"news_context": news_context}
            debate_inputs: DebateState = {
                "news_context": news_context,
                "retry_count": 0,
                "eval_critique": ""
            }
            debate_result = debate_app.invoke(debate_inputs)
            print("\n🧪 [Debate Evaluation]")
            print(f" - eval_score     : {debate_result.get('eval_score')}")
            print(f" - is_sufficient  : {debate_result.get('is_sufficient')}")
            print(f" - eval_critique  : {debate_result.get('eval_critique')}")
            print(f" - retry_count    : {debate_result.get('retry_count')}")

            bull_op = (debate_result.get("bull_opinion") or "").strip()
            bear_op = (debate_result.get("bear_opinion") or "").strip()
            final_consensus = (debate_result.get("final_consensus") or "").strip()

            market_trend_str = str(debate_result.get("market_trend", "Neutral")).strip()
            normalized_trend = _normalize_trend(market_trend_str)

            risk_score = float(debate_result.get("risk_score", 5.0))
            # news_sentiment = str(debate_result.get("news_sentiment", "Neutral")).strip().capitalize()
            news_sentiment = normalize_sentiment_label(debate_result.get("news_sentiment", "Neutral"))
            divergence_note = str(debate_result.get("divergence_note", "")).strip()

            # 가격 액션(외부 연동 전): 기존 코드 컨셉 유지
            # is_price_rising = False
            kospi_realtime = get_kospi200_index()  # LS API 호출

            # (A) 매크로 기반 "다음 거래일" 방향 예측 -> is_price_rising 생성
            try:
                macro_pred = run_analysis_return(current_realtime_kospi=kospi_realtime, start_date="2020-01-01",
                                                 alpha=1.0, train_ratio=0.85, show_corr=False)

                # 기준: 예측 로그수익률이 0보다 크면 상승(True)
                threshold = 0.0001  # 로그수익률 기준이 아니라면 조정 필요
                is_price_rising = (macro_pred["pred_log_ret"] > 0)

                print("\n📈 [Macro Predictor]")
                print(f" - Predicted Return: {macro_pred['pred_pct']:+.2f}%")
                print(f" - Current KOSPI200: {macro_pred['current_kospi']:.2f}")
                print(f" - Next KOSPI200   : {macro_pred['next_kospi']:.2f}")
                print(f" - is_price_rising : {is_price_rising}")
                print(f" - (RMSE {macro_pred['rmse']:.5f}, DirAcc {macro_pred['directional_acc'] * 100:.2f}%)\n")

            except Exception as e:
                # 예측 실패 시 기존처럼 보수적으로 처리(원하면 True/False 정책 바꾸세요)
                print(f"⚠️ [Macro Predictor] Failed: {e}\n")
                is_price_rising = True  # 또는 False / 또는 최근 KOSPI 변화로 대체
                macro_pred = {}  # <--- NEW

            print(
                f"⚖️ [Judge Decision] Trend: {market_trend_str} -> {normalized_trend} | Risk Score: {risk_score:.2f}")
            print(f"📝 [Consensus] {final_consensus}")

            view_text = build_manager_view_from_debate(
                bull_opinion=bull_op,
                bear_opinion=bear_op,
                final_consensus=final_consensus,
                news_sentiment=news_sentiment,
                divergence_note=divergence_note,
                is_price_rising=is_price_rising
            )

            global scenario_title

            div_tag = "Divergence" if ("부정적" in divergence_note and "상승" in divergence_note) or (
                    "긍정적" in divergence_note and "미약" in divergence_note) else "Aligned"
            scenario_title = f"AI Debate: {normalized_trend.upper()} | Risk {risk_score:.1f} | {news_sentiment} | {div_tag}"

            mock_scenario = {
                "title": scenario_title,
                "summary_for_scenarios": final_consensus,
                "risk_aversion_score": risk_score,
                "score_desc": "AI_DEBATE",
                "trend": normalized_trend,
                "risk": "AI_DEBATE",
                "driver": "AI_DEBATE",
                "key_factors": [
                    f"News Sentiment: {news_sentiment}",
                    f"Price Action: {'Rising' if is_price_rising else 'Weak'}",
                    f"Divergence: {divergence_note}" if divergence_note else "Divergence: None",
                ],
                "strategy": "Dynamic Allocation"
            }
            insert_market_scenario(mock_scenario)

            dynamic_risk = compute_dynamic_risk_score(
                llm_score=risk_score,
                macro_pred=macro_pred,
                news_sentiment=news_sentiment,
                is_price_rising=is_price_rising,
                market_trend=normalized_trend,
                divergence_note=divergence_note,
            )

            run_simulation(
                view_text=view_text,
                trend=normalized_trend,
                risk_level=dynamic_risk,
                news_data=news_data,
                test_news=test_news,
                news_sentiment=news_sentiment,
                is_price_rising=is_price_rising,
                divergence_note=divergence_note,
                macro_pred=macro_pred
            )

        else:
            print("⚠️ 분석할 뉴스 데이터가 없습니다.")

    except Exception as e:
        print(f"❌ [Scheduler] 작업 실행 중 치명적 오류 발생: {e}")

    print(f"✅ [Scheduler] 작업 종료. 다음 실행 대기 중...\n")


if __name__ == "__main__":
    # 전역 변수를 수정하기 위해 global 키워드 명시
    # global scenario_manager

    print("🚀 [System] 시나리오 지식 베이스 초기화 중...")

    scenario_manager = None
    try:
        # 1. DB 시나리오 매니져 초기화
        scenario_manager = ScenarioManager(db_pool, embeddings)
        # 2. 시나리오 데이터 로드 및 벡터화
        scenario_manager.load_and_index_scenarios()
    except Exception as e:
        print(f"⚠️ [Scenario] 초기화 실패: {e}")
        scenario_manager = None

    job()

    schedule.every(0.5).hours.do(job)

    print("🚀 [System] 자동 매매 봇이 시작되었습니다. (0.5시간 간격 실행)")

    while True:
        try:
            schedule.run_pending()
            time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 [System] 사용자에 의해 프로그램이 종료되었습니다.")
            break
        except Exception as e:
            print(f"⚠️ [System] 스케줄러 루프 오류: {e}")
            time.sleep(60)
