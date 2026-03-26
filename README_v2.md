# LLMTradEx34OptMiniFOV17 기술 보고서

**작성일:** 2026년 03월 26일
**파일명:** LLMTradEx34OptMiniFOV17.py (3,873 lines)
**버전:** Weekly Options Optimizer — Evaluator/Refinement Loop 포함

---

## 목차

1. [시스템 개요](#1-시스템-개요)
2. [하드웨어 및 소프트웨어 스택](#2-하드웨어-및-소프트웨어-스택)
3. [전체 아키텍처](#3-전체-아키텍처)
4. [핵심 최적화 모듈 (상세)](#4-핵심-최적화-모듈)
5. [매크로 예측기 Ridge Regression](#5-매크로-예측기-ridge-regression)
6. [Multi-Agent Debate Engine](#6-multi-agent-debate-engine)
7. [학습 및 피드백 시스템](#7-학습-및-피드백-시스템)
8. [Key References](#8-key-references)

---

## 1. 시스템 개요

LLMTradEx34OptMiniFOV17은 **KOSPI200 주간 옵션(Weekly Options)** 과 **미니 선물(Mini KOSPI200 Futures)** 을 대상으로 하는 LLM 기반 완전 자동화 퀀트 트레이딩 시스템이다.

핵심 설계 철학:

- **LLM-in-the-Loop:** 로컬 LLM(Qwen3.5-35B)이 시장 상황을 해석하여 수익률 벡터(μ), 변동성 벡터(σ), 상관관계 행렬(Σ)을 생성한다.
- **Multi-Objective Optimization:** 수익, 위험, 델타, 베가, 세타, 감마를 동시 고려하는 통합 비용 함수를 SLSQP로 최적화한다.
- **Self-Improvement Loop:** Evaluator 노드가 LLM 합의문의 품질을 평가하고 기준 미달 시 재토론을 트리거한다.
- **RAG(Retrieval-Augmented Generation):** 과거 시나리오 및 뉴스를 FAISS 벡터 DB에 저장하여 현재 시장과 가장 유사한 앵커(Anchor)를 검색한다.

대상 자산 구성 (5종):

| 인덱스 | 자산명 | 유형 | 역할 |
|--------|--------|------|------|
| 0 | Call Strategy (Long) | Deep OTM Call | 상승 방향성 / 볼록성 |
| 1 | Call Hedge (Short) | OTM Call | 헤지 / 프리미엄 수취 |
| 2 | Put Strategy (Long) | Deep OTM Put | 하락 방향성 / 볼록성 |
| 3 | Put Hedge (Short) | OTM Put | 헤지 / 프리미엄 수취 |
| 4 | Mini Future (Delta) | 선물 | 델타 헤징 (선형) |

---

## 2. 하드웨어 및 소프트웨어 스택

### 2.1 하드웨어

| 항목 | 사양 |
|------|------|
| 시스템 | Alienware Area-51 |
| CPU | Intel Core Ultra 9 285K (24코어, P+E 하이브리드) |
| GPU | NVIDIA RTX 5090 32GB VRAM |
| RAM | 64GB DDR5 |
| OS | Windows 11 Pro (Build 26200) |
| CUDA | 13.0 / Driver 581.95 |

### 2.2 LLM 서버

```
llama-server --model Qwen3.5-35B-A3B-Q6_K.gguf -np 4 -cb --port 8080
```

- **모델:** Qwen3.5-35B-A3B-Q6_K (MoE, 활성 파라미터 ~3.5B)
- **병렬 슬롯:** `-np 4 -cb` → 4개 동시 요청 처리
- **API:** OpenAI 호환 (http://localhost:8080/v1)
- **Thinking 모드 비활성화:** `/no_think` 시스템 메시지로 `<think>` 블록 억제

### 2.3 주요 Python 라이브러리

| 라이브러리 | 버전/용도 |
|-----------|----------|
| LangGraph | StateGraph 기반 멀티-에이전트 워크플로우 |
| LangChain | LLM 체인, RAG, FAISS 벡터스토어 |
| scipy | SLSQP 최적화 (`scipy.optimize.minimize`) |
| scikit-learn | Ridge Regression, StandardScaler, Pipeline |
| numpy / pandas | 수치 연산, 데이터프레임 |
| yfinance | 해외 지수 / 매크로 데이터 수집 |
| MySQL Connector | 뉴스 DB, 시나리오 DB 연결 |
| statsmodels | 통계 분석 보조 |

### 2.4 데이터 소스

| 소스 | 내용 |
|------|------|
| LS Securities REST API | 옵션 시장가, Greeks (실시간) |
| Alpha Vantage API | FED금리, CPI, GDP, 실업률, 소매판매 |
| yfinance | KOSPI200, S&P500, NASDAQ, SOX, Nikkei225, USD/KRW, VIX, WTI |
| 네이버 금융 | CD 91일물 금리 (무위험이자율) |
| Forex Factory | 주요 경제지표 캘린더 (cloudscraper) |
| MySQL DB | news_data, MarketScenario, macro_scenarios 테이블 |

---

## 3. 전체 아키텍처

### 3.1 두 개의 LangGraph 워크플로우

시스템은 **두 개의 독립적인 LangGraph StateGraph** 로 구성된다.

#### [A] Debate Workflow (시장 뷰 생성)

```
Bull → Bear → Judge → Evaluate
                         ↓         ↑
                      approved   refine (점수 미달 시 Bull로 복귀)
                         ↓
                        END
```

상태 타입: `DebateState`

- **Bull 에이전트:** 뉴스 컨텍스트에서 상승 논거만 추출 (뉴스에 없는 사실 생성 금지)
- **Bear 에이전트:** 동일 컨텍스트에서 하락 논거 추출
- **Judge (CIO):** 두 논거를 종합, JSON 출력 (`market_trend`, `risk_score`, `divergence_note`)
- **Evaluator:** 합의문 품질 평가 (1~10점), 점수 미달 → 재토론 (최대 2회)

#### [B] Trading Workflow (포트폴리오 최적화 실행)

```
RAG_Search → DivergenceChecker → Engine → MarketData
→ FuturesStrategy → Optimizer → Reporter → Notifier → Learning → END
```

상태 타입: `QuantState`

| 노드 | 역할 |
|------|------|
| RAG_Search | FAISS 하이브리드 검색 (벡터 + 키워드) |
| DivergenceChecker | 매크로 예측 vs 뉴스 심리 괴리 검증, Trend 확정 |
| Engine (quant_engine) | LLM으로 μ, σ, Σ 추정 |
| MarketData | LS API에서 옵션 시장가/Greeks 수집 |
| FuturesStrategy | 미니 선물 시그널 생성 |
| Optimizer | SLSQP 다목적 최적화 |
| Reporter | 포지션 수량 산출, 증거금 계산, HTML 리포트 생성 |
| Notifier | 이메일 발송 |
| Learning | 우수 파라미터 DB 피드백 저장 |

### 3.2 RAG 엔진 (FinancialRAGEngine)

```
[뉴스 DB] → RecursiveCharacterTextSplitter(chunk=400, overlap=50)
         → Parent Document Retriever (Parent/Child 분리)
         → FAISS VectorStore (OpenAI Embeddings 호환, port 8081)
         → hybrid_search(벡터 유사도 + 키워드 매칭)
```

- **Parent Document Retrieval:** 검색은 작은 청크(Child)로, 문맥 반환은 큰 청크(Parent)로 분리하여 검색 정밀도와 문맥 완전성을 동시에 확보
- **Hybrid Search:** 벡터 검색 결과에 키워드 기반 필터링을 결합

---

## 4. 핵심 최적화 모듈

### 4.1 Black-Scholes 옵션 가격 결정 모델

#### 4.1.1 기본 수식

Black-Scholes 모델은 무위험 이자율 r, 기초자산 가격 S, 행사가 K, 잔존시간 T(년), 변동성 σ 를 입력으로 옵션 이론가와 Greeks를 산출한다.

**d1, d2 계산:**

$$d_1 = \\frac{{\\ln(S/K) + (r + \\frac{{1}}{{2}}\\sigma^2)T}}{{\\sigma\\sqrt{{T}}}}$$

$$d_2 = d_1 - \\sigma\\sqrt{{T}}$$

**콜옵션(Call) 가격:**

$$C = S \\cdot N(d_1) - K e^{{-rT}} \\cdot N(d_2)$$

**풋옵션(Put) 가격:**

$$P = K e^{{-rT}} \\cdot N(-d_2) - S \\cdot N(-d_1)$$

여기서 $N(\\cdot)$은 표준 정규분포의 누적분포함수(CDF)이다.

#### 4.1.2 만기 처리

잔존시간 $T \\le 10^{{-5}}$ 이면 내재가치(Intrinsic Value)만 반환:

$$C_{{intrinsic}} = \\max(0, S-K), \\quad P_{{intrinsic}} = \\max(0, K-S)$$

#### 4.1.3 구현 특이사항

- **잔존시간(T) 계산:** 캘린더 기준 일수 + 당일 장중 비율(Intraday Fraction)을 합산한 `days_to_expiry`를 252로 나누어 연환산
- **Intraday Fraction:** 장 시작(08:45) ~ 종료(15:45) 기준 잔여 비율: $f_{{intraday}} = (t_{{close}} - t_{{now}}) / (t_{{close}} - t_{{open}})$
- **무위험이자율:** 네이버 금융에서 CD 91일물 금리를 크롤링하여 적용

---

### 4.2 변동성 스큐 모델 (Volatility Skew)

실제 옵션 시장에서 내재변동성(IV)은 행사가에 따라 달라지는 **스큐(Skew) 및 스마일(Smile)** 구조를 보인다. 주가지수 옵션은 특히 OTM Put(낮은 행사가)의 IV가 높은 **Put Skew** 특성을 가진다.

#### 4.2.1 머니니스 기반 2차 스큐 모델

$$m = \\ln\\left(\\frac{{K}}{{S}}\\right) \\quad \\text{{(Log-Moneyness)}}$$

$$\\sigma_{{adjusted}}(K) = \\sigma_{{ATM}} \\cdot \\left(1 - 0.15 \\cdot m + 0.5 \\cdot m^2\\right)$$

- $-0.15 \\cdot m$ 항: Put Skew — 낮은 행사가(음의 m)일수록 IV 상승
- $+0.5 \\cdot m^2$ 항: Volatility Smile — OTM 양방향으로 IV 상승
- 최소 IV 보정: $\\sigma_{{adjusted}} = \\max(0.01, \\sigma_{{adjusted}})$

#### 4.2.2 스큐 인식 Black-Scholes (`calculate_bs_skew_aware`)

```
1. 행사가 K에 대한 스큐 조정 IV 산출: σ(K) = get_iv_curve(σ_ATM, K, S)
2. 조정된 σ(K)로 표준 BS 계산 수행: calculate_bs_all(S, K, T, r, σ(K))
```

실제 시장 데이터(API)가 없거나 검증 실패 시 폴백으로 사용된다.

---

### 4.3 그릭스(Greeks) 계산

#### 4.3.1 델타 (Delta, Δ)

방향성 위험: 기초자산 가격 $S$의 단위 변화에 대한 옵션 가격 민감도

$$\\Delta_{{call}} = N(d_1), \\quad \\Delta_{{put}} = N(d_1) - 1$$

범위: Call $\\Delta \\in [0, 1]$, Put $\\Delta \\in [-1, 0]$

#### 4.3.2 감마 (Gamma, Γ)

볼록성 위험: 델타의 변화율 (S에 대한 2차 민감도)

$$\\Gamma = \\frac{{N'(d_1)}}{{S \\cdot \\sigma \\cdot \\sqrt{{T}}}} = \\frac{{e^{{-d_1^2/2}}}}{{S \\cdot \\sigma \\cdot \\sqrt{{2\\pi T}}}}$$

- Long position: $\\Gamma > 0$ (볼록성 이익)
- Short position: $\\Gamma < 0$ (볼록성 손실 — 급격한 가격 이동 시 손실 확대)

#### 4.3.3 베가 (Vega, ν)

변동성 위험: IV 1% 변화에 대한 옵션 가격 민감도

$$\\nu = S \\cdot \\sqrt{{T}} \\cdot N'(d_1) / 100 = \\frac{{S \\cdot \\sqrt{{T}} \\cdot e^{{-d_1^2/2}}}}{{100\\sqrt{{2\\pi}}}}$$

(100으로 나눔: 1% 단위 환산)

#### 4.3.4 세타 (Theta, Θ)

시간 감소 효과: 시간 경과에 따른 옵션 가치 손실 (1일 기준)

**콜옵션:**
$$\\Theta_{{call}} = -\\frac{{S \\sigma N'(d_1)}}{{2\\sqrt{{2\\pi T}}}} - r K e^{{-rT}} N(d_2)$$

**풋옵션:**
$$\\Theta_{{put}} = -\\frac{{S \\sigma N'(d_1)}}{{2\\sqrt{{2\\pi T}}}} + r K e^{{-rT}} N(-d_2)$$

일일 세타: $\\Theta_{{daily}} = \\Theta / 365$

**포트폴리오 Greeks:**

$$\\Delta_{{port}} = \\sum_{{i}} w_i \\Delta_i, \\quad \\Gamma_{{port}} = \\sum_{{i}} w_i \\Gamma_i$$
$$\\nu_{{port}} = \\sum_{{i}} w_i \\nu_i, \\quad \\Theta_{{port}} = \\sum_{{i}} w_i \\Theta_i$$

---

### 4.4 행사가 결정 알고리즘 (Strike Selection)

ATM 가격 $S_{{ATM}}$, 리스크 회피도 $\\lambda$, IV $\\sigma$, 시장 방향 `trend`를 입력으로 각 포지션의 행사가를 결정한다.

**기본 공식 (Z-Score 방식):**

$$K_i = S_{{ATM}} \\cdot \\exp\\left(z_i \\cdot \\sigma \\cdot \\sqrt{{\\frac{{T}}{{252}}}}\\right)$$

여기서 $z_i$는 방향별로 미리 정의된 Z-Score 오프셋이다:

| 시장 방향 | Call Long $z_0$ | Call Short $z_1$ | Put Long $z_2$ | Put Short $z_3$ |
|----------|---------------|----------------|--------------|---------------|
| Bullish  | -0.05 (Near ATM) | +0.40 (OTM) | -0.80 (Deep OTM) | -0.60 |
| Bearish  | +0.80 (Deep OTM) | +0.60 | +0.05 (Near ATM) | -0.40 (OTM) |
| Volatile | -0.30 | +0.50 | -0.30 | +0.50 |
| Neutral  | +0.40 | +0.80 | -0.40 | -0.80 |

설계 의도: **Long 포지션을 ATM에 근접 배치**하여 작은 가격 이동에도 수익이 발생하도록 하고, Short 포지션은 충분히 벌려(Spread) 손익비를 개선한다.

행사가는 KRX 틱 단위(2.5pt)로 반올림되며, 안전 장치(Safety Limits)로 상한/하한이 적용된다.

---

### 4.5 동적 리스크 타겟 (Dynamic Risk Targets)

만기까지 잔존일수(DTE)와 시장 방향에 따라 목표 Greek 값을 동적으로 결정한다.

#### 4.5.1 베가 스케일 팩터

$$s_{{vega}} = \\sqrt{{\\frac{{DTE}}{{5}}}}$$

DTE가 작을수록 베가 노출 목표를 줄여 만기 직전 변동성 리스크를 축소한다.

#### 4.5.2 감마 허용 한도

$$\\Gamma_{{limit}} = 0.05 \\cdot \\frac{{DTE}}{{4}}$$

DTE가 작을수록 허용 감마가 줄어들어 만기 핀 리스크를 제한한다.

#### 4.5.3 방향별 목표 델타

| 시장 방향 | 목표 델타 ($\\Delta_{{target}}$) | 목표 베가 |
|---------|-------------------------|---------|
| Bullish  | +2.35 | $0.05 \\cdot s_{{vega}}$ |
| Bearish  | -3.50 | $0.15 \\cdot s_{{vega}}$ |
| Volatile | -0.10 | $0.40 \\cdot s_{{vega}}$ (최대 변동성 노출) |
| Neutral  | +0.15 | $-0.10 \\cdot s_{{vega}}$ |

---

### 4.6 만기 효과 가중치 (Expiration Effects)

#### 4.6.1 세타 가속 계수

만기가 가까울수록 세타 감소가 가속되는 현상을 반영:

$$w_{{\\theta}} = \\frac{{1}}{{\\sqrt{{DTE}}}}$$

DTE=4일이면 $w_\\theta = 0.5$, DTE=1일이면 $w_\\theta = 1.0$, DTE=0.25일이면 $w_\\theta = 2.0$

#### 4.6.2 감마 리스크 가중치

$$w_{{\\Gamma}} = \\begin{{cases}} 1/DTE^2 & \\text{{if }} DTE < 1 \\\\ 1/DTE & \\text{{otherwise}} \\end{{cases}}$$

만기 당일(DTE<1)에는 감마 리스크가 급격히 증가하므로 2차 함수로 패널티를 강화한다.

#### 4.6.3 실행 모드 분류

| DTE 범위 | 모드 | 설명 |
|---------|------|------|
| ≤ 0.5일 | `EXPIRATION_SCALPING` | 집중 패널티, 포지션 최소화 |
| ≤ 1.5일 | `THETA_ACCELERATION` | 세타 극대화 전략 |
| > 1.5일 | `TREND_FOLLOWING` | 방향성 추종 전략 |

---

### 4.7 다목적 비용 함수 (Multi-Objective Cost Function)

이 섹션이 시스템의 핵심이다. `portfolio_optimizer_greeks_mo` 함수 내의 `multi_objective_cost(w, iv_val)` 는 총 12개의 항목을 가중 합산하여 최소화 대상 스칼라 비용을 반환한다.

**입력 벡터:** $\\mathbf{{w}} = [w_0, w_1, w_2, w_3, w_4, w_{{cash}}]^T$
여기서 $w_0..w_3$: 옵션 가중치, $w_4$: 선물 가중치, $w_{{cash}}$: 현금 비중

$$\\mathcal{{L}}(\\mathbf{{w}}) = \\sum_{{k}} \\lambda_k \\cdot f_k(\\mathbf{{w}})$$

#### 4.7.1 수익-위험 비율 (Sharpe-Style)

$$f_{{return}} = -\\frac{{\\mu_{{port}}}}{{\\sigma_{{port}} + 0.1}}$$

$$\\mu_{{port}} = \\mathbf{{w}}_{{assets}}^T \\boldsymbol{{\\mu}}, \\quad \\sigma_{{port}} = \\sqrt{{\\mathbf{{w}}_{{assets}}^T \\boldsymbol{\\Sigma} \\mathbf{{w}}_{{assets}}}}$$

0.1 정규화 항은 포트폴리오 표준편차가 0에 가까울 때의 수치 불안정성을 방지한다.

가중치 $\\lambda_{{return}}$: Bullish=1.5, Bearish/Volatile/Neutral=1.0

#### 4.7.2 포트폴리오 분산 리스크

$$f_{{risk}} = \\sigma_{{port}}^2 \\cdot 1500$$

가중치 $\\lambda_{{risk}} = RA \\times c$: 여기서 $RA$는 리스크 회피도(1~10), $c$는 방향별 계수 (Bullish: 1.2, Bearish: 2.2, Volatile/Neutral: 2.0)

#### 4.7.3 델타 타겟팅 패널티

포트폴리오 델타를 목표값으로 유도하는 2차 페널티:

$$f_{{\\Delta}} = \\left(\\frac{{\\Delta_{{port}} - \\Delta_{{target}}}}{{1.0}}\\right)^2 \\cdot 500$$

**IV-Adaptive 방향성 가드레일 (Direction Guardrail):**

Bullish 시 $\\Delta_{{port}} < 0$ 이면:
$$f_{{dir}} = (|\\Delta_{{port}}| + 0.5)^2 \\cdot 50000$$

Bearish 시 $\\Delta_{{port}} > 0$ 이면:
$$f_{{dir}} = (\\Delta_{{port}} + 0.5)^2 \\cdot 50000$$

**효과적 델타 가중치:** Bearish/Volatile 에서는 1.5배 강화:
$$\\lambda_{{\\Delta}}^{{eff}} = \\lambda_{{\\Delta}} \\times 1.5 \\quad \\text{{(bearish/volatile)}}$$

#### 4.7.4 베가 타겟팅 패널티

IV 스케일 정규화를 통해 IV 수준에 비례한 베가 페널티를 부과:

$$f_{{\\nu}} = \\left(\\frac{{\\nu_{{port}} - \\nu_{{target}}}}{{\\max(IV/100, 0.1)}}\\right)^2$$

**IV 스케일링 팩터:**

$$s_{{IV}} = \\sqrt{{\\frac{{\\max(IV, 5.0)}}{{15.0}}}}$$

기준 IV = 15%로 설정. IV가 높을수록 베가 페널티 임계값이 완화된다.

#### 4.7.5 세타 수익 최대화

Bullish/Neutral 장세에서 세타 수취를 극대화:

$$f_{{\\Theta}} = -\\Theta_{{port}} \\cdot 70 \\cdot w_{{\\theta}}$$

Volatile/Bearish 장세에서는 $f_\\Theta = 0$ (세타보다 방향성 우선)

가중치 $\\lambda_\\Theta$: Neutral=6.0, Bearish=6.0, Volatile=4.0, Bullish=5.0 (기본값)

#### 4.7.6 감마 리스크 페널티

허용 한도 초과분에 대한 2차 페널티:

$$f_{{\\Gamma}} = \\left(\\max\\left(0, |\\Gamma_{{port}}| - \\Gamma_{{limit}}\\right)\\right)^2 \\cdot \\frac{{150}}{{DTE + 0.05}}$$

DTE → 0 이면 페널티가 기하급수적으로 증가한다.

Volatile 장세에서 음의 감마($\\Gamma_{{port}} < 0$)에 추가 패널티:

$$f_{{\\Gamma,extra}} = \\Gamma_{{port}}^2 \\cdot 4000 \\quad \\text{{(if volatile and }} \\Gamma < 0)$$

#### 4.7.7 비용 제어 — Log Barrier & Exponential Wall

Long 프리미엄 과다 지출을 억제하는 적응형 비용 장벽:

$$net\\_debit = (w_0 + w_2) - (w_1 + w_3) \\cdot 0.4$$

$$dist = MAX\\_DEBIT\\_RATIO - net\\_debit \\quad (MAX\\_DEBIT\\_RATIO = 0.50)$$

**로그 장벽 (한도 이내):**
$$f_{{debit}} = -\\ln(\\max(dist, 10^{{-6}})) \\cdot 50$$

**지수 장벽 (한도 초과):**
$$f_{{debit}} = c \\cdot (1 + |dist| \\cdot 100), \\quad c = 10^3\\text{{(bearish)}}, 10^4\\text{{(others)}}$$

이 구조는 최적화 엔진이 한도에 가까워질수록 부드러운 압박을 받다가, 초과 시 급격한 벽에 부딪히도록 설계되었다.

#### 4.7.8 스프레드 구조 균형 (Spread Balance)

Long 포지션 대비 Short 포지션의 최소 비율(40%)을 강제하여 Naked Long 구조를 방지:

**Bullish (Bull Call Spread 구조):**
$$f_{{spread}} = \\begin{{cases}} (0.4 \\cdot w_0 - w_1)^2 \\cdot 2000 & \\text{{if }} w_1 < 0.4 \\cdot w_0 \\\\ 0 & \\text{{otherwise}} \\end{{cases}}$$

**Bearish (Bear Put Spread 구조):**
$$f_{{spread}} = \\begin{{cases}} (0.4 \\cdot w_2 - w_3)^2 \\cdot 2000 & \\text{{if }} w_3 < 0.4 \\cdot w_2 \\\\ 0 & \\text{{otherwise}} \\end{{cases}}$$

#### 4.7.9 집중 리스크 방지 (L2 Regularization)

$$f_{{conc}} = \\sum_{{i}} w_i^2 \\cdot 50$$

Herfindahl 지수와 동일한 형태로, 단일 자산 과집중을 패널티한다.

`EXPIRATION_SCALPING` 모드에서는 계수를 150으로 강화하여 만기 전 전체 포지션 축소를 유도한다.

#### 4.7.10 선물 과다 사용 방지

$$f_{{fut}} = \\left(\\max(0, |w_4| - W_{{fut,max}})\\right)^2 \\cdot 3000$$

$$W_{{fut,max}} = \\begin{{cases}} 0.10 & \\text{{volatile}} \\\\ 0.30 & \\text{{others}} \\end{{cases}}$$

#### 4.7.11 전체 비용 함수 요약

$$\\mathcal{{L}} = \\lambda_r f_r + \\lambda_\\sigma f_\\sigma + \\lambda_\\Delta^{{eff}} f_\\Delta + \\lambda_\\nu f_\\nu + \\lambda_\\Theta f_\\Theta + f_{{spread}} + f_{{debit}} + f_{{\\Gamma}} + f_{{dir}} + f_{{fut}} + f_{{conc}} + f_{{mode}}$$

| 항목 | 변수 | 기본 계수 |
|------|------|---------|
| 수익-위험 | $f_r$ | $\\lambda_r \\in [1.0, 1.5]$ |
| 포트폴리오 분산 | $f_\\sigma$ | $\\lambda_\\sigma = RA \\times [1.2, 2.2]$ |
| 델타 페널티 | $f_\\Delta$ | 500 × ($\\lambda_\\Delta \\in [10, 50]$) |
| 베가 페널티 | $f_\\nu$ | $\\lambda_\\nu \\in [2, 4]$ |
| 세타 수익 | $f_\\Theta$ | -70 × $\\lambda_\\Theta \\in [4, 6]$ |
| 감마 페널티 | $f_\\Gamma$ | $150/(DTE+0.05)$ |
| 방향성 가드레일 | $f_{{dir}}$ | 50,000 |
| 비용 장벽 | $f_{{debit}}$ | Log(-50) / Wall($10^3 \\sim 10^4$) |
| 스프레드 균형 | $f_{{spread}}$ | 2,000 |
| 집중 방지 | $f_{{conc}}$ | 50 (스캘핑: 150) |
| 선물 제한 | $f_{{fut}}$ | 3,000 |

---

### 4.8 SLSQP 최적화

Sequential Least Squares Programming (SLSQP)는 비선형 제약 조건을 가진 연속형 최적화 문제에 가장 적합한 방법이다.

```python
res = scipy.optimize.minimize(
    multi_objective_cost,
    x0 = init_w,            # 초기값: 방향별 경험적 가중치
    args = (current_iv,),
    method = 'SLSQP',
    bounds = bounds,
    constraints = constraints,
    options = {{'maxiter': 2000, 'ftol': 1e-6, 'eps': 1e-3}}
)
```

**SLSQP 알고리즘 원리:**

SLSQP는 KKT(Karush-Kuhn-Tucker) 조건을 활용하는 QP(Quadratic Programming) 서브문제를 반복적으로 풀어 전역 최적해에 수렴한다. 각 반복에서:

$$\\mathbf{{w}}^{{(k+1)}} = \\mathbf{{w}}^{{(k)}} + \\alpha_k \\mathbf{{d}}_k$$

여기서 $\\mathbf{{d}}_k$는 QP 서브문제의 해, $\\alpha_k$는 라인 서치로 결정된 스텝 크기이다.

KKT 조건:
$$\\nabla \\mathcal{{L}} + \\sum_j \\mu_j \\nabla g_j + \\sum_k \\lambda_k \\nabla h_k = 0$$

---

### 4.9 포트폴리오 제약 조건 (Portfolio Constraints)

#### 4.9.1 등호 제약 (Equality)

$$\\sum_{{i=0}}^{{5}} w_i = 1.0 \\quad \\text{{(완전 투자 원칙)}}$$

#### 4.9.2 부등호 제약 (Inequality)

| 제약 | 수식 | 의도 |
|------|------|------|
| 현금 최소 보유 | $w_{{cash}} \\ge w_{{cash,min}}$ | 유동성 확보 (기본 20~30%) |
| 헤지 최소 비중 | $\\sum_{{i \\in hedge}} w_i \\ge 0.15$ | 최소 15% 헤지 강제 |
| Bullish 헤지 상한 | $w_1 + w_2 \\le 0.40$ | 방향 전환 방지 |
| Put Spread 보호 | $w_2 \\ge w_3$ | Naked Put 차단 (Bearish/Volatile) |

**방향별 헤지 인덱스:**

| 장세 | 헤지 자산 |
|------|----------|
| Bullish | [1: Call Short, 2: Put Long] |
| Bearish | [0: Call Long, 3: Put Short] |
| Volatile | [1: Call Short, 3: Put Short] |
| Neutral | [0: Call Long, 2: Put Long] |

#### 4.9.3 자산별 경계값 (Bounds)

| 자산 | 하한 | 상한 |
|------|------|------|
| Call Long ($w_0$) | 0.0 | INSURANCE_LIMIT(0.05) if Bearish, else MAX_WEIGHT(0.45) |
| Call Short ($w_1$) | 0.0 | 0.15 if Volatile, else 0.45 |
| Put Long ($w_2$) | 0.10 if Bearish/Volatile, else 0.0 | 0.65 if Bearish, else 0.45 |
| Put Short ($w_3$) | 0.0 | 0.45 |
| Mini Future ($w_4$) | $-W_{{fut,max}}$ | $+W_{{fut,max}}$ |
| Cash ($w_5$) | $w_{{cash,min}}$ | 1.0 |

---

### 4.10 선물 델타 헤징 알고리즘

옵션 포지션의 잔여 델타를 미니 선물로 정밀 보정하는 2단계 알고리즘.

#### 4.10.1 기대 이동량 기반 델타 조정

단순 델타가 아닌, 감마를 고려한 **조정 델타(Adjusted Delta)** 를 사용한다:

$$E[\\Delta S] = S \\cdot \\sigma \\cdot \\sqrt{{DTE / 252}}$$

$$\\Delta_{{adj}} = \\Delta_{{opt}} + \\Gamma_{{opt}} \\cdot E[\\Delta S]$$

여기서 $\\Delta_{{opt}}, \\Gamma_{{opt}}$는 확정된 옵션 포지션 전체의 합산 Greeks이다:

$$\\Delta_{{opt}} = \\sum_{{i}} N_i \\cdot \\Delta_i, \\quad \\Gamma_{{opt}} = \\sum_{{i}} N_i \\cdot \\Gamma_i$$

#### 4.10.2 선물 수량 산출

$$\\delta_{{gap}} = \\Delta_{{target}} - \\Delta_{{adj}}$$

$$N_{{futures}} = \\text{{round}}(\\delta_{{gap}})$$

- 최소 헤지 임계값: $|\\delta_{{gap}}| < 0.15$ (Volatile: 0.25) 이면 헤지 스킵
- Volatile 장세 상한: $N_{{futures}} \\in [-2, +2]$ (과도한 방향 노출 방지)
- 자본 기준 상한: $\\max |N| = \\lfloor (Capital \\cdot |w_4|) / M_{{init}} \\rfloor$

여기서 $M_{{init}}$ = 미니 선물 개시 증거금 (기본: 2,500,000원)

미니 KOSPI200 선물 손익:
$$PnL_{{future}} = (S_{{exit}} - S_{{entry}}) \\times 50,000 \\times N$$

#### 4.10.3 증거금 기반 통합 제어

증거금 계산 (Naked Short 기준):
$$M_{{naked,i}} = K_i \\cdot 50,000 \\times 0.15 \\quad \\text{{(Short 포지션)}}$$
$$M_{{long,i}} = P_i \\cdot 50,000 \\quad \\text{{(Long 포지션)}}$$

스프레드 포지션의 실질 증거금:
$$M_{{spread}} = \\max(0, N_{{short}} - N_{{long}}) \\times M_{{naked}}$$

**전체 증거금 사용률:**
$$R_{{margin}} = \\frac{{\\sum M_{{options}} + |N_{{fut}}| \\cdot M_{{init}}}}{{Capital}}$$

**자동 스케일링:** $R_{{margin}} > 70\\%$ 이면:
$$scale = \\frac{{0.70 - 0.05}}{{R_{{margin}}}} = \\frac{{0.65}}{{R_{{margin}}}}$$
$$N_i^{{new}} = \\lfloor N_i \\times scale \\rfloor, \\quad N_{{fut}}^{{new}} = \\lfloor N_{{fut}} \\times scale \\rfloor$$

---

### 4.11 동적 리스크 스코어 (Dynamic Risk Score)

LLM 출력 점수를 기반으로 실제 시장 데이터를 결합한 종합 리스크 점수:

$$RA = \\text{{clip}}\\left(S_{{LLM}} + \\delta_{{macro}} + \\delta_{{acc}} + \\delta_{{sentiment}} + \\delta_{{price}} + \\delta_{{trend}} + \\delta_{{div}} + \\delta_{{IV}}, 1.0, 10.0\\right)$$

| 조정 항목 | 조건 | 조정값 |
|---------|------|-------|
| 매크로 예측 ($\\delta_{{macro}}$) | $pred < 0$ | $+\\min(2.5, |pred| \\times 0.5)$ |
| 방향 정확도 ($\\delta_{{acc}}$) | 항상 | $+\\max(0, (1-acc) \\times 1.5)$ |
| 뉴스 심리 ($\\delta_{{sent}}$) | Negative / Positive | +1.5 / -0.8 |
| 가격 방향 ($\\delta_{{price}}$) | 하락 중 / 상승 중 | +0.6 / -0.4 |
| 트렌드 ($\\delta_{{trend}}$) | Bearish/Volatile/Bullish | +1.2 / +0.6 / -0.6 |
| 다이버전스 ($\\delta_{{div}}$) | divergence 감지 | +0.5 |
| IV ($\\delta_{{IV}}$) | ≥30 / ≥25 / ≥20 | +2.0 / +1.2 / +0.6 |

---

## 5. 매크로 예측기 Ridge Regression

### 5.1 목적

해외 지수 및 거시 경제 지표의 전일 데이터로 **다음 거래일 KOSPI200 로그수익률**을 예측한다.

### 5.2 데이터 수집 및 피처 엔지니어링

**수집 지표:**

- **가격 지수:** KOSPI200, S&P500, NASDAQ, SOX, Nikkei225, USD/KRW, WTI Oil
- **수준 지표:** VIX, US 10Y금리, FED금리
- **거시 지표:** CPI, 실업률, GDP, 소매판매, 내구재 주문

**로그 수익률 변환 (가격 지수):**

$$r_t = \\ln\\left(\\frac{{P_t}}{{P_{{t-1}}}}\\right)$$

**발표 시차(Publication Lag) 처리:**

| 지표 | Lag (거래일) |
|------|------------|
| CPI | 20일 |
| GDP | 45일 |
| 소매판매 | 15일 |
| 내구재 주문 | 20일 |
| 실업률 | 10일 |

**니케이 동조화:** 한국 장 개장 전 일본 시장이 이미 종료되므로 T 시점 동기화 반영.

**타겟 변수:**
$$Y = r_{{KOSPI200,t+1}} \\quad \\text{{(다음 거래일 KOSPI 로그수익률)}}$$

**피처 벡터:**
$$\\mathbf{{X}} = [r_{{SP500,t}}, r_{{NASDAQ,t}}, r_{{SOX,t}}, r_{{USDJPY,t}}, r_{{WTI,t}}, VIX_t, US10Y_t, FED_t, \\ldots]^T$$

(모두 T-1 시차 적용: 전일 데이터로 당일 예측)

### 5.3 Ridge Regression 모델

Ridge는 OLS에 L2 정규화를 추가하여 다중공선성 문제를 완화한다:

$$\\hat{{\\boldsymbol{{\\beta}}}} = \\arg\\min_{{\\boldsymbol{{\\beta}}}} \\left[ \\sum_{{t=1}}^{{n}} \\left(Y_t - \\mathbf{{X}}_t^T \\boldsymbol{{\\beta}}\\right)^2 + \\alpha \\|\\boldsymbol{{\\beta}}\\|_2^2 \\right]$$

닫힌 형태의 해:
$$\\hat{{\\boldsymbol{{\\beta}}}} = (\\mathbf{{X}}^T \\mathbf{{X}} + \\alpha \\mathbf{{I}})^{{-1}} \\mathbf{{X}}^T \\mathbf{{Y}}$$

**정규화 파라미터:** $\\alpha = 1.0$ (기본값)

**StandardScaler 전처리:**
$$\\tilde{{X}}_j = \\frac{{X_j - \\bar{{X}}_j}}{{s_j}}$$

**학습/검증 분리:** 시계열 순서 유지, 85%/15% 분할 (데이터 누수 방지)

**예측값 변환:**

$$\\hat{{r}}_{{pred}} = \\text{{pipeline.predict}}(\\mathbf{{X}}_{{latest}})$$

$$\\hat{{P}}_{{tomorrow}} = P_{{today}} \\cdot e^{{\\hat{{r}}_{{pred}}}}$$

$$\\hat{{ret}}(\\%) = (e^{{\\hat{{r}}_{{pred}}}} - 1) \\times 100$$

**평가 지표:**
- RMSE: $\\sqrt{{\\frac{{1}}{{n_{{test}}}} \\sum (\\hat{{r}} - r)^2}}$
- 방향 정확도: $acc = \\frac{{1}}{{n_{{test}}}} \\sum \\mathbf{{1}}[\\text{{sign}}(\\hat{{r}}) = \\text{{sign}}(r)]$

**DivergenceChecker 연동:** pred_pct < -0.5% 이면 시장 트렌드를 강제 Bearish 전환하고 리스크 회피도 +1.0 조정.

---

## 6. Multi-Agent Debate Engine

### 6.1 설계 철학

단일 LLM의 확증 편향(Confirmation Bias)을 억제하기 위해 **적대적 논쟁 구조**를 채택한다. Bull Agent와 Bear Agent가 동일한 뉴스를 서로 다른 관점에서 분석하고, Judge가 중재한다.

### 6.2 Self-Improvement Loop (Evaluator)

기존 debate 시스템에 없던 **품질 자기평가 루프**가 FOV17의 핵심 추가 기능이다:

```
Judge 합의문 → Evaluator LLM 평가 (1~10점)
    ├── score ≥ 임계값 또는 retry ≥ 2 → approved → END
    └── score 미달 → eval_critique 생성 → Bull/Bear에 피드백으로 전달 → 재토론
```

Evaluator 출력 스키마 (`EvalOutput`):
- `score`: 1~10 정수 (논리 완결성 + 뉴스 반영도)
- `is_sufficient`: bool (채택 가능 여부)
- `critique`: 한국어 피드백 (보완점 설명)

재토론 시 Bull/Bear 에이전트는 이전 `eval_critique`를 프롬프트에 포함하여 개선된 논거를 생성한다.

### 6.3 Divergence Detection

Judge가 출력하는 `divergence_note`는 가격 액션(Price Action)과 뉴스 심리 간의 괴리를 설명한다.

`DivergenceChecker` 노드는 이 정보와 매크로 예측기 결과를 결합하여:
1. 매크로 하락 예측 시 강제 Bearish 전환
2. 뉴스 강세에도 가격 하락 중이면 "Bullish Climber" → Long 자산 mu 가중치 강화

---

## 7. 학습 및 피드백 시스템

### 7.1 ScenarioManager

`macro_scenarios` 테이블에서 과거 시나리오를 로드하여 FAISS 벡터 인덱스를 구축한다.

**유사 시나리오 검색:**
$$anchor = \\arg\\max_{{s}} \\text{{cos\\_sim}}(\\text{{embed}}(consensus), \\text{{embed}}(desc_s))$$

**앵커 기반 파라미터 제약:**
- LLM이 μ, σ, corr를 추정할 때 앵커 값에서 ±20% 이내로 조정 (할루시네이션 방지)
- `CRITICAL RULE: Adjust anchor mu by no more than 20% based on current IV`

### 7.2 Learning Node

기대 수익률 ≥ 3% 달성 시 해당 시장 뷰와 파라미터를 DB에 저장:

```python
scenario_manager.update_successful_scenario(
    view_text, mu, vol, corr, expected_ret, anchor_name
)
```

축적된 성공 케이스가 ScenarioManager의 앵커 풀에 추가되어 미래 최적화에 활용된다 (강화학습적 피드백 구조).

---

## 8. Key References

### 옵션 가격 결정 및 Greeks

1. **Black, F. & Scholes, M. (1973).** "The Pricing of Options and Corporate Liabilities." *Journal of Political Economy*, 81(3), 637–654.

2. **Merton, R. C. (1973).** "Theory of Rational Option Pricing." *Bell Journal of Economics and Management Science*, 4(1), 141–183.

3. **Hull, J. C. (2022).** *Options, Futures, and Other Derivatives* (11th ed.). Pearson.

### 변동성 스큐

4. **Rubinstein, M. (1994).** "Implied Binomial Trees." *Journal of Finance*, 49(3), 771–818.

5. **Derman, E. & Kani, I. (1994).** "Riding on a Smile." *Risk*, 7(2), 32–39.

6. **Gatheral, J. (2006).** *The Volatility Surface: A Practitioner's Guide*. Wiley Finance.

### 포트폴리오 최적화

7. **Markowitz, H. (1952).** "Portfolio Selection." *Journal of Finance*, 7(1), 77–91.

8. **Nocedal, J. & Wright, S. J. (2006).** *Numerical Optimization* (2nd ed.). Springer. (SLSQP 참조)

9. **Kraft, D. (1988).** "A software package for sequential quadratic programming." Tech. Rep. DFVLR-FB 88-28. (scipy SLSQP 기반 알고리즘)

### 다목적 최적화

10. **Zitzler, E., Laumanns, M. & Thiele, L. (2001).** "SPEA2: Improving the Strength Pareto Evolutionary Algorithm." ETH Zurich Technical Report.

11. **Boyd, S. & Vandenberghe, L. (2004).** *Convex Optimization*. Cambridge University Press.

### Ridge Regression / 거시 예측

12. **Hoerl, A. E. & Kennard, R. W. (1970).** "Ridge Regression: Biased Estimation for Nonorthogonal Problems." *Technometrics*, 12(1), 55–67.

13. **Welch, I. & Goyal, A. (2008).** "A Comprehensive Look at the Empirical Performance of Equity Premium Prediction." *Review of Financial Studies*, 21(4), 1455–1508.

### LLM 및 RAG

14. **Lewis, P. et al. (2020).** "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." *NeurIPS 2020*.

15. **Qwen Team (2024).** "Qwen2.5: A Party of Foundation Models." arXiv:2412.15115.

### LangGraph / 에이전트 시스템

16. **Harrison Chase et al. (2023).** *LangChain / LangGraph Documentation*. LangChain Inc.

17. **Park, J. S. et al. (2023).** "Generative Agents: Interactive Simulacra of Human Behavior." *UIST 2023*. (Multi-Agent 설계 참조)

---

*본 보고서는 LLMTradEx34OptMiniFOV17.py (3,873 lines) 코드베이스를 기반으로 자동 생성되었습니다.*
*생성일시: 2026년 03월 26일*
