import os
import faiss
from collections import deque
from pathlib import Path

# ===================== 0) CONFIG =====================
CONFIG_TODAY_OVERRIDE = None  # 오늘 날짜 오버라이드 (YYYY-MM-DD 형식)

# ===================== 1) Fine-tuned Job IDs =====================
DIALECT_FT_JOB_ID  = "ftjob-"
EMPATHY_FT_JOB_ID  = "ftjob-"

FINETUNED_DIALECT_MODEL = None
FINETUNED_EMPATHY_MODEL = None

# ===================== 2) Embedding / FAISS =====================
dimension = 768
faiss_index = faiss.IndexFlatL2(dimension)

# ===================== 3) Memories / Globals =====================
conversation_memory_std = []  # 표준어 대화 기록
conversation_memory_raw = []  # 사용자 원문 대화 기록

fact_memory = []        # 추출된 사실 저장
fact_embeddings = []    # 사실 임베딩
fact_id_counter = 0     # 사실 ID 카운터

memory_score = 100      # 일관성 점수
RECENT_CONSISTENCY_BIN = deque(maxlen=5) # 최근 일관성 기록

CONTEXT_TOPIC_LABEL = None # 현재 맥락 토픽
CONTEXT_TOPIC_CONF  = 0.0  # 토픽 확신도

diary_memory = []       # 완료된 일기 세션
diary_id_counter = 0

conversation_log = []   # 전체 대화 로그
_conv_idx = 0

# ===================== 4) Topic Pool (백업) =====================
BACKUP_MACRO_TOPICS = [
    "가족모임","경로당","복지관","학창시절","졸업식","환갑","칠순","명절","설날","추석",
    "시장","극장","손주","건강검진","병원","약국","실버교실",
    "교회","성당","절","봉사","동호회","산책","공원","등산","바다","강변",
    "버스","지하철","청소","집정리","편지","선물","날씨","비","눈","회상"
]

# ===================== 5) Diary Definitions =====================
CHECKS = [
    ("today_date", "오늘이 몇월 며칠일까요?"),
    ("today_weather", "오늘 날씨는 어떤가요?"),
    ("current_location", "지금 어디에 계신가요?"),
    ("date_7days_ago", "오늘로부터 7일 전은 몇월 며칠일까요?"),
    ("yesterday_activity", "어제 뭐하셨어요?"),
]
DIARY_CHECK_KEYS = [k for k, _ in CHECKS]

DIARY_QUESTION_TEMPLATES = [
    "‘{t}’ 하면 떠오르는 장면이나 느낌이 있으세요?",
    "‘{t}’와 관련해서 최근에 있었던 일 하나만 이야기해 주실래요?",
    "‘{t}’이(가) 요즘 일상에 어떤 영향을 주고 있나요?",
    "‘{t}’과(와) 관련해 가장 기억에 남는 순간은 언제였나요?",
    "‘{t}’에 대해 예전과 지금을 비교하면 뭐가 달라졌나요?",
    "‘{t}’이(가) 요즘 마음이나 건강에 어떤 도움(또는 어려움)을 주나요?",
    "‘{t}’을(를) 가족/친구와 연결해서 떠오르는 일이 있을까요?",
    "‘{t}’을(를) 다음에 할 때 바라는 점이나 계획이 있나요?"
]

STOP_WORDS = ["그만","일기 끝","일기 종료","종료","끝낼래","그만할래"]
NEXT_WORDS = ["다른 단어"]

CONSENT_PROMPT = "이 주제에 대해 더 이야기해볼까요? 원하시면 이어가고, 아니면 다음 주제로 넘어갈게요."