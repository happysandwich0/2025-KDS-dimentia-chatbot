import os, json, time, random, re, csv
import numpy as np
from datetime import datetime, timedelta, date
from collections import deque
from pathlib import Path

# 모듈 분리된 공용 함수
from model import ask_gpt, resolve_finetuned_model, load_few_shot_empathy

# -------------------- Optional deps: FAISS / SentenceTransformer --------------------
try:
    import faiss
except Exception:
    os.system("pip install -q faiss-cpu")
    import faiss

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    os.system("pip install -q sentence_transformers")
    from sentence_transformers import SentenceTransformer

# [ADD] pandas (DataFrame로 대화 불러오기용)
try:
    import pandas as pd
except Exception:
    os.system("pip install -q pandas")
    import pandas as pd

# ===================== 0) CONFIG =====================
CONFIG_TODAY_OVERRIDE = None  # e.g., "2025-09-03"  ← 필요 시만 설정

# ===================== 1) Fine-tuned Job IDs =====================
DIALECT_FT_JOB_ID  = "ftjob-5OaW41C27QVbi2NkMRjWgJEh"
EMPATHY_FT_JOB_ID  = "ftjob-EQXVYCKS5YaoknI9njHKvFvT"

FINETUNED_DIALECT_MODEL = None
FINETUNED_EMPATHY_MODEL = None

# ===================== 2) Embedding / FAISS =====================
embedder = SentenceTransformer("jhgan/ko-sbert-nli")
dimension = 768
faiss_index = faiss.IndexFlatL2(dimension)

# ===================== 3) Memories / Globals =====================
conversation_memory_std = []  # 표준어
conversation_memory_raw = []  # 원문(사투리)

fact_memory = []        # list[dict]
fact_embeddings = []    # list[np.ndarray]
fact_id_counter = 0

memory_score = 100
RECENT_CONSISTENCY_BIN = deque(maxlen=5)

# 맥락 토픽 상태
CONTEXT_TOPIC_LABEL = None
CONTEXT_TOPIC_CONF  = 0.0

# Diary state
DIARY_MODE = False
diary_memory = []
diary_id_counter = 0

# [FIX] Diary control keywords
DIARY_NEXT_WORDS = ["다른 단어"]
DIARY_STOP_WORDS = ["그만", "일기 끝", "일기 종료", "종료", "끝낼래", "그만할래"]

# [ADD] ===== 전체 대화 로그 =====
conversation_log = []   # list[dict]: {idx, ts, role, topic, content_raw, content_std, meta}
_conv_idx = 0

def _ts(ts):
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ""

def log_event(role: str,
              content_raw: str | None = None,
              content_std: str | None = None,
              topic: str | None = None,
              meta: dict | None = None,
              ts: float | None = None):
    """모든 대화 턴을 일관 포맷으로 기록."""
    global _conv_idx
    if ts is None:
        ts = time.time()
    conversation_log.append({
        "idx": _conv_idx,
        "ts": ts,
        "ts_str": _ts(ts),
        "role": role,
        "topic": topic or "",
        "content_raw": content_raw or "",
        "content_std": content_std or (content_raw or ""),
        "meta": json.dumps(meta, ensure_ascii=False) if meta else ""
    })
    _conv_idx += 1

def conversation_log_dataframe() -> pd.DataFrame:
    """세션 내 전체 대화 DataFrame 반환(메모리 상)."""
    if not conversation_log:
        return pd.DataFrame(columns=["idx","ts","ts_str","role","topic","content_raw","content_std","meta"])
    df = pd.DataFrame(conversation_log)
    return df.sort_values("idx", ignore_index=True)

get_conversation_log_df = conversation_log_dataframe

# ===================== 4) Topic Pool (백업) =====================
BACKUP_MACRO_TOPICS = [
    "가족모임","경로당","복지관","학창시절","졸업식","환갑","칠순","명절","설날","추석",
    "시장","극장","손주","건강검진","병원","약국","실버교실",
    "교회","성당","절","봉사","동호회","산책","공원","등산","바다","강변",
    "버스","지하철","청소","집정리","편지","선물","날씨","비","눈","회상"
]

# ===================== 5) Utilities =====================
TIME_WORDS = ["오늘","어제","내일","지금","방금","저녁","아침","점심",
              "월요일","화요일","수요일","목요일","금요일","토요일","일요일",
              "이번 주","지난 주","다음 주","이번 달","지난 달","다음 달"]

def calc_specificity_score(text: str) -> float:
    if not text: return 0.0
    tokens = text.split()
    n = max(1, len(tokens))
    digits = sum(ch.isdigit() for ch in text)
    has_time = any(w in text for w in TIME_WORDS) or bool(re.search(r"\d{1,2}\s*월\s*\d{1,2}\s*일|\d{4}[-/.]\d{1,2}[-/.]\d{1,2}", text))
    named_like = sum(text.count(s) for s in ["님","씨","선생","교수","과장","팀장","박사","군","양"])
    score = (0.3 * (1 if has_time else 0) +
             0.3 * min(1.0, digits/4) +
             0.2 * min(1.0, named_like/2) +
             0.2 * min(1.0, n/12))
    return max(0.0, min(1.0, score))

# ===================== 6) GPT Few-shot =====================
few_shot_empathy = load_few_shot_empathy()

# ===================== 7) Dialect → Standard =====================
_DIALECT_SYSTEM_PROMPT = (
    "너는 한국어 사투리를 한국어 표준어(존댓말)로 자연스럽게 바꾸는 도우미야. "
    "입력 문장이 사투리인지 감지하고, 표준어로 매끄럽게 변환해. "
    "존댓말로 바꾸되 의미를 바꾸지 말고, 출력은 JSON으로만 해."
)
_DIALECT_JSON_SCHEMA = """
다음 JSON 형식으로만 답해:
{
  "standard": "표준어로 자연스럽게 바꾼 문장",
  "is_dialect": true,
  "confidence": 0.0
}
"""
_DIALECT_MARKERS = ["데이","카이","아이가","아입니꺼","하께","쿠다","머시","카노","카더라","하믄","그라믄",
                    "했심더","했데이","하이소","하입니더","무했노","마","예","그카이","고마","그라제"]

def _looks_like_dialect(s: str) -> bool:
    s = s or ""
    return any(tok in s for tok in _DIALECT_MARKERS)

def _safe_json_loads(raw: str, fallback: dict | None = None) -> dict | None:
    if not raw: return fallback
    try:
        return json.loads(raw)
    except Exception:
        pass
    if "{" in raw and "}" in raw:
        try:
            start, end = raw.index("{"), raw.rindex("}")+1
            return json.loads(raw[start:end])
        except Exception:
            return fallback
    return fallback

def normalize_user_utterance(user_text: str) -> dict:
    global FINETUNED_DIALECT_MODEL
    if FINETUNED_DIALECT_MODEL is None:
        FINETUNED_DIALECT_MODEL = resolve_finetuned_model(DIALECT_FT_JOB_ID)

    model_to_use = FINETUNED_DIALECT_MODEL or "gpt-4o-mini"
    prompt = f"{_DIALECT_SYSTEM_PROMPT}\n\n입력:\n\"\"\"\n{user_text}\n\"\"\"\n\n{_DIALECT_JSON_SCHEMA}"
    raw = ask_gpt(prompt, model=model_to_use, temperature=0.0, max_tokens=200,
                  response_format={"type":"json_object"})
    data = _safe_json_loads(raw, fallback={})
    standard = (data.get("standard") or "").strip()
    is_dialect = bool(data.get("is_dialect", False))
    conf = float(data.get("confidence", 0.0) or 0.0)

    need_retry = (not standard) or ("너는 한국어 사투리를" in standard) or _looks_like_dialect(standard)
    if need_retry:
        raw2 = ask_gpt(prompt, model="gpt-4o-mini", temperature=0.0, max_tokens=200,
                       response_format={"type":"json_object"})
        data2 = _safe_json_loads(raw2, fallback={})
        standard2 = (data2.get("standard") or "").strip()
        if standard2 and "너는 한국어 사투리를" not in standard2 and not _looks_like_dialect(standard2):
            standard = standard2
            is_dialect = bool(data2.get("is_dialect", is_dialect))
            conf = float(data2.get("confidence", conf) or conf)

    if not standard or _looks_like_dialect(standard):
        minimalist_prompt = (
            "다음 문장을 한국어 표준어(존댓말)로 한 문장으로만 바꿔줘. "
            "JSON 없이 결과 문장만 출력해.\n\n"
            f"문장: {user_text}"
        )
        std3 = ask_gpt(minimalist_prompt, model="gpt-4o-mini", temperature=0.0, max_tokens=120,
                       response_format={"type":"text"})
        if std3:
            standard = std3.strip()

    if not standard:
        standard = user_text

    return {
        "standard": standard,
        "is_dialect": is_dialect,
        "confidence": max(0.0, min(1.0, conf)),
        "raw": user_text
    }

# ===================== 8) 맥락 토픽 라벨링 =====================
def _window_text(history_std: list[str], current_std: str, k:int=6) -> str:
    tail = " ".join(history_std[-k:])
    return (tail + " " + (current_std or "")).strip()

def infer_context_topic_label(history_std: list[str], current_std: str) -> tuple[str, float]:
    ctx = _window_text(history_std, current_std, k=6)
    prompt = (
        "다음 한국어 대화 맥락의 전반 주제를 1~3어절의 일반명사/짧은 구로 요약하세요.\n"
        "세부어/희귀어 금지, 새 정보 창작 금지.\n"
        "JSON으로만 답: {\"label\":\"...\",\"confidence\":0.0}\n\n"
        f"[대화 맥락]\n{ctx}\n"
    )
    raw = ask_gpt(prompt, model="gpt-4o-mini", temperature=0.0, max_tokens=120,
                  response_format={"type":"json_object"})
    try:
        data = json.loads(raw)
        label = (data.get("label") or "").strip() or "일상"
        conf = float(data.get("confidence", 0.0) or 0.0)
        if label in ["이야기","대화","일상 이야기","소소한 대화"]:
            label, conf = "일상", min(conf, 0.55)
        if len(label) > 20:
            label = label[:20]
        return label, conf
    except Exception:
        return "일상", 0.0

def smooth_context_topic(new_label: str, new_conf: float,
                         prev_label: str | None, prev_conf: float,
                         min_change_conf: float = 0.60,
                         drift_guard: float = 0.15) -> tuple[str, float, bool]:
    if not prev_label:
        return new_label, new_conf, True
    if new_label == prev_label:
        fused = max(new_conf, (new_conf + prev_conf) / 2)
        return prev_label, fused, False
    if (new_conf >= min_change_conf) and ((new_conf - prev_conf) >= drift_guard):
        return new_label, new_conf, True
    return prev_label, prev_conf, False

# ===================== 9) Fact Extract / Store / Consistency =====================
def extract_claims_from_utterance(user_input_std: str, recent_history_std: list[str], original_raw: str, nrm_meta: dict):
    global fact_id_counter
    formatted_history = []
    for i, msg in enumerate(recent_history_std[-6:]):
        role = "사용자(표준어)" if i % 2 == 0 else "시스템"
        formatted_history.append(f"{role}: {msg}")
    recent_history_str = "\n".join(formatted_history)
    prompt = f"""
최근 대화 기록 (참고용, 표준어 기준):
{recent_history_str}

다음 **사용자 발화(표준어)**에서 **핵심적인 사실(Claim)**만 추출하여 JSON 배열 형태로 반환해 주세요.
의견/감탄/질문/일반 인사/추측/중요치 않은 말은 제외.
항상 **배열**만 반환. 없으면 [].

각 사실 JSON 형식:
{{
  "claim_text": "...",
  "entities": ["..."],
  "type": "개인정보|일상활동|감정|사건|계획|...",
  "summary": "...",
  "time_reference": "어제|지난주|오늘|내일|현재",
  "relative_offset_days": -1|0|1|null
}}

사용자 발화(표준어): "{user_input_std}"
"""
    json_str = ask_gpt(prompt, model="gpt-4o", temperature=0.2,
                       response_format={"type":"json_object"})
    try:
        claims_data = json.loads(json_str)
        if isinstance(claims_data, dict) and claims_data:
            claims_data = [claims_data] if 'claim_text' in claims_data else []
        elif not isinstance(claims_data, list):
            return []
        extracted = []
        for c in claims_data:
            ct = c.get("claim_text")
            if not ct: continue
            rod = c.get("relative_offset_days")
            if isinstance(rod, str) and rod.lower() == "null":
                rod = None
            extracted.append({
                "id": f"fact_{fact_id_counter}",
                "claim_text": ct,
                "entities": c.get("entities", []),
                "type": c.get("type","미분류"),
                "summary": c.get("summary", ct),
                "time_reference": c.get("time_reference","현재"),
                "relative_offset_days": rod,
                "original_utterance_raw": original_raw,
                "original_utterance_std": user_input_std,
                "was_dialect_normalized": bool(nrm_meta.get("is_dialect", False)),
                "dialect_confidence": float(nrm_meta.get("confidence", 0.0))
            })
            fact_id_counter += 1
        return extracted
    except Exception:
        return []

def store_extracted_facts(extracted_facts):
    for fact in extracted_facts:
        fact_ts = time.time()
        rod = fact.get("relative_offset_days")
        if isinstance(rod, (int,float)):
            fact_ts = (datetime.fromtimestamp(time.time()) + timedelta(days=rod)).timestamp()
        fact["timestamp"] = fact_ts
        fact_memory.append(fact)
        emb = embedder.encode([fact["claim_text"]])[0].astype("float32")
        fact_embeddings.append(emb)
        faiss_index.add(np.array([emb]))
    print(f"💾 {len(extracted_facts)}개의 사실 저장됨.")

def find_related_old_facts(new_claim_embedding, top_k=5):
    if not fact_embeddings: return []
    distances, indices = faiss_index.search(np.array([new_claim_embedding]).astype("float32"),
                                            min(top_k, len(fact_embeddings)))
    return [fact_memory[i] for i in indices[0] if i != -1]

def fact_tracking_agent(new_fact, related_old_facts):
    related = "\n".join([
        f"- ID:{f['id']}, 사실:{f['claim_text']}, (표준어발화:'{f.get('original_utterance_std','')}', 시간:{datetime.fromtimestamp(f['timestamp']).strftime('%Y-%m-%d %H:%M:%S')})"
        for f in related_old_facts
    ]) or "없음."
    prompt = f"""
새로운 사용자 발화에서 추출된 사실:
- ID:{new_fact['id']}, 사실:"{new_fact['claim_text']}", (표준어발화:'{new_fact.get('original_utterance_std','')}', 시간:{datetime.fromtimestamp(new_fact['timestamp']).strftime('%Y-%m-%d %H:%M:%S')})

관련 기존 사실들:
{related}

위 정보를 바탕으로, 관계를 다음 중 하나로 JSON으로만 반환:
{{ "decision": "CONSISTENT|UPDATE|CONTRADICTION|NEW" }}
"""
    raw = ask_gpt(prompt, model="gpt-4o", temperature=0.2, max_tokens=200,
                  response_format={"type":"json_object"})
    try:
        data = json.loads(raw)
        if "decision" in data:
            return data
    except Exception:
        pass
    return {"decision":"ERROR"}

def check_memory_consistency(user_input_std: str, user_input_raw: str, nrm_meta: dict):
    global memory_score, CONTEXT_TOPIC_LABEL, CONTEXT_TOPIC_CONF, FINETUNED_EMPATHY_MODEL

    # 맥락 토픽 추정 + 스무딩
    new_topic_label, new_topic_conf = infer_context_topic_label(conversation_memory_std, user_input_std)
    CONTEXT_TOPIC_LABEL, CONTEXT_TOPIC_CONF, _ = smooth_context_topic(
        new_topic_label, new_topic_conf, CONTEXT_TOPIC_LABEL, CONTEXT_TOPIC_CONF,
        min_change_conf=0.60, drift_guard=0.15
    )

    # 사실 추출
    extracted = extract_claims_from_utterance(user_input_std, conversation_memory_std, user_input_raw, nrm_meta)
    if not extracted:
        print("💡 추출된 사실 없음.")
        return

    # 각 fact에 맥락 토픽 부여
    for nf in extracted:
        nf["topic"] = CONTEXT_TOPIC_LABEL or "일상"
        nf["topic_confidence"] = float(CONTEXT_TOPIC_CONF)

    # 저장
    store_extracted_facts(extracted)

    # 관계/점수
    for nf in extracted:
        emb = embedder.encode([nf["claim_text"]])[0]
        olds = [f for f in find_related_old_facts(emb) if f["id"] != nf["id"]]
        d = fact_tracking_agent(nf, olds)
        dval = d.get("decision")
        cb = 1 if dval in ("CONSISTENT","UPDATE","NEW") else 0
        RECENT_CONSISTENCY_BIN.append(cb)
        for f in reversed(fact_memory):
            if f["id"] == nf["id"]:
                f["decision"] = dval
                f["consistency_binary"] = cb
                break
        if dval == "CONTRADICTION":
            memory_score = max(0, memory_score - 15)
        elif dval == "UPDATE":
            memory_score = min(100, memory_score + 5)
        elif dval in ("CONSISTENT","NEW"):
            memory_score = min(100, memory_score + 1)
        print(f"결정:{dval}, 현재기억점수:{memory_score}, 토픽:{CONTEXT_TOPIC_LABEL}({CONTEXT_TOPIC_CONF:.2f})")

# ===================== 10) 날짜/스코어 (KST 안정화) =====================
try:
    from zoneinfo import ZoneInfo
    _KST = ZoneInfo("Asia/Seoul")
except Exception:
    _KST = None

def _now_kst_dt():
    if CONFIG_TODAY_OVERRIDE:
        try:
            y, m, d = map(int, CONFIG_TODAY_OVERRIDE.split("-"))
            return datetime(y, m, d, 12, 0, 0)  # 정오 고정(경계 회피)
        except Exception:
            pass
    try:
        return datetime.now(_KST) if _KST else datetime.now()
    except Exception:
        return datetime.now()

def _today_mmdd_ko():
    now = _now_kst_dt()
    return f"{now.month}월 {now.day}일"

def _date_7days_ago_mmdd_ko():
    d = _now_kst_dt() - timedelta(days=7)
    return f"{d.month}월 {d.day}일"

def _canonicalize_mmdd(text: str):
    if not text: return None
    s = re.sub(r"\s+", " ", text).strip()
    m = re.search(r'(\d{1,2})\s*월\s*(\d{1,2})\s*일', s)
    if m: return int(m.group(1)), int(m.group(2))
    m = re.search(r'(\d{4})[./-](\d{1,2})[./-](\d{1,2})', s)
    if m: return int(m.group(2)), int(m.group(3))
    m = re.search(r'(\d{1,2})[./-](\d{1,2})', s)
    if m: return int(m.group(1)), int(m.group(2))
    return None

def score_today_date(user_answer_std: str) -> int:
    gold = _canonicalize_mmdd(_today_mmdd_ko())
    pred = _canonicalize_mmdd(user_answer_std or "")
    return 1 if (gold and pred and gold == pred) else 0

def get_today_weather_truth() -> str: return "맑음"
def normalize_weather_token(s: str) -> str:
    s = (s or "").replace("합니다","").replace("에요","").replace("예요","").strip()
    if any(x in s for x in ["맑","해","쨍"]): return "맑음"
    if any(x in s for x in ["흐","구름"]): return "흐림"
    if "비" in s: return "비"
    if "눈" in s: return "눈"
    if any(x in s for x in ["덥","무덥","더움","폭염"]): return "더움"
    if any(x in s for x in ["추","한파","쌀쌀"]): return "추움"
    return s

def score_today_weather(user_answer_std: str) -> int:
    gold = normalize_weather_token(get_today_weather_truth())
    pred = normalize_weather_token(user_answer_std)
    return 1 if (gold and pred and gold == pred) else 0

def get_current_location_truth() -> str: return "서울"
def location_match(user_loc: str, truth_loc: str) -> bool:
    return bool(user_loc) and (truth_loc in user_loc or user_loc in truth_loc)
def score_current_location(user_answer_std: str) -> int:
    gold = get_current_location_truth()
    return 1 if location_match(user_answer_std, gold) else 0

def _parse_any_mmdd(text: str):
    return _canonicalize_mmdd(text)

def score_seven_days_ago(user_answer_std: str) -> int:
    gold = _canonicalize_mmdd(_date_7days_ago_mmdd_ko())
    pred = _parse_any_mmdd(user_answer_std or "")
    return 1 if (gold and pred and gold == pred) else 0

def score_yesterday_activity(answer_std: str) -> int:
    return 1 if answer_std and len(answer_std.strip()) >= 2 else 0

# ===================== 11) Attention =====================
def pick_attention_question() -> str:
    return random.choice(["최근에 가족이나 친구들과 무슨 대화를 하셨나요?",
                          "요즘 어떻게 지내세요?"])

def score_attention(answer_std: str) -> int:
    if not answer_std or len(answer_std.strip()) < 3: return 0
    low = ["몰라","모르겠","없어","글쎄","대충","잘 기억이","기억 안","생각 안","나중에","귀찮"]
    if any(k in answer_std for k in low): return 0
    return 1

# ===================== 12) Diary (명령 전환 + 3문장 요약 + 저장 강화) =====================
DIARY_CHECK_KEYS = ["today_date","today_weather","current_location","date_7days_ago","yesterday_activity"]

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
def _pick_question(t: str, used: set[int]) -> tuple[str,int]:
    cands = [i for i in range(len(DIARY_QUESTION_TEMPLATES)) if i not in used]
    if not cands: cands = list(range(len(DIARY_QUESTION_TEMPLATES)))
    idx = random.choice(cands)
    return DIARY_QUESTION_TEMPLATES[idx].format(t=t), idx

def diary_rag_reminder(topic: str) -> str | None:
    for sess in reversed(diary_memory):
        for m in reversed(sess.get("messages", [])):
            if m.get("topic") == topic and m.get("role") == "user":
                snippet = m.get("content_std","") or m.get("content","")
                if snippet:
                    return f"지난번에 '{topic}'에 대해 \"{snippet[:40]}...\" 라고 말씀하셨어요."
    return None

def pick_diary_topics(k=3) -> list[str]:
    pool = BACKUP_MACRO_TOPICS[:]
    random.shuffle(pool)
    return pool[:k]

def run_diary_checklist_session(sess: dict) -> dict:
    print("\n📔 일기장 모드 시작! 먼저 다섯 가지를 확인해볼게요.\n")
    scores = {}

    def ask_and_log(question: str, tag: str):
        sess["messages"].append({"role":"assistant","content": question, "topic":"체크리스트", "ts": time.time()})
        log_event("assistant", content_raw=question, content_std=question, topic="체크리스트")

        a_raw = input(f"👵 사용자(답변) ← {question} ").strip()
        nrm = normalize_user_utterance(a_raw)
        a_std = nrm["standard"]
        ts_now = time.time()
        sess["messages"].append({"role":"user","content_raw": a_raw, "content_std": a_std, "topic":"체크리스트", "ts": ts_now})
        log_event("user", content_raw=a_raw, content_std=a_std, topic="체크리스트", meta={"tag": tag}, ts=ts_now)
        return a_std

    a1_std = ask_and_log("오늘이 몇월 며칠일까요?", "today_date")
    scores["today_date"] = score_today_date(a1_std)

    a2_std = ask_and_log("오늘 날씨는 어떤가요?", "today_weather")
    scores["today_weather"] = score_today_weather(a2_std)

    a3_std = ask_and_log("지금 어디에 계신가요?", "current_location")
    scores["current_location"] = score_current_location(a3_std)

    a4_std = ask_and_log("오늘로부터 7일 전은 몇월 며칠일까요?", "date_7days_ago")
    scores["date_7days_ago"] = score_seven_days_ago(a4_std)

    a5_std = ask_and_log("어제 뭐하셨어요?", "yesterday_activity")
    scores["yesterday_activity"] = score_yesterday_activity(a5_std)

    total = sum(scores.get(k,0) for k in DIARY_CHECK_KEYS)
    print(f"\n📊 체크 결과: {scores} (합계 {total}/5)\n")
    return {"scores": scores, "total": total, "asked_at": time.time()}

def _collect_topic_user_texts(sess: dict) -> dict:
    bucket = {t: [] for t in sess.get("topics", [])}
    for m in sess.get("messages", []):
        if m.get("role") == "user":
            t = m.get("topic")
            if t in bucket:
                txt = (m.get("content_std") or "").strip()
                if txt:
                    bucket[t].append(txt)
    return bucket

def _one_sentence_summary(topic: str, texts: list[str]) -> str:
    if not texts:
        return f"‘{topic}’에 대해서는 특별히 남긴 내용이 없었어요."
    joined = " / ".join(texts[-8:])
    prompt = (
        "아래 한국어 사용자 발화를 바탕으로, **정확히 한 문장**으로 핵심만 간결하게 요약해 주세요.\n"
        "새 정보 창작 금지, 존댓말, 30~60자.\n"
        f"[주제] {topic}\n[발화들]\n{joined}\n\n[출력] 한 문장:"
    )
    sent = ask_gpt(prompt, model="gpt-4o-mini", temperature=0.2, max_tokens=80,
                   response_format={"type":"text"})
    return (sent or "").strip() or f"‘{topic}’에 대해 한 문장으로 요약할 내용이 적었습니다."

def summarize_diary_session(sess: dict) -> list[dict]:
    topics = sess.get("topics", [])
    buckets = _collect_topic_user_texts(sess)
    summaries = []
    print("📝 오늘의 일기장 요약 (토픽당 1문장)")
    for t in topics:
        sent = _one_sentence_summary(t, buckets.get(t, []))
        print(f"- {sent}")
        summaries.append({"topic": t, "summary": sent})
    sess["diary_summaries"] = summaries
    return summaries

def run_diary_topic_conversations(sess: dict, k_topics=3):
    topics = pick_diary_topics(k_topics)
    sess["topics"] = topics
    print(f"🧩 오늘의 대화 주제: {', '.join(topics)}\n")
    remaining = topics[:]
    used_question_idx_map = {t:set() for t in topics}

    while remaining:
        t = remaining[0]
        print(f"— 주제 [{t}] —")
        memo = diary_rag_reminder(t)
        if memo:
            print(f"📝 리마인드: {memo}")
            log_event("assistant", content_raw=memo, content_std=memo, topic=t, meta={"type":"reminder"})

        q, qidx = _pick_question(t, used_question_idx_map[t])
        used_question_idx_map[t].add(qidx)

        ts_q = time.time()
        sess["messages"].append({"role":"assistant","content": q, "topic": t, "ts": ts_q})
        log_event("assistant", content_raw=q, content_std=q, topic=t, ts=ts_q, meta={"qidx": qidx})

        user_a_raw = input(f"👵 사용자(답변) ← {q} ").strip()
        nrm = normalize_user_utterance(user_a_raw)
        a_std = nrm["standard"]

        ts_a = time.time()
        sess["messages"].append({"role":"user","content_raw": user_a_raw, "content_std": a_std, "topic": t, "ts": ts_a})
        log_event("user", content_raw=user_a_raw, content_std=a_std, topic=t, ts=ts_a)

        if any(w in a_std for w in DIARY_STOP_WORDS):
            msg = "📝 알겠습니다. 오늘 일기장은 여기까지로 마칠게요.\n"
            print(msg)
            log_event("assistant", content_raw=msg, content_std=msg, topic=t, meta={"type":"diary_end"})
            sess["__force_end__"] = True
            break

        if any(w in a_std for w in DIARY_NEXT_WORDS) and len(remaining) > 1:
            next_topic = remaining[1]
            transition = f"좋아요. 그럼 다음으로 ‘{next_topic}’ 이야기도 해볼게요."
            print(f"🧠 시스템: {transition}\n")
            sess["messages"].append({"role":"assistant","content": transition, "topic": t, "ts": time.time()})
            log_event("assistant", content_raw=transition, content_std=transition, topic=t, meta={"type":"topic_switch","next":next_topic})
            remaining.pop(0)
            continue

        global FINETUNED_EMPATHY_MODEL
        if FINETUNED_EMPATHY_MODEL is None:
            FINETUNED_EMPATHY_MODEL = resolve_finetuned_model(EMPATHY_FT_JOB_ID)
        prompt = (
            f"{few_shot_empathy}\n\n사용자 발화: \"{a_std}\"\n"
            f"이전 대화 흐름: {' '.join(conversation_memory_std[-3:])}\n"
            f"공감 1문장 + 자연스러운 추가 질문 1문장 만들어주세요.\n응답:"
        )
        follow = ask_gpt(prompt, model=FINETUNED_EMPATHY_MODEL or "gpt-4o-mini",
                         temperature=0.6, max_tokens=180, response_format={"type":"text"})
        print(f"🧠 시스템: {follow}\n")
        sess["messages"].append({"role":"assistant","content": follow, "topic": t, "ts": time.time()})
        log_event("assistant", content_raw=follow, content_std=follow, topic=t, meta={"type":"followup"})

        if len(used_question_idx_map[t]) >= len(DIARY_QUESTION_TEMPLATES) and len(remaining) > 1:
            next_topic = remaining[1]
            transition = f"그럼 ‘{next_topic}’ 이야기도 이어서 나눠볼까요?"
            print(f"🧠 시스템: {transition}\n")
            sess["messages"].append({"role":"assistant","content": transition, "topic": t, "ts": time.time()})
            log_event("assistant", content_raw=transition, content_std=transition, topic=t, meta={"type":"topic_switch","next":next_topic})
            remaining.pop(0)

def start_diary_mode():
    global diary_id_counter, DIARY_MODE
    DIARY_MODE = True
    sess = {
        "diary_id": f"diary_{diary_id_counter}",
        "started_at": time.time(),
        "scores": {},
        "score_total": 0,
        "messages": [],
        "topics": [],
        "diary_summaries": []
    }
    check_res = run_diary_checklist_session(sess)
    sess["scores"] = check_res["scores"]; sess["score_total"] = check_res["total"]
    run_diary_topic_conversations(sess, k_topics=3)
    summarize_diary_session(sess)
    sess["ended_at"] = time.time()
    diary_memory.append(sess)
    diary_id_counter += 1
    DIARY_MODE = False
    print(f"✅ 일기장 세션 저장 완료: {sess['diary_id']} (체크 {sess['score_total']}/5, 메시지 {len(sess['messages'])}개)\n")

# ===================== 13) 일반 대화 =====================
def generate_question(user_input_std: str):
    global FINETUNED_EMPATHY_MODEL
    if FINETUNED_EMPATHY_MODEL is None:
        FINETUNED_EMPATHY_MODEL = resolve_finetuned_model(EMPATHY_FT_JOB_ID)
    prompt = (
        f"{few_shot_empathy}\n\n사용자 발화: \"{user_input_std}\"\n"
        f"이전 대화 흐름: {' '.join(conversation_memory_std[-3:])}\n"
        f"공감 1문장 + 자연스러운 일상 질문 1문장을 만들어주세요.\n응답:"
    )
    return ask_gpt(prompt, model=FINETUNED_EMPATHY_MODEL or "gpt-4o-mini",
                   temperature=0.6, max_tokens=180, response_format={"type":"text"})

# ===================== 14) CSV Export =====================
def export_fact_memory_csv(path: str = "fact_memory.csv") -> str:
    if not fact_memory:
        print("fact_memory가 비어 있어요. 파일을 만들지 않았습니다.")
        return ""
    header = set()
    for f in fact_memory:
        header.update(f.keys())
    header = list(header)
    preferred = ["id","claim_text","summary","entities","type",
                 "topic","topic_confidence",
                 "time_reference","relative_offset_days","timestamp",
                 "original_utterance_raw","original_utterance_std",
                 "decision","consistency_binary"]
    ordered = [h for h in preferred if h in header] + [h for h in header if h not in preferred]
    p = Path(path)
    with p.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=ordered)
        writer.writeheader()
        for row in fact_memory:
            r = dict(row)
            if isinstance(r.get("entities"), list):
                r["entities"] = ", ".join(map(str, r["entities"]))
            if r.get("timestamp"):
                r["timestamp"] = _ts(r["timestamp"])
            writer.writerow({k: r.get(k, "") for k in ordered})
    print(f"✅ fact_memory를 CSV로 저장했습니다: {p.resolve()}")
    return str(p.resolve())

def export_diary_memory_csv(sessions_csv_path: str, messages_csv_path: str):
    """
    diary_memory(list of sess dict)를 두 개 CSV로 저장:
      1) sessions_csv_path: 세션 요약 (5문항 점수 0/1 + 합계 + 토픽 + 요약문)
      2) messages_csv_path: 모든 메시지 (role, topic, content_raw, content_std, ts, meta)
    """
    import csv, json

    # ---- 1) 세션 CSV ----
    session_cols = [
        "id", "started_at",
        "score_today_date", "score_today_weather", "score_current_location",
        "score_date_7days_ago", "score_yesterday_activity",
        "score_total",
        "topics",
        "summaries_text",
    ]
    # ✅ 엑셀 한글 호환: utf-8-sig
    with open(sessions_csv_path, "w", newline="", encoding="utf-8-sig") as sf:
        w = csv.DictWriter(sf, fieldnames=session_cols)
        w.writeheader()
        for sess in diary_memory:
            sid = sess.get("id")
            started_at = sess.get("started_at")

            sc = sess.get("scores", {}) or {}
            s_today   = int(sc.get("today_date", 0) or 0)
            s_weather = int(sc.get("today_weather", 0) or 0)
            s_loc     = int(sc.get("current_location", 0) or 0)
            s_7ago    = int(sc.get("date_7days_ago", 0) or 0)
            s_yest    = int(sc.get("yesterday_activity", 0) or 0)
            s_total   = s_today + s_weather + s_loc + s_7ago + s_yest

            topics = sess.get("topics", []) or []
            sums   = sess.get("diary_summaries", []) or []
            summaries_text = " | ".join(
                f"[{x.get('topic','')}] {x.get('summary','')}" for x in sums
            )

            w.writerow({
                "id": sid,
                "started_at": started_at,
                "score_today_date": s_today,
                "score_today_weather": s_weather,
                "score_current_location": s_loc,
                "score_date_7days_ago": s_7ago,
                "score_yesterday_activity": s_yest,
                "score_total": s_total,
                "topics": ", ".join(topics),
                "summaries_text": summaries_text,
            })

    # ---- 2) 메시지 CSV ----
    msg_cols = ["session_id", "ts", "role", "topic", "content_raw", "content_std", "meta_json"]
    # ✅ 엑셀 한글 호환: utf-8-sig
    with open(messages_csv_path, "w", newline="", encoding="utf-8-sig") as mf:
        w = csv.DictWriter(mf, fieldnames=msg_cols)
        w.writeheader()
        for sess in diary_memory:
            sid = sess.get("id")
            msgs = sess.get("messages", []) or []
            for m in msgs:
                raw = m.get("content_raw")
                std = m.get("content_std")
                base = m.get("content")  # assistant 질문만 content에 있을 수도 있음
                if raw is None and base is not None:
                    raw = base
                if std is None and base is not None:
                    std = base

                w.writerow({
                    "session_id": sid,
                    "ts": m.get("ts"),
                    "role": m.get("role"),
                    "topic": m.get("topic"),
                    "content_raw": raw or "",
                    "content_std": std or "",
                    "meta_json": json.dumps(m.get("meta", {}), ensure_ascii=False),
                })

    return sessions_csv_path, messages_csv_path



# ===================== 15) Main Chat Loop =====================
def chat_loop():
    print("\n🧠 치매 대화 시스템 시작 (종료: '그만') — 일기장 저장/날짜 채점 FIX 반영")
    print("처음에 주의력(Attention) 질문 하나를 드릴게요.\n")

    att_q = pick_attention_question()
    log_event("assistant", content_raw=att_q, content_std=att_q, topic="주의력")

    att_ans_raw = input(f"👵 사용자: {att_q} ").strip()
    nrm = normalize_user_utterance(att_ans_raw)
    att_ans_std = nrm["standard"]
    log_event("user", content_raw=att_ans_raw, content_std=att_ans_std, topic="주의력")

    att_score = score_attention(att_ans_std)
    print(f"📊 Attention 점수: {att_score}")

    att_sess = {
        "diary_id": f"attention_{int(time.time())}",
        "started_at": time.time(),
        "attention_question": att_q,
        "attention_answer_raw": att_ans_raw,
        "attention_answer_std": att_ans_std,
        "attention": att_score,
        "messages": []
    }
    diary_memory.append(att_sess)

    while True:
        user_input_raw = input("👵 사용자: ").strip()
        if user_input_raw.lower() == "그만":
            print("\n시스템을 종료합니다. 안녕히 계세요!")
            log_event("user", content_raw="그만", content_std="그만", topic="")
            break
        if not user_input_raw:
            continue

        nrm = normalize_user_utterance(user_input_raw)
        user_input_std = nrm["standard"]

        conversation_memory_raw.append(user_input_raw)
        conversation_memory_std.append(user_input_std)

        log_event("user", content_raw=user_input_raw, content_std=user_input_std, topic=CONTEXT_TOPIC_LABEL or "")

        if "일기" in user_input_std or "일기" in user_input_raw:
            start_diary_mode()
            continue

        check_memory_consistency(user_input_std, user_input_raw, nrm)

        reply = generate_question(user_input_std)
        print(f"\n🧠 시스템: {reply}\n")
        log_event("assistant", content_raw=reply, content_std=reply, topic=CONTEXT_TOPIC_LABEL or "")
        time.sleep(0.3)
def export_fact_memory_json(filepath: str = "fact_memory.json"):
    import json
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(fact_memory, f, ensure_ascii=False, indent=2)
    print(f"✅ fact_memory를 {filepath}로 저장했습니다.")


def export_diary_memory_json(filepath: str = "diary_memory.json"):
    import json
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(diary_memory, f, ensure_ascii=False, indent=2)
    print(f"✅ diary_memory를 {filepath}로 저장했습니다.")
