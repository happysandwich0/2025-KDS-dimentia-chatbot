import os, json, time, random, re, csv
import numpy as np
import pandas as pd
import asyncio
from datetime import datetime, timedelta
from pathlib import Path

try:
    from zoneinfo import ZoneInfo
    _KST = ZoneInfo("Asia/Seoul")
except Exception:
    _KST = None

try:
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("jhgan/ko-sbert-nli")
except Exception:
    def embedder_encode(texts):
        return np.zeros((len(texts), 768), dtype="float32")
    class DummyEmbedder:
        def encode(self, texts): return embedder_encode(texts)
    embedder = DummyEmbedder()

from config import (
    CONFIG_TODAY_OVERRIDE, CONTEXT_TOPIC_LABEL, CONTEXT_TOPIC_CONF,
    conversation_memory_std, conversation_memory_raw,
    fact_memory, fact_embeddings, faiss_index,
    memory_score, RECENT_CONSISTENCY_BIN,
    conversation_log, _conv_idx,
    diary_memory, diary_id_counter,
    BACKUP_MACRO_TOPICS,
    DIARY_CHECK_KEYS, CHECKS,
    DIARY_QUESTION_TEMPLATES
)
from llm_utils import (
    async_ask_gpt, async_ask_gpt_json_batch, get_finetuned_model_or_default, FEW_SHOT_EMPATHY,
    DIALECT_SYSTEM_PROMPT, DIALECT_JSON_SCHEMA, TOOL_DEFINITIONS
)

# ===================== 1) Utilities =====================
TIME_WORDS = ["오늘","어제","내일","지금","방금","저녁","아침","점심",
              "월요일","화요일","수요일","목요일","금요일","토요일","일요일",
              "이번 주","지난 주","다음 주","이번 달","지난 달","다음 달"]

def _ts(ts):
    """타임스탬프를 문자열로 변환"""
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ""

def _now_kst_dt():
    """오늘 날짜 (KST 기준 또는 오버라이드)"""
    if CONFIG_TODAY_OVERRIDE:
        try:
            y, m, d = map(int, CONFIG_TODAY_OVERRIDE.split("-"))
            return datetime(y, m, d, 12, 0, 0)
        except Exception:
            pass
    try:
        return datetime.now(_KST) if _KST else datetime.now()
    except Exception:
        return datetime.now()

def _today_mmdd_ko():
    """오늘 날짜 (월 일)"""
    now = _now_kst_dt()
    return f"{now.month}월 {now.day}일"

def _date_7days_ago_mmdd_ko():
    """7일 전 날짜 (월 일)"""
    d = _now_kst_dt() - timedelta(days=7)
    return f"{d.month}월 {d.day}일"

def _canonicalize_mmdd(text):
    """텍스트에서 월/일 추출"""
    if not text: return None
    s = re.sub(r"\s+", " ", text).strip()
    m = re.search(r'(\d{1,2})\s*월\s*(\d{1,2})\s*일', s)
    if m: return int(m.group(1)), int(m.group(2))
    m = re.search(r'(\d{4})[./-](\d{1,2})[./-](\d{1,2})', s)
    if m: return int(m.group(2)), int(m.group(3))
    m = re.search(r'(\d{1,2})[./-](\d{1,2})', s)
    if m: return int(m.group(1)), int(m.group(2))
    return None

def _safe_json_loads(raw, fallback = None):
    """ JSON 파싱"""
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

# ===================== 2) Logging / Memory =====================
def log_event(role,
              content_raw = None,
              content_std = None,
              topic = None,
              meta = None,
              ts = None):
    """모든 대화 턴을 일관 포맷으로 기록"""
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

def conversation_log_dataframe():
    """전체 대화 DataFrame 반환"""
    if not conversation_log:
        return pd.DataFrame(columns=["idx","ts","ts_str","role","topic","content_raw","content_std","meta"])
    df = pd.DataFrame(conversation_log)
    return df.sort_values("idx", ignore_index=True)

# ===================== 3) Dialect → Standard =====================
DIALECT_MARKERS = ["데이","카이","아이가","아입니꺼","하께","쿠다","머시","카노","카더라","하믄","그라믄",
                    "했심더","했데이","하이소","하입니더","무했노","마","예","그카이","고마","그라제"]

def _looks_like_dialect(s):
    """사투리 마커 포함 여부 확인"""
    s = s or ""
    return any(tok in s for tok in DIALECT_MARKERS)

def normalize_user_utterance(user_text):
    """사투리를 표준어로 변환"""
    model_to_use = get_finetuned_model_or_default("DIALECT_FT_JOB_ID")
    prompt = f"{DIALECT_SYSTEM_PROMPT}\n\n입력:\n\"\"\"\n{user_text}\n\"\"\"\n\n{DIALECT_JSON_SCHEMA}"
    
    # 동기 함수를 비동기로 래핑하여 호출
    loop = asyncio.get_event_loop()
    
    def sync_call(p, m):
        return asyncio.run(async_ask_gpt(p, model=m, temperature=0.0, max_tokens=200, response_format={"type":"json_object"}))

    def minimalist_sync_call(p, m):
        return asyncio.run(async_ask_gpt(p, model=m, temperature=0.0, max_tokens=120, response_format={"type":"text"}))

    raw_response = sync_call(prompt, model_to_use)
    data = _safe_json_loads(raw_response['choices'][0]['message']['content'], fallback={})
    
    standard = (data.get("standard") or "").strip()
    is_dialect = bool(data.get("is_dialect", False))
    conf = float(data.get("confidence", 0.0) or 0.0)

    if (not standard) or _looks_like_dialect(standard):
        raw_response2 = sync_call(prompt, "gpt-4o-mini")
        data2 = _safe_json_loads(raw_response2['choices'][0]['message']['content'], fallback={})
        standard2 = (data2.get("standard") or "").strip()
        if standard2 and not _looks_like_dialect(standard2):
            standard = standard2
            is_dialect = bool(data2.get("is_dialect", is_dialect))
            conf = float(data2.get("confidence", conf) or conf)

    if not standard or _looks_like_dialect(standard):
        minimalist_prompt = (
            "다음 문장을 한국어 표준어(존댓말)로 한 문장으로만 바꿔줘. "
            "JSON 없이 결과 문장만 출력해.\n\n"
            f"문장: {user_text}"
        )
        std3 = minimalist_sync_call(minimalist_prompt, "gpt-4o-mini")
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

# ===================== 4) Topic Labeling =====================
def _window_text(history_std, current_std, k=6):
    """대화 맥락 윈도우 생성"""
    tail = " ".join(history_std[-k:])
    return (tail + " " + (current_std or "")).strip()

async def infer_context_topic_label(history_std, current_std):
    """LLM으로 맥락 토픽 추론 (비동기)"""
    ctx = _window_text(history_std, current_std, k=6)
    prompt = (
        "다음 한국어 대화 맥락의 전반 주제를 1~3어절의 일반명사/짧은 구로 요약하세요.\n"
        "세부어/희귀어 금지, 새 정보 창작 금지.\n"
        "JSON으로만 답: {\"label\":\"...\",\"confidence\":0.0}\n\n"
        f"[대화 맥락]\n{ctx}\n"
    )
    raw_response = await async_ask_gpt(prompt, model="gpt-4o-mini", temperature=0.0, max_tokens=120,
                  response_format={"type":"json_object"})
    
    if raw_response.get("error"):
        return "일상", 0.0

    raw = raw_response['choices'][0]['message']['content']
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

def smooth_context_topic(new_label, new_conf,
                         prev_label, prev_conf,
                         min_change_conf = 0.60,
                         drift_guard = 0.15):
    """토픽 스무딩 로직"""
    if not prev_label:
        return new_label, new_conf, True
    if new_label == prev_label:
        fused = max(new_conf, (new_conf + prev_conf) / 2)
        return prev_label, fused, False
    if (new_conf >= min_change_conf) and ((new_conf - prev_conf) >= drift_guard):
        return new_label, new_conf, True
    return prev_label, prev_conf, False

# ===================== 5) Fact Extract / Consistency (통합 및 배치) =====================

async def check_memory_consistency_and_reply(user_input_raw):
    """사용자 발화 입력 → 교정/추출/검증/응답 생성을 Tool Call과 배치 처리로 통합 수행"""
    global CONTEXT_TOPIC_LABEL, CONTEXT_TOPIC_CONF

    messages = [
        {"role": "system", "content": "당신은 치매 환자를 위한 공감형 대화 챗봇입니다. 사용자 발화를 받으면, 표준어 교정, 사실 추출, 그리고 최종 응답 생성 도구를 단 한 번의 호출로 사용하세요. 사실 추출/검증은 메모리 일관성 검사에 필요하며, 최종 응답 생성에 핵심적인 메모리 요약만 전달합니다."},
        {"role": "user", "content": f"사용자 발화: {user_input_raw}"}
    ]
    
    response = await async_ask_gpt(
        messages=messages,
        model="gpt-4o",
        temperature=0.0,
        tools=TOOL_DEFINITIONS,
        tool_choice="auto"
    )
    
    if response.get("error"):
        return f"죄송합니다. 처리 중 오류가 발생했습니다: {response['error']}"

    response_message = response['choices'][0]['message']
    
    if not response_message.get("tool_calls"):
        return response_message.get("content") or "말씀을 이해하지 못했어요. 다시 말씀해주시겠어요?"
    
    tool_calls = response_message.get("tool_calls", [])
    function_results = {}
    
    nrm_meta = None
    user_input_std = None
    claims_to_store = []
    
    for tool_call in tool_calls:
        function_name = tool_call['function']['name']
        arguments = json.loads(tool_call['function']['arguments'])

        if function_name == "normalize_utterance":
            # 사투리 교정
            nrm_result = normalize_user_utterance(arguments.get("user_text"))
            nrm_meta = {k: nrm_result[k] for k in ["is_dialect", "confidence"]}
            user_input_std = nrm_result["standard"]
            function_results[tool_call['id']] = json.dumps(nrm_result, ensure_ascii=False)
            
        elif function_name == "extract_claims_from_utterance":
            # 사실 추출
            claims_to_store = _extract_claims_from_utterance_sync(
                arguments.get("user_input_std"), 
                conversation_memory_std, 
                arguments.get("original_raw"), 
                arguments.get("nrm_meta")
            )
            claims_summary = f"총 {len(claims_to_store)}개의 사실을 추출했습니다."
            function_results[tool_call['id']] = json.dumps({"summary": claims_summary}, ensure_ascii=False)
            
        elif function_name == "generate_response":
            pass

    if not user_input_std:
        nrm = normalize_user_utterance(user_input_raw) 
        user_input_std = nrm["standard"]
        nrm_meta = {k: nrm[k] for k in ["is_dialect", "confidence"]}

    fact_check_report = ""
    if claims_to_store:
        new_topic_label, new_topic_conf = await infer_context_topic_label(conversation_memory_std, user_input_std)
        CONTEXT_TOPIC_LABEL, CONTEXT_TOPIC_CONF, _ = smooth_context_topic(
            new_topic_label, new_topic_conf, CONTEXT_TOPIC_LABEL, CONTEXT_TOPIC_CONF,
            min_change_conf=0.60, drift_guard=0.15
        )
        for nf in claims_to_store:
            nf["topic"] = CONTEXT_TOPIC_LABEL or "일상"
            nf["topic_confidence"] = float(CONTEXT_TOPIC_CONF)
            
        _store_extracted_facts(claims_to_store)
        
        results = await track_facts_batch(claims_to_store)
        fact_check_report = _process_batch_consistency_results(results)
    else:
        new_topic_label, new_topic_conf = await infer_context_topic_label(conversation_memory_std, user_input_std)
        CONTEXT_TOPIC_LABEL, CONTEXT_TOPIC_CONF, _ = smooth_context_topic(
            new_topic_label, new_topic_conf, CONTEXT_TOPIC_LABEL, CONTEXT_TOPIC_CONF,
            min_change_conf=0.60, drift_guard=0.15
        )
        fact_check_report = "추출된 새로운 사실이 없습니다."

    current_messages = messages
    current_messages.append(response_message)
    for tool_call in tool_calls:
        if tool_call['id'] in function_results:
            current_messages.append({
                "role": "tool",
                "tool_call_id": tool_call['id'],
                "name": tool_call['function']['name'],
                "content": function_results[tool_call['id']]
            })
    
    final_response_call = next((tc for tc in tool_calls if tc['function']['name'] == "generate_response"), None)
    
    if final_response_call:
        current_messages.append({
            "role": "tool",
            "tool_call_id": final_response_call['id'],
            "name": final_response_call['function']['name'],
            "content": json.dumps({
                "user_input_std": user_input_std,
                "fact_check_result": fact_check_report
            }, ensure_ascii=False)
        })
        
        final_response = await async_ask_gpt(
            messages=current_messages,
            model="gpt-4o",
            temperature=0.7,
            max_tokens=220,
        )
        
        return final_response['choices'][0]['message']['content']
    
    else:
        return await empathetic_reply(user_input_std)


def _extract_claims_from_utterance_sync(user_input_std, recent_history_std, original_raw, nrm_meta):
    """기존 LLM 호출을 포함한 사실 추출 동기 함수 (내부 사용)"""
    global fact_id_counter
    formatted_history = []
    for i, msg in enumerate(recent_history_std[-6:]):
        role = "사용자(표준어)" if i % 2 == 0 else "시스템"
        formatted_history.append(f"{role}: {msg}")
    recent_history_str = "\n".join(formatted_history)
    prompt = f"""
        최근 대화 기록 :
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
    # async_ask_gpt를 동기적으로 호출
    loop = asyncio.get_event_loop()
    raw_response = asyncio.run(async_ask_gpt(
        prompt=prompt,
        model="gpt-4o-mini",
        temperature=0.2,
        response_format={"type":"json_object"},
        max_tokens=2048
    ))
    
    json_str = raw_response['choices'][0]['message']['content']

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

def _store_extracted_facts(extracted_facts):
    """추출된 사실 저장 및 임베딩 생성"""
    for fact in extracted_facts:
        fact_ts = time.time()
        rod = fact.get("relative_offset_days")
        if isinstance(rod, (int,float)):
            fact_ts = (_now_kst_dt() + timedelta(days=rod)).timestamp()
        fact["timestamp"] = fact_ts
        fact_memory.append(fact)
        emb = embedder.encode([fact["claim_text"]])[0].astype("float32")
        fact_embeddings.append(emb)
        faiss_index.add(np.array([emb]))

def _find_related_old_facts(new_claim_embedding, top_k=5):
    """FAISS로 관련 기존 사실 찾기"""
    if not fact_embeddings: return []
    distances, indices = faiss_index.search(np.array([new_claim_embedding]).astype("float32"),
                                            min(top_k, len(fact_embeddings)))
    return [fact_memory[i] for i in indices[0] if i != -1]

async def track_facts_batch(new_facts):
    """N개의 신규 사실에 대해 배치로 LLM을 호출하여 일관성을 검사"""
    batch_data = []
    
    for nf in new_facts:
        emb = embedder.encode([nf["claim_text"]])[0]
        olds = [f for f in _find_related_old_facts(emb) if f["id"] != nf["id"]]
        
        related_old_facts_str = "\n".join([
            f"ID:{f['id']}, 사실:{f['claim_text']}, 시간:{_ts(f['timestamp'])}"
            for f in olds
        ]) or "없음."
        
        batch_data.append({
            "new_fact_id": nf['id'],
            "new_fact_claim": nf['claim_text'],
            "related_old_facts": related_old_facts_str
        })

    prompt = f"""
        당신은 메모리 일관성 검사 에이전트입니다.
        제공된 JSON 배열을 분석하여 각 신규 사실에 대한 일관성 판단 결과를 JSON 배열로 반환하세요.
        N개의 입력에 대해 N개의 출력을 보장해야 합니다.
        출력은 다음 형식의 JSON 배열로만 구성되어야 합니다: 
        [{{ "new_fact_id": "...", "decision": "CONSISTENT|UPDATE|CONTRADICTION|NEW" }}, ...]

        [배치 입력 데이터 (신규 사실 및 관련 기존 사실)]
        {json.dumps(batch_data, ensure_ascii=False, indent=2)}
        """

    response = await async_ask_gpt_json_batch(
        prompt=prompt, 
        model="gpt-4o",
        temperature=0.0
    )
    
    if response.get("error"):
        return [{"new_fact_id": nf['id'], "decision": "ERROR"} for nf in new_facts]
        
    try:
        results = json.loads(response['choices'][0]['message']['content'])
        return results if isinstance(results, list) else []
    except Exception:
        return [{"new_fact_id": nf['id'], "decision": "ERROR"} for nf in new_facts]

def _process_batch_consistency_results(results):
    """배치 일관성 검사 결과를 처리하고 점수 업데이트"""
    global memory_score
    summary = {"NEW": 0, "CONSISTENT": 0, "UPDATE": 0, "CONTRADICTION": 0, "ERROR": 0}
    
    fact_map = {f['id']: f for f in fact_memory if f.get('decision') is None}
    
    for r in results:
        fact_id = r.get("new_fact_id")
        dval = r.get("decision", "ERROR").upper()
        summary[dval] = summary.get(dval, 0) + 1
        
        cb = 1 if dval in ("CONSISTENT","UPDATE","NEW") else 0
        RECENT_CONSISTENCY_BIN.append(cb)
        
        if fact_id in fact_map:
            fact = fact_map[fact_id]
            fact["decision"] = dval
            fact["consistency_binary"] = cb
        
        if dval == "CONTRADICTION":
            memory_score = max(0, memory_score - 15)
        elif dval == "UPDATE":
            memory_score = min(100, memory_score + 5)
        elif dval in ("CONSISTENT","NEW"):
            memory_score = min(100, memory_score + 1)
            
    report = (
        f"✅ 메모리 일관성 검사 완료 (점수: {memory_score}/100). "
        f"새로운 사실 {summary['NEW']}개, 일관 {summary['CONSISTENT']}개, 갱신 {summary['UPDATE']}개, 상충 {summary['CONTRADICTION']}개."
    )
    return report

# ===================== 6) Scoring =====================
SCORE_FN = {}

def score_today_date(user_answer_std):
    gold = _canonicalize_mmdd(_today_mmdd_ko())
    pred = _canonicalize_mmdd(user_answer_std or "")
    return 1 if (gold and pred and gold == pred) else 0

def normalize_weather_token(s):
    s = (s or "").replace("합니다","").replace("에요","").replace("예요","").strip()
    if any(x in s for x in ["맑","해","쨍"]): return "맑음"
    if any(x in s for x in ["흐","구름"]): return "흐림"
    if "비" in s: return "비"
    if "눈" in s: return "눈"
    if any(x in s for x in ["덥","무덥","더움","폭염"]): return "더움"
    if any(x in s for x in ["추","한파","쌀쌀"]): return "추움"
    return s

def score_today_weather(user_answer_std):
    gold = normalize_weather_token("맑음")
    pred = normalize_weather_token(user_answer_std)
    return 1 if (gold and pred and gold == pred) else 0

def score_current_location(user_answer_std):
    gold = "서울"
    return 1 if (user_answer_std and (gold in user_answer_std or user_answer_std in gold)) else 0

def score_seven_days_ago(user_answer_std):
    gold = _canonicalize_mmdd(_date_7days_ago_mmdd_ko())
    pred = _canonicalize_mmdd(user_answer_std or "")
    return 1 if (gold and pred and gold == pred) else 0

def score_yesterday_activity(answer_std):
    return 1 if answer_std and len(answer_std.strip()) >= 2 else 0

def pick_attention_question():
    return random.choice(["최근에 가족이나 친구들과 무슨 대화를 하셨나요?",
                          "요즘 어떻게 지내세요?"])

def score_attention(answer_std):
    if not answer_std or len(answer_std.strip()) < 3: return 0
    low = ["몰라","모르겠","없어","글쎄","대충","잘 기억이","기억 안","생각 안","나중에","귀찮"]
    if any(k in answer_std for k in low): return 0
    return 1

SCORE_FN = {
    "today_date":         score_today_date,
    "today_weather":      score_today_weather,
    "current_location":   score_current_location,
    "date_7days_ago":     score_seven_days_ago,
    "yesterday_activity": score_yesterday_activity,
}

# ===================== 7) Diary Logic =====================
def pick_diary_topics(k=3):
    """랜덤으로 일기 주제 선택"""
    pool = BACKUP_MACRO_TOPICS[:]
    random.shuffle(pool)
    return pool[:k]

def _pick_question(t, used):
    """주제에 맞는 질문 선택 및 인덱스 반환"""
    cands = [i for i in range(len(DIARY_QUESTION_TEMPLATES)) if i not in used]
    if not cands: cands = list(range(len(DIARY_QUESTION_TEMPLATES)))
    idx = random.choice(cands)
    return DIARY_QUESTION_TEMPLATES[idx].format(t=t), idx

def diary_rag_reminder(topic):
    """일기장 RAG 리마인더 생성"""
    for sess in reversed(diary_memory):
        for m in reversed(sess.get("messages", [])):
            if m.get("topic") == topic and m.get("role") == "user":
                snippet = m.get("content_std","") or m.get("content","")
                if snippet:
                    return f"지난번에 '{topic}'에 대해 \"{snippet[:40]}...\" 라고 말씀하셨어요."
    return None

async def summarize_diary_session(sess):
    """일기장 세션 요약 (토픽당 1문장)"""
    topics = sess.get("topics", [])
    buckets = _collect_topic_user_texts(sess)
    summaries = []
    
    tasks = []
    for t in topics:
        tasks.append(_one_sentence_summary(t, buckets.get(t, [])))
    
    results = await asyncio.gather(*tasks)
    
    for t, sent in zip(topics, results):
        summaries.append({"topic": t, "summary": sent})
        
    sess["diary_summaries"] = summaries
    return summaries

def _collect_topic_user_texts(sess):
    """세션에서 주제별 사용자 발화 수집"""
    bucket = {t: [] for t in sess.get("topics", [])}
    for m in sess.get("messages", []):
        if m.get("role") == "user":
            t = m.get("topic")
            if t in bucket:
                txt = (m.get("content_std") or "").strip()
                if txt:
                    bucket[t].append(txt)
    return bucket

async def _one_sentence_summary(topic, texts):
    """LLM으로 한 문장 요약 (비동기)"""
    if not texts:
        return f"‘{topic}’에 대해서는 특별히 남긴 내용이 없었어요."
    joined = " / ".join(texts[-8:])
    prompt = (
        "아래 한국어 사용자 발화를 바탕으로, **정확히 한 문장**으로 핵심만 간결하게 요약해 주세요.\n"
        "새 정보 창작 금지, 존댓말, 30~60자.\n"
        f"[주제] {topic}\n[발화들]\n{joined}\n\n[출력] 한 문장:"
    )
    raw_response = await async_ask_gpt(prompt, model="gpt-4o-mini", temperature=0.2, max_tokens=80,
                   response_format={"type":"text"})
    
    sent = raw_response['choices'][0]['message']['content'] if not raw_response.get("error") else ""
    return (sent or "").strip() or f"‘{topic}’에 대해 한 문장으로 요약할 내용이 적었습니다."

async def classify_consent(user_std, topic):
    """사용자 답변을 1(계속)/0(다음)으로 분류 (비동기)"""
    ctx = " | ".join(conversation_memory_std[-3:])
    prompt = (
        "당신은 화제 지속 의사 분류기입니다.\n"
        "규칙:\n- 사용자가 주제에 대해 더 얘기하고 싶으면 1,\n"
        "- 그만하거나 다른 주제로 넘어가고 싶으면 0.\n"
        "- 다른 출력 금지.\n\n"
        f"주제: {topic}\n"
        f"최근 맥락: {ctx}\n"
        f"사용자 최신 발화: {user_std}\n\n"
        "출력: 1 또는 0"
    )
    raw_response = await async_ask_gpt(prompt=prompt, model="gpt-4o-mini",
                  temperature=0.0, max_tokens=4, response_format={"type":"text"})
    
    out = raw_response['choices'][0]['message']['content'] if not raw_response.get("error") else ""
    return (out or "").strip().startswith("1")

# ===================== 8) Empathy / Question Generation =====================
async def empathetic_reply(user_text):
    """공감 1문장 + 구체적 질문 1개 (비동기)"""
    model_to_use = get_finetuned_model_or_default("EMPATHY_FT_JOB_ID")
    prompt = (
        "역할: 공감형 노년 맞춤 대화 코치.\n"
        "규칙:\n- 1문장 공감\n- 이어서 구체적이고 답하기 쉬운 질문 딱 1개\n- 전체 2~3문장, 존댓말\n"
        f"{FEW_SHOT_EMPATHY}\n\n"
        f"사용자 발화: \"{user_text.strip()}\"\n응답:"
    )
    response = await async_ask_gpt(
        prompt=prompt,
        model=model_to_use,
        temperature=0.7,
        max_tokens=220,
        response_format={"type":"text"}
    )
    out = response['choices'][0]['message']['content'] if not response.get("error") else ""
    return out.strip() or "말씀을 들으니 마음이 쓰이네요. 혹시 그때 어떤 상황이었는지 알려주실 수 있을까요?"

async def empathy_only(user_text):
    """질문 없는 공감만 (비동기)"""
    model_to_use = get_finetuned_model_or_default("EMPATHY_FT_JOB_ID")
    prompt = (
        "역할: 공감형 노년 맞춤 대화 코치.\n"
        "규칙:\n- 사용자의 감정을 한 문장으로 공감만 한다.\n"
        "- 질문이나 요청 금지. 오직 공감 1문장.\n"
        f"{FEW_SHOT_EMPATHY}\n\n"
        f"사용자 발화: \"{user_text.strip()}\"\n출력:"
    )
    response = await async_ask_gpt(
        prompt=prompt, 
        model=model_to_use,
        temperature=0.6, 
        max_tokens=120, 
        response_format={"type":"text"}
    )
    out = response['choices'][0]['message']['content'] if not response.get("error") else ""
    return (out or "말씀을 들으니 마음이 쓰이네요.").strip()

# ===================== 9) CSV Export =====================
def export_fact_memory_csv(path = "fact_memory.csv"):
    """fact_memory를 CSV로 저장"""
    if not fact_memory: return ""
    
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
    return str(p.resolve())

def export_diary_memory_csv(sessions_csv_path, messages_csv_path):
    """diary_memory를 세션 및 메시지 두 개 CSV로 저장"""
    
    # ---- 1) 세션 CSV ----
    session_cols = [
        "id", "started_at",
        "score_today_date", "score_today_weather", "score_current_location",
        "score_date_7days_ago", "score_yesterday_activity",
        "score_total",
        "topics",
        "summaries_text",
    ]
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
    with open(messages_csv_path, "w", newline="", encoding="utf-8-sig") as mf:
        w = csv.DictWriter(mf, fieldnames=msg_cols)
        w.writeheader()
        for sess in diary_memory:
            sid = sess.get("id")
            msgs = sess.get("messages", []) or []
            for m in msgs:
                raw = m.get("content_raw")
                std = m.get("content_std")
                base = m.get("content")
                if raw is None and base is not None: raw = base
                if std is None and base is not None: std = base
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