import os, sys, time, json, random
import streamlit as st
import pandas as pd
import asyncio
from pathlib import Path

from config import (
    CHECKS, SCORE_FN, DIARY_CHECK_KEYS, BACKUP_MACRO_TOPICS,
    STOP_WORDS, NEXT_WORDS, CONSENT_PROMPT
)
from main import (
    normalize_user_utterance, score_attention, empathetic_reply, empathy_only,
    log_event, conversation_log_dataframe, check_memory_consistency_and_reply,
    pick_diary_topics, summarize_diary_session, pick_attention_question,
    classify_consent, export_fact_memory_csv, export_diary_memory_csv
)

from config import (
    conversation_memory_std, conversation_memory_raw,
    conversation_log, fact_memory, diary_memory, CONTEXT_TOPIC_LABEL
)

# ===================== 1) UI 기본 설정 및 상태 초기화 =====================
st.set_page_config(page_title="당신의 소중한 말벗 또랑이", page_icon="🍊", layout="wide")
st.markdown("## 🍊 당신의 소중한 말벗, 또랑이")
st.write("‘일기’라고 말하면 체크리스트 → 점수 계산 → 주제 3개로 진행돼요.")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role":"assistant","content":"안녕하세요👋 오늘 하루는 어떠셨어요?"}]
if "diary_mode" not in st.session_state:
    st.session_state.diary_mode = False
if "topic_i" not in st.session_state:
    st.session_state.topic_i = 0
if "qcount_in_topic" not in st.session_state:
    st.session_state.qcount_in_topic = 0
if "used_idx_by_topic" not in st.session_state:
    st.session_state.used_idx_by_topic = []
if "awaiting_consent" not in st.session_state:
    st.session_state.awaiting_consent = False
if "diary_sess" not in st.session_state:
    st.session_state.diary_sess = None
if "topics" not in st.session_state:
    st.session_state.topics = []

if "current_topic" not in st.session_state:
    st.session_state.current_topic = ""
if "candidate_topic" not in st.session_state:
    st.session_state.candidate_topic = ""
if "candidate_votes" not in st.session_state:
    st.session_state.candidate_votes = 0

# ===================== 2) UI Helper Functions =====================

def log_user_turn(user_raw, topic = "", meta = None, ts = None):
    """사용자 발화 로그 및 메모리 저장"""
    nrm = normalize_user_utterance(user_raw or "")
    std = nrm.get("standard") or (user_raw or "")
    
    conversation_memory_raw.append(user_raw)
    conversation_memory_std.append(std)

    log_event("user", content_raw=user_raw, content_std=std, topic=topic, meta=meta, ts=ts)
    
    return std

def log_assistant_turn(text, topic = "", meta = None, ts = None):
    """시스템 발화 로그 저장"""
    log_event("assistant", content_raw=text, content_std=text, topic=topic, meta=meta, ts=ts)

# ===================== 3) 일기장 흐름 제어 =====================

def start_diary_session():
    """일기장 세션 초기화"""
    st.session_state.diary_sess = {
        "diary_id": f"diary_{int(time.time())}",
        "started_at": time.time(),
        "scores": {},
        "score_total": 0,
        "messages": [],
        "topics": [],
        "diary_summaries": []
    }
    st.session_state.diary_mode = True
    st.session_state.topic_i = 0
    st.session_state.qcount_in_topic = 0
    st.session_state.topics = []
    st.session_state.used_idx_by_topic = [set() for _ in range(len(BACKUP_MACRO_TOPICS))]
    st.session_state.awaiting_consent = False

def ask_check_question(i):
    """체크리스트 질문"""
    _, q = CHECKS[i]
    ts = time.time()
    st.session_state.messages.append({"role":"assistant","content":f"[일기장] {q}"})
    log_assistant_turn(q, topic="체크리스트", ts=ts)
    st.session_state.diary_sess["messages"].append({"role":"assistant","content":q,"topic":"체크리스트","ts":ts})

def handle_check_answer(i, user_raw):
    """체크리스트 답변 처리 및 채점"""
    key, _ = CHECKS[i]
    ts = time.time()
    std = log_user_turn(user_raw, topic="체크리스트", meta={"tag": key}, ts=ts)
    st.session_state.diary_sess["messages"].append(
        {"role":"user","content_raw":user_raw,"content_std":std,"topic":"체크리스트","ts":ts}
    )
    score = int(SCORE_FN[key](std))
    st.session_state.diary_sess["scores"][key] = score
    st.session_state.diary_sess["score_total"] = sum(st.session_state.diary_sess["scores"].values())

def setup_topics():
    """일기 주제 3개 설정"""
    topics = pick_diary_topics(3)
    st.session_state.topics = topics
    st.session_state.used_idx_by_topic = [set() for _ in topics]
    st.session_state.topic_i = 0
    st.session_state.qcount_in_topic = 0
    st.session_state.awaiting_consent = False
    st.session_state.diary_sess["topics"] = topics
    msg = f"[일기장] 오늘의 주제: {', '.join(topics)}"
    st.session_state.messages.append({"role":"assistant","content":msg})
    log_assistant_turn(msg)

def pick_question_for_topic(ti):
    """주제에 대한 질문 하나 선택"""
    from core_logic import DIARY_QUESTION_TEMPLATES as QT
    
    used = st.session_state.used_idx_by_topic[ti]
    all_idx = list(range(len(QT)))
    cand = [i for i in all_idx if i not in used]
    if not cand: used.clear(); cand = all_idx[:]
    idx = random.choice(cand)
    used.add(idx)
    
    t = st.session_state.topics[ti]
    return QT[idx].format(t=t)

def ask_topic_question():
    """주제별 질문"""
    ti = st.session_state.topic_i
    q = pick_question_for_topic(ti)
    ts = time.time()
    msg = f"[일기장] {q}"
    st.session_state.messages.append({"role":"assistant","content":msg})
    log_assistant_turn(q, topic=st.session_state.topics[ti], ts=ts)
    st.session_state.diary_sess["messages"].append({"role":"assistant","content":q,"topic":st.session_state.topics[ti],"ts":ts})
    st.session_state.qcount_in_topic += 1

def ask_consent():
    """추가 대화 동의 여부 질문"""
    ts = time.time()
    st.session_state.awaiting_consent = True
    st.session_state.messages.append({"role":"assistant","content":f"[일기장] {CONSENT_PROMPT}"})
    log_assistant_turn(CONSENT_PROMPT, topic=st.session_state.topics[st.session_state.topic_i],
                       meta={"type":"consent"}, ts=ts)
    st.session_state.diary_sess["messages"].append(
        {"role":"assistant","content":CONSENT_PROMPT,"topic":st.session_state.topics[st.session_state.topic_i],"ts":ts}
    )

def handle_consent_input(user_raw):
    """동의 답변 처리"""
    topic = st.session_state.topics[st.session_state.topic_i]
    ts = time.time()
    std = log_user_turn(user_raw, topic=topic, meta={"phase":"consent"}, ts=ts)
    st.session_state.diary_sess["messages"].append(
        {"role":"user","content_raw":user_raw,"content_std":std,"topic":topic,"ts":ts}
    )
    
    empath = asyncio.run(empathy_only(std))
    st.session_state.messages.append({"role":"assistant","content":empath})
    log_assistant_turn(empath, topic=topic, meta={"type":"empathy_after_consent"})
    st.session_state.diary_sess["messages"].append({"role":"assistant","content":empath,"topic":topic,"ts":time.time()})
    
    cont = asyncio.run(classify_consent(std, topic))
    st.session_state.awaiting_consent = False
    if cont: ask_topic_question()
    else: goto_next_topic_or_finish()

def goto_next_topic_or_finish():
    """다음 주제로 이동 또는 종료"""
    st.session_state.topic_i += 1
    st.session_state.qcount_in_topic = 0
    st.session_state.awaiting_consent = False
    if st.session_state.topic_i < len(st.session_state.topics):
        ask_topic_question()
    else:
        st.session_state.diary_mode = False
        st.session_state.diary_sess["ended_at"] = time.time()
        try: asyncio.run(summarize_diary_session(st.session_state.diary_sess))
        except Exception: pass
        diary_memory.append(st.session_state.diary_sess)
        st.session_state.messages.append({"role":"assistant","content":"[일기장] 오늘 기록이 정리되었어요. 이어서 자유롭게 이야기 나눠요. 😊"})

def handle_topic_answer(user_raw):
    """주제별 질문 답변 처리"""
    ti = st.session_state.topic_i
    topic = st.session_state.topics[ti]
    ts = time.time()
    std = log_user_turn(user_raw, topic=topic, ts=ts)
    st.session_state.diary_sess["messages"].append({"role":"user","content_raw":user_raw,"content_std":std,"topic":topic,"ts":ts})
    
    empath = asyncio.run(empathy_only(std))
    st.session_state.messages.append({"role":"assistant","content":empath})
    log_assistant_turn(empath, topic=topic, meta={"type":"followup_empathy"})
    st.session_state.diary_sess["messages"].append({"role":"assistant","content":empath,"topic":topic,"ts":time.time()})
    
    if st.session_state.qcount_in_topic < 3:
        ask_topic_question()
    else:
        ask_consent()

# ===================== 4) 렌더링 및 입력 처리 =====================

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

user_text = st.chat_input("편하게 이야기해 주세요.")

if user_text:
    
    if st.session_state.diary_mode and any(w in user_text for w in STOP_WORDS):
        if st.session_state.get("diary_sess"):
            st.session_state.diary_sess["ended_at"] = time.time()
            try: asyncio.run(summarize_diary_session(st.session_state.diary_sess))
            except Exception: pass
            diary_memory.append(st.session_state.diary_sess)

        st.session_state.diary_mode = False
        st.session_state.messages.append(
            {"role":"assistant","content":"[일기장] 오늘 기록을 저장했어요. 오늘은 여기까지 기록할게요."}
        )
        log_assistant_turn("일기 종료(저장 완료)", topic="체크리스트", meta={"cmd":"stop"})
        st.rerun()

    st.session_state.messages.append({"role":"user","content":user_text})

    if (not st.session_state.diary_mode) and ("일기" in user_text):
        if "chat_started" not in st.session_state:
            st.session_state["chat_started"] = True
            
        start_diary_session(); ask_check_question(0)
    
    elif st.session_state.diary_mode:
        answered_checks = sum(1 for m in st.session_state.diary_sess["messages"]
                       if m.get("topic")=="체크리스트" and m.get("role")=="user")
        
        if answered_checks < len(CHECKS):
            handle_check_answer(answered_checks, user_text)
            if answered_checks + 1 < len(CHECKS):
                ask_check_question(answered_checks + 1)
            else:
                setup_topics()
                ask_topic_question()
        else:
            if st.session_state.awaiting_consent:
                handle_consent_input(user_text)
            else:
                handle_topic_answer(user_text)
    
    else:
        # 일반 대화 모드 
        
        reply = asyncio.run(check_memory_consistency_and_reply(user_text))
        
        nrm = normalize_user_utterance(user_text or "")
        std = nrm.get("standard") or user_text
        
        auto_topic = CONTEXT_TOPIC_LABEL or "일상"
        st.session_state.current_topic = auto_topic

        log_event("user", content_raw=user_text, content_std=std, topic=auto_topic, meta=None, ts=time.time())
        
        st.session_state.messages.append({"role":"assistant","content":reply})
        log_event("assistant", content_raw=reply, content_std=reply, topic=auto_topic, meta=None, ts=time.time())

    st.rerun()

# ===================== 5) 다운로드 영역 =====================
st.markdown("---")

HERE = Path(__file__).resolve().parent

st.markdown("### 💾 로그 및 메모리 다운로드")
col1, col2, col3 = st.columns(3)

log_data = json.dumps(conversation_log, ensure_ascii=False, indent=2).encode("utf-8")
col1.download_button("💾 conversation_log.json", data=log_data, file_name="conversation_log.json", mime="application/json")

fact_data = json.dumps(fact_memory, ensure_ascii=False, indent=2).encode("utf-8")
col2.download_button("🧠 fact_memory.json", data=fact_data, file_name="fact_memory.json", mime="application/json")

diary_data = json.dumps(diary_memory, ensure_ascii=False, indent=2).encode("utf-8")
col3.download_button("📔 diary_memory.json", data=diary_data, file_name="diary_memory.json", mime="application/json")


st.markdown("---")
st.markdown("### 📊 CSV 내보내기")
col_csv1, col_csv2, col_csv3, col_csv4 = st.columns(4)

with col_csv1:
    if st.button("⬇️ Fact Memory CSV"):
        out_path = export_fact_memory_csv(str(HERE / "fact_memory.csv"))
        with open(out_path, "rb") as f:
            st.download_button("Download fact_memory.csv", f, file_name="fact_memory.csv")

with col_csv2:
    if st.button("⬇️ Diary CSV (세션/메시지)"):
        s_path = HERE / "diary_sessions.csv"
        m_path = HERE / "diary_messages.csv"
        out_s, out_m = export_diary_memory_csv(str(s_path), str(m_path))
        st.success(f"저장됨: {s_path.name}, {m_path.name}")
        with open(out_s, "rb") as f1:
            col_csv3.download_button("Download sessions.csv", f1, file_name="diary_sessions.csv")
        with open(out_m, "rb") as f2:
            col_csv4.download_button("Download messages.csv", f2, file_name="diary_messages.csv")

if st.session_state.get("diary_sess"):
    st.markdown("---")
    sess_data = json.dumps(st.session_state.diary_sess, ensure_ascii=False, indent=2).encode("utf-8")
    st.download_button("📝 현재 일기장 세션(JSON)",
        data=sess_data, file_name="diary_session_current.json", mime="application/json")