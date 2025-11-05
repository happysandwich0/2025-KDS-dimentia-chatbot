import os, sys, time
from pathlib import Path
import streamlit as st

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import main
from main import (
    normalize_user_utterance, pick_attention_question, log_event,
    conversation_log_dataframe, check_memory_consistency, generate_question,
    score_attention, export_fact_memory_csv, export_diary_memory_csv,
    pick_diary_topics, diary_rag_reminder, _pick_question, summarize_diary_session
)

st.set_page_config(
    page_title="당신의 소중한 말벗 또랑이 (Streamlit)",
    page_icon="🍊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("⚙️ 설정")
api_key = st.sidebar.text_input("OPENAI_API_KEY", type="password", value=os.environ.get("OPENAI_API_KEY", ""))
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
    st.sidebar.success("API Key loaded")

date_override = st.sidebar.text_input("오늘 날짜 고정 (YYYY-MM-DD) — 필요 시만", value=str(main.CONFIG_TODAY_OVERRIDE or ""))
if date_override.strip():
    main.CONFIG_TODAY_OVERRIDE = date_override.strip()

use_ft_models = st.sidebar.checkbox("파인튜닝된 모델 우선 사용 (가능할 때)", value=True)
st.sidebar.markdown("---")
if st.sidebar.button("🧹 대화 초기화 (메모리 유지)"):
    for k in list(st.session_state.keys()):
        if k.startswith(("chat_", "diary_", "ck_", "used_qidx_map")):
            st.session_state.pop(k)
    st.rerun()

# -----------------------------
# Helpers
# -----------------------------
def render_conversation():
    """일반 대화 로그를 채팅 버블로 렌더링 (user=raw 우선)"""
    df = conversation_log_dataframe()
    for _, row in df.iterrows():
        role = row["role"]
        if role == "user":
            text = row.get("content_raw") or row.get("content_std") or row.get("content")
            with st.chat_message("user"):
                st.write(text)
        else:
            text = row.get("content_std") or row.get("content_raw") or row.get("content")
            with st.chat_message("assistant"):
                st.write(text)

def save_user_turn_global(user_text: str, topic: str = ""):
    """모든 사용자 발화를 무조건 저장 (global + fact 처리)"""
    nrm = normalize_user_utterance(user_text or "")
    user_std = nrm["standard"]

    # 전역 메모리
    main.conversation_memory_raw.append(user_text or "")
    main.conversation_memory_std.append(user_std)

    # 전역 로그
    log_event("user", content_raw=(user_text or ""), content_std=user_std, topic=topic)

    # 사실 추출/일관성 검사 (표준어 기준)
    check_memory_consistency(user_std, user_text, nrm)
    return user_std

def assistant_say(text: str, topic: str = "", meta=None):
    """어시스턴트 발화 저장"""
    log_event("assistant", content_raw=text, content_std=text, topic=topic, meta=meta or {})

# -----------------------------
# Tabs
# -----------------------------
tab_chat, tab_diary, tab_logs = st.tabs(["💬 일반 대화", "📔 일기장", "📊 로그/내보내기"])

# =============================
# 1) 일반 대화
# =============================
with tab_chat:
    st.subheader("💬 일반 대화")
    st.caption("첫 턴은 주의력 질문으로 시작합니다. (사용자 발화는 사투리 원문 그대로 표시)")

    if "chat_started" not in st.session_state:
        st.session_state["chat_started"] = True
        att_q = pick_attention_question()
        assistant_say(att_q, topic="주의력")

    render_conversation()

    chat_input = st.chat_input("메시지를 입력하세요…")
    if chat_input is not None:
        # 첫 사용자 답변이면 attention 점수 부여
        if "chat_att_scored" not in st.session_state:
            user_std = save_user_turn_global(chat_input, topic="주의력")
            att_score = score_attention(user_std)
            msg = f"📊 Attention 점수: {att_score}"
            assistant_say(msg, topic="주의력")
            st.session_state["chat_att_scored"] = True
        else:
            # 일반 흐름: 저장 후 응답 생성
            user_std = save_user_turn_global(chat_input, topic=main.CONTEXT_TOPIC_LABEL or "")
            reply = generate_question(user_std) or "말씀 감사합니다. 좀 더 들려주실래요?"
            assistant_say(reply, topic=main.CONTEXT_TOPIC_LABEL or "")
        st.rerun()

# =============================
# 2) 일기장 (버튼 없는 자동 진행)
# =============================
with tab_diary:
    st.subheader("📔 일기장")
    st.caption("체크리스트 5문항(한 문항씩) → 토픽 3개 × 질문 3개(자동) → 요약/저장")

    # --- 상태 초기화 ---
    if "diary_state" not in st.session_state:
        st.session_state.diary_state = {
            "phase": "idle",        # idle → checklist → topics → talk → done
            "sess": None,
            "scores": {},
            "ck_idx": 0,
            "ck_answers": {},
            "topics": [],
            "topic_idx": 0,         # 현재 토픽 인덱스
            "qcount_in_topic": 0,   # 현재 토픽에서 진행한 질문 수 (0..3)
            "current_q": None,      # 현재 질문 텍스트
            "current_qidx": None,   # 현재 질문 인덱스
        }
        st.session_state.used_qidx_map = {}

    S = st.session_state.diary_state

    # 체크리스트 문항
    q_and_tags = [
        ("오늘이 몇월 며칠일까요?", "today_date"),
        ("오늘 날씨는 어떤가요?", "today_weather"),
        ("지금 어디에 계신가요?", "current_location"),
        ("오늘로부터 7일 전은 몇월 며칠일까요?", "date_7days_ago"),
        ("어제 뭐하셨어요?", "yesterday_activity"),
    ]

    # --- 시작 버튼(한 번만) ---
    if S["phase"] == "idle":
        if st.button("📔 일기장 시작하기"):
            now_ts = time.time()
            S["sess"] = {"id": int(now_ts), "started_at": now_ts, "messages": []}
            S["phase"] = "checklist"
            S["ck_idx"] = 0
            S["ck_answers"] = {}
            st.success("체크리스트를 시작합니다.")
            st.rerun()

    # --- 체크리스트 단계 ---
    elif S["phase"] == "checklist":
        idx = S["ck_idx"]
        total = len(q_and_tags)
        st.markdown(f"### ✅ 체크리스트 ({idx+1}/{total})")

        if idx < total:
            q, tag = q_and_tags[idx]

            # 질문을 화면/로그에 남김(한 번만)
            key_q_logged = f"ck_q_logged_{idx}"
            if not st.session_state.get(key_q_logged, False):
                S["sess"]["messages"].append({"role": "assistant", "content": q, "topic": "체크리스트", "ts": time.time()})
                assistant_say(q, topic="체크리스트")
                st.session_state[key_q_logged] = True

            st.write(f"**Q. {q}**")
            a = st.text_input("↳ 답변을 입력하세요", key=f"ck_ans_{idx}")

            # 사용자가 Enter로 제출하면 즉시 저장
            if a is not None and a != "" and st.session_state.get(f"ck_saved_{idx}") != a:
                # raw/std 저장
                nrm = normalize_user_utterance(a)
                a_std = nrm["standard"]
                ts_now = time.time()

                S["ck_answers"][tag] = a
                S["sess"]["messages"].append({"role": "user", "content_raw": a, "content_std": a_std, "topic": "체크리스트", "ts": ts_now})
                log_event("user", content_raw=a, content_std=a_std, topic="체크리스트", meta={"tag": tag}, ts=ts_now)

                st.session_state[f"ck_saved_{idx}"] = a
                S["ck_idx"] += 1
                st.rerun()
        else:
            # 채점
            A = S["ck_answers"]
            scores = {
                "today_date":         main.score_today_date(A.get("today_date","")),
                "today_weather":      main.score_today_weather(A.get("today_weather","")),
                "current_location":   main.score_current_location(A.get("current_location","")),
                "date_7days_ago":     main.score_seven_days_ago(A.get("date_7days_ago","")),
                "yesterday_activity": main.score_yesterday_activity(A.get("yesterday_activity","")),
            }
            S["scores"] = scores
            S["sess"]["scores"] = scores
            total_score = sum(scores.get(k, 0) for k in main.DIARY_CHECK_KEYS)
            st.success(f"📊 체크 결과: {scores} (합계 {total_score}/5)")

            # 토픽 준비
            S["topics"] = pick_diary_topics(3)
            st.session_state.used_qidx_map = {t: set() for t in S["topics"]}
            S["topic_idx"] = 0
            S["qcount_in_topic"] = 0
            S["current_q"] = None
            S["current_qidx"] = None
            S["phase"] = "topics"
            st.rerun()

    # --- 토픽 소개(한 번) → 바로 talk으로 ---
    elif S["phase"] == "topics":
        st.markdown("### 🧩 오늘의 대화 주제 3개")
        st.write(", ".join(f"‘{t}’" for t in S["topics"]))
        st.info("각 단어(주제)마다 질문 3개를 자동으로 진행합니다. 답변을 입력하면 즉시 저장돼요.")
        S["phase"] = "talk"
        st.rerun()

    # --- 토픽 대화: 버튼 없이 chat_input으로만 진행 ---
    elif S["phase"] == "talk":
        topics = S["topics"]
        i = S["topic_idx"]
        if i >= len(topics):
            S["phase"] = "done"
            st.rerun()
        else:
            t = topics[i]
            st.markdown(f"### — 주제 [{t}] — (질문 {S['qcount_in_topic']+1}/3)")

            # 리마인드(토픽 최초 진입 시에만 한 번)
            key_memolog = f"memo_logged_{i}"
            if not st.session_state.get(key_memolog):
                memo = diary_rag_reminder(t)
                if memo:
                    S["sess"]["messages"].append({"role": "assistant", "content": memo, "topic": t, "ts": time.time()})
                    assistant_say(memo, topic=t, meta={"type": "reminder"})
                    with st.chat_message("assistant"):
                        st.write(f"📝 리마인드: {memo}")
                st.session_state[key_memolog] = True

            # 현재 질문이 없으면 뽑기
            used = st.session_state.used_qidx_map.get(t, set())
            if S["current_q"] is None:
                q, qidx = _pick_question(t, used)
                S["current_q"] = q
                S["current_qidx"] = qidx

                # 질문을 화면/세션/로그에 남김
                S["sess"]["messages"].append({"role":"assistant","content": q, "topic": t, "ts": time.time()})
                assistant_say(q, topic=t, meta={"type":"diary_q","qidx":qidx})

            # 질문 표시
            st.write(f"**Q. {S['current_q']}**")
            user_a = st.chat_input("여기에 답변을 입력하고 Enter를 눌러주세요…")

            if user_a is not None:
                # 답변 저장(raw/std)
                nrm = normalize_user_utterance(user_a)
                a_std = nrm["standard"]
                ts_a = time.time()
                S["sess"]["messages"].append({"role":"user","content_raw": user_a, "content_std": a_std, "topic": t, "ts": ts_a})
                log_event("user", content_raw=user_a, content_std=a_std, topic=t, ts=ts_a)

                # 공감형 후속 반응 1회
                follow = generate_question(a_std) or "말씀 감사합니다. 조금만 더 들려주실래요?"
                S["sess"]["messages"].append({"role":"assistant","content": follow, "topic": t, "ts": time.time()})
                assistant_say(follow, topic=t, meta={"type":"followup"})

                # 사용 처리 + 카운트 증가
                used.add(S["current_qidx"])
                st.session_state.used_qidx_map[t] = used
                S["qcount_in_topic"] += 1

                # 3개 채웠으면 다음 토픽으로
                if S["qcount_in_topic"] >= 3:
                    S["topic_idx"] += 1
                    S["qcount_in_topic"] = 0
                    S["current_q"] = None
                    S["current_qidx"] = None
                else:
                    # 같은 토픽에서 다음 질문 준비
                    S["current_q"] = None
                    S["current_qidx"] = None

                st.rerun()

    # --- 종료/요약 ---
    elif S["phase"] == "done":
        st.success("오늘의 일기장을 마쳤습니다. 1문장 요약을 생성하고 저장합니다.")

        sess = S["sess"]
        summarize_diary_session(sess)
        if "topics" not in sess:
            sess["topics"] = S.get("topics", [])
        if "scores" not in sess:
            sess["scores"] = S.get("scores", {})

        main.diary_memory.append(sess)

        st.write("**요약:**")
        for item in sess.get("diary_summaries", []):
            st.write(f"- [{item['topic']}] {item['summary']}")

        total_score = sum(S.get("scores", {}).get(k, 0) for k in main.DIARY_CHECK_KEYS)
        st.info(f"📊 오늘 체크리스트 점수 합계: {total_score}/5")

        # 다음 세션 대비 초기화
        st.info("새로 시작하려면 상단의 '대화 초기화'를 눌러주세요.")

# =============================
# 3) 로그/내보내기
# =============================
with tab_logs:
    st.subheader("📊 대화 로그 & 내보내기")

    st.markdown("#### 전체 대화 로그")
    df = conversation_log_dataframe()
    st.dataframe(df, width="stretch")

    st.markdown("#### fact_memory (최근 20개)")
    st.write(f"총 {len(main.fact_memory)}건")
    st.json(main.fact_memory[-20:])

    st.markdown("#### diary_memory (세션 개수)")
    st.write(f"{len(main.diary_memory)}개 세션")
    if main.diary_memory:
        last = main.diary_memory[-1]
        st.write("최근 세션 일부")
        st.json({"id": last.get("id"), "started_at": last.get("started_at"), "keys": list(last.keys())})

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬇️ fact_memory.csv 저장"):
            out_path = export_fact_memory_csv(str(HERE / "fact_memory.csv"))  # main.py 쪽도 utf-8-sig 적용 권장
            st.success(f"저장됨: {out_path}")
            with open(out_path, "rb") as f:
                st.download_button("Download fact_memory.csv", f, file_name="fact_memory.csv")
    with col2:
        if st.button("⬇️ diary_memory CSV 저장 (sessions/messages)"):
            s_path = HERE / "diary_sessions.csv"
            m_path = HERE / "diary_messages.csv"
            out_s, out_m = export_diary_memory_csv(str(s_path), str(m_path))
            st.success(f"저장됨: {out_s}, {out_m}")
            with open(out_s, "rb") as f1:
                st.download_button("Download diary_sessions.csv", f1, file_name="diary_sessions.csv")
            with open(out_m, "rb") as f2:
                st.download_button("Download diary_messages.csv", f2, file_name="diary_messages.csv")

    st.markdown("---")
    st.caption("일반 대화는 사용자 원문(사투리)로 표시되고, fact 추출은 표준어로 처리됩니다.")

