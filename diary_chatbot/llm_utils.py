import os
import json
import asyncio
from openai import OpenAI
from config import FINETUNED_DIALECT_MODEL, FINETUNED_EMPATHY_MODEL, DIALECT_FT_JOB_ID, EMPATHY_FT_JOB_ID

# === OpenAI Client ===
_client = None

def get_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("환경변수 OPENAI_API_KEY가 설정되지 않았습니다.")
    return OpenAI(api_key=api_key)

def client():
    global _client
    if _client is None:
        _client = get_client()
    return _client

async def async_ask_gpt(
    prompt = None,
    messages = None,
    model = "gpt-4o",
    temperature = 0.7,
    max_tokens = 300,
    response_format = None,
    tools = None,
    tool_choice = None
):
    """공용 비동기 GPT 호출. Tool Use 지원"""
    if response_format is None:
        response_format = {"type": "text"}
    
    if not messages:
        if prompt:
            messages = [{"role": "user", "content": prompt}]
        else:
            return {"error": "Prompt나 messages가 필요합니다."}
            
    try:
        if tools: temperature = 0.0
        
        loop = asyncio.get_event_loop()
        
        # 동기 호출을 비동기로 래핑하여 실행
        resp = await loop.run_in_executor(None, lambda: client().chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice
        ))
        return resp.model_dump()
        
    except Exception as e:
        print(f"[async_ask_gpt] 호출 오류: {e}")
        return {"error": str(e)}

def resolve_finetuned_model(job_id):
    try:
        job = client().fine_tuning.jobs.retrieve(job_id)
        return getattr(job, "fine_tuned_model", None)
    except Exception as e:
        print(f"[MODEL] 파인튜닝 모델 조회 실패: {e}")
        return None

def load_few_shot_empathy(path):
    """지정된 경로에서 공감 few-shot 프롬프트를 로드"""
    try:
        p = Path(path)
        with p.open("r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return (
            "예시:\n"
            "- 사용자: 요즘 제가 좀 헷갈려요\n"
            "  -> 시스템: 그럴 때가 있죠, 마음이 복잡하실 텐데요. 혹시 지금 어떤 일에 대해 헷갈리시는지 말씀해주실 수 있으신가요?\n"
            "- 사용자: 친구들이랑 싸워서 속상해\n"
            "  -> 시스템: 속상하셨겠어요. 친구들과의 관계가 소중하니까 더 마음이 쓰이셨을 것 같아요. 혹시 어떤 이야기로 다투게 되셨나요?\n"
        )
    
def get_finetuned_model_or_default(ft_job_id):
    """파인튜닝 모델명 반환"""
    if ft_job_id == DIALECT_FT_JOB_ID:
        global FINETUNED_DIALECT_MODEL
        return FINETUNED_DIALECT_MODEL or "gpt-4o-mini"
    
    elif ft_job_id == EMPATHY_FT_JOB_ID:
        global FINETUNED_EMPATHY_MODEL
        return FINETUNED_EMPATHY_MODEL or "gpt-4o-mini"
    
    return "gpt-4o-mini"

FEW_SHOT_EMPATHY = load_few_shot_empathy()

DIALECT_SYSTEM_PROMPT = (
    "너는 한국어 사투리를 한국어 표준어(존댓말)로 자연스럽게 바꾸는 도우미야. "
    "입력 문장이 사투리인지 감지하고, 표준어로 매끄럽게 변환해. "
    "존댓말로 바꾸되 의미를 바꾸지 말고, 출력은 JSON으로만 해."
)
DIALECT_JSON_SCHEMA = """
다음 JSON 형식으로만 답해:
{
  "standard": "표준어로 자연스럽게 바꾼 문장",
  "is_dialect": true,
  "confidence": 0.0
}
"""

# ===================== Tool Definitions for GPT-4o =====================
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "normalize_utterance",
            "description": "사용자 발화를 표준어로 교정하고 사투리 여부 및 확신도를 판단합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_text": {"type": "string", "description": "교정할 사용자 발화 원문"}
                },
                "required": ["user_text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extract_claims_from_utterance",
            "description": "표준어로 교정된 사용자 발화에서 핵심 사실(Claim)을 추출합니다. 질문이나 의견은 제외하고, 사실만 추출해야 합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_input_std": {"type": "string", "description": "표준어로 교정된 사용자 발화"},
                    "original_raw": {"type": "string", "description": "사용자 발화 원문"},
                    "nrm_meta": {"type": "object", "description": "사투리 교정 결과 메타데이터 (예: {\"is_dialect\": true, \"confidence\": 0.9})"}
                },
                "required": ["user_input_std", "original_raw", "nrm_meta"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_response",
            "description": "최종적으로 사용자에게 보낼 공감 및 후속 질문을 생성합니다. 모든 선행 작업 완료 후 마지막에 호출됩니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_input_std": {"type": "string", "description": "표준어로 교정된 사용자 발화"},
                    "fact_check_result": {"type": "string", "description": "사실 검증 결과를 포함한 종합 메모리 요약. 응답 생성에 필요한 핵심 정보만 요약하여 제공."}
                },
                "required": ["user_input_std"]
            }
        }
    }
]

async def async_ask_gpt_json_batch(prompt, model = "gpt-4o", temperature = 0.0):
    """Tool 없이 JSON 응답만 필요한 호출 (배치 처리에 사용)"""
    return await async_ask_gpt(
        prompt=prompt,
        model=model,
        temperature=temperature,
        max_tokens=4096,
        response_format={"type": "json_object"}
    )