# -*- coding: utf-8 -*-
"""
model.py
- OpenAI 클라이언트 초기화
- 공용 GPT 호출 유틸
- 파인튜닝 모델 ID 해석기
- 프롬프트 로더
"""
import os
from typing import Optional
from openai import OpenAI

# === OpenAI Client ===
def get_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("환경변수 OPENAI_API_KEY가 설정되어 있지 않습니다.")
    return OpenAI(api_key=api_key)

_client: Optional[OpenAI] = None

def client() -> OpenAI:
    global _client
    if _client is None:
        _client = get_client()
    return _client

def ask_gpt(prompt, model="gpt-4o-mini", temperature=0.7, max_tokens=300, response_format=None):
    """
    공용 GPT 호출. response_format은 {"type":"text"} 또는 {"type":"json_object"} 등을 사용.
    """
    if response_format is None:
        response_format = {"type": "text"}
    try:
        resp = client().chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[ask_gpt] 호출 오류: {e}")
        return ""

def resolve_finetuned_model(job_id: str) -> Optional[str]:
    """
    Fine-tuning Job ID로 최종 모델명을 조회합니다.
    """
    try:
        job = client().fine_tuning.jobs.retrieve(job_id)
        return getattr(job, "fine_tuned_model", None)
    except Exception as e:
        print(f"[MODEL] 파인튜닝 모델 조회 실패: {e}")
        return None

def load_few_shot_empathy(path: str = "prompts/few_shot_empathy.txt") -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        # fallback 기본 프롬프트
        return (
            "예시:\n"
            "- 사용자: 요즘 제가 좀 헷갈려요\n"
            "  -> 시스템: 그럴 때가 있죠, 마음이 복잡하실 텐데요. 혹시 지금 어떤 일에 대해 헷갈리시는지 말씀해주실 수 있으신가요?\n"
            "- 사용자: 친구들이랑 싸워서 속상해\n"
            "  -> 시스템: 속상하셨겠어요. 친구들과의 관계가 소중하니까 더 마음이 쓰이셨을 것 같아요. 혹시 어떤 이야기로 다투게 되셨나요?\n"
        )