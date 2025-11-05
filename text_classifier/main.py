import os
import json
import torch
import numpy as np
import random
import shutil
from datetime import datetime
from typing import Dict

from data_preprocessing import load_wellness_data, load_emotion_data, preprocess_affect_data
from data_preprocessing import preprocess_story_data, AFFECT_Y_CLASSES, RISK_TARGET_COLS, REPEAT_LABEL
from models_and_trainers import train_affect_model, train_repeat_model, train_risk_model


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT = 'data'
if not os.path.exists(DATA_ROOT): os.makedirs(DATA_ROOT)

WELLNESS_DATA_PATH = os.path.join(DATA_ROOT, '웰니스_대화_스크립트_데이터셋.xlsx')
EMOTION_DATA_PATH  = os.path.join(DATA_ROOT, '감성대화말뭉치(최종데이터)_Training.json')
STORY_ZIP_PATH     = os.path.join(DATA_ROOT, '025.고령자 근현대 경험 기반 스토리 구술 데이터.zip')
STORY_EXTRACT_DIR  = 'story_extracted' 
UNZIPPED_ROOT      = 'story_unzipped' 

MODEL_SAVE_ROOT = 'models_output'
if not os.path.exists(MODEL_SAVE_ROOT): os.makedirs(MODEL_SAVE_ROOT)
TS = datetime.now().strftime("%Y%m%d_%H%M%S")

AFFECT_Y_COLS = [f'y_{c}' for c in AFFECT_Y_CLASSES]
AFFECT_CONFIG = {
    'MODEL_NAME': "klue/roberta-base",
    'Y_COLS': AFFECT_Y_COLS,
    'MAX_LEN': 256, 'BATCH': 32, 'EPOCHS': 3, 'LR': 2e-5,
    'CKPT_PATH': os.path.join(MODEL_SAVE_ROOT, "affect_best.pt"),
    'SAVE_DIR': os.path.join(MODEL_SAVE_ROOT, f"affect_model_{TS}"),
}

REPEAT_CONFIG = {
    'MODEL_NAME': "monologg/koelectra-base-v3-discriminator",
    'LABEL': REPEAT_LABEL,
    'MAX_LEN': 256, 'BATCH': 32, 'EPOCHS': 3, 'LR': 3e-5,
    'CKPT_PATH': os.path.join(MODEL_SAVE_ROOT, "repeat_best.pt"),
    'SAVE_DIR': os.path.join(MODEL_SAVE_ROOT, f"repeat_binary_{TS}"),
}

RISK_CONFIG = {
    'MODEL_NAME': "beomi/KcELECTRA-base-v2022",
    'TARGET_COLS': RISK_TARGET_COLS,
    'MAX_LEN': 256, 'BATCH': 32, 'EPOCHS': 3, 'LR': 2e-5,
    'CKPT_PATH': os.path.join(MODEL_SAVE_ROOT, "risk_best_4.pt"),
    'SAVE_DIR': os.path.join(MODEL_SAVE_ROOT, f"risk_4_kcelectra_{TS}"),
}

def save_model_assets(config: Dict, tokenizer):
    """학습된 모델 파일 (체크포인트, 토크나이저, 메타데이터) 저장"""
    save_dir = config['SAVE_DIR']
    ckpt_path = config['CKPT_PATH']
    os.makedirs(save_dir, exist_ok=True)
    shutil.copy2(ckpt_path, os.path.join(save_dir, os.path.basename(ckpt_path)))
    tokenizer.save_pretrained(save_dir)

    pack = torch.load(ckpt_path, map_location="cpu")
    # 메타데이터 추출
    meta = {k: v for k, v in pack.items() if k not in ['model_state']}

    with open(os.path.join(save_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n[저장 완료] 경로: {save_dir}")


def main_run():
    # 1. 감성 분류 데이터 로드 및 학습
    print("="*50 + "\n## 1. 감성 분류 모델 학습 시작 (RoBERTa)")
    if not (os.path.exists(WELLNESS_DATA_PATH) and os.path.exists(EMOTION_DATA_PATH)):
        print(f"경고: 감성 분류 데이터 파일이 경로 ({DATA_ROOT})에 없습니다. 학습을 건너뜁니다.")
    else:
        wellness_df = load_wellness_data(WELLNESS_DATA_PATH)
        emotion_df  = load_emotion_data(EMOTION_DATA_PATH)
        train_df_affect, val_df_affect = preprocess_affect_data(wellness_df, emotion_df, SEED)
        print(f"감성 분류 데이터 (Train: {len(train_df_affect)}, Val: {len(val_df_affect)})")
        affect_tokenizer = train_affect_model(train_df_affect, val_df_affect, AFFECT_CONFIG, device)
        save_model_assets(AFFECT_CONFIG, affect_tokenizer)

    # 2. 스토리 구술 데이터 로드 및 전처리
    print("="*50 + "\n## 2. 스토리 구술 데이터 전처리 시작")
    if not os.path.exists(STORY_ZIP_PATH):
        print(f"경고: 스토리 구술 데이터 zip 파일이 경로 ({DATA_ROOT})에 없습니다. 학습을 건너뜁니다.")
        return

    (train_df_repeat, val_df_repeat), (train_df_risk, val_df_risk) = preprocess_story_data(STORY_ZIP_PATH, STORY_EXTRACT_DIR, UNZIPPED_ROOT)
    print(f"스토리 데이터 (Train: {len(train_df_repeat)}, Val: {len(val_df_repeat)})")

    # 3. 같은말반복 분류 모델 학습
    print("="*50 + "\n## 3. 같은말반복 분류 모델 학습 시작 (Electra)")
    repeat_tokenizer = train_repeat_model(train_df_repeat, val_df_repeat, REPEAT_CONFIG, device)
    save_model_assets(REPEAT_CONFIG, repeat_tokenizer)

    # 4. 스토리 위험지표 분류 모델 학습
    print("="*50 + "\n## 4. 스토리 위험지표(4개 구체성) 분류 모델 학습 시작 (KcELECTRA)")
    risk_tokenizer = train_risk_model(train_df_risk, val_df_risk, RISK_CONFIG, device)
    save_model_assets(RISK_CONFIG, risk_tokenizer)


if __name__ == '__main__':
    print("--- 텍스트 분류 모델 학습 스크립트 실행 준비 ---")
    print(f"작업 경로: {os.getcwd()}")
    print(f"데이터 경로는 'data_preprocessing.py'의 DATA_ROOT 및 main_run.py의 상단 설정을 확인해주세요.")
    # main_run() 
    print("실행을 원하시면 main_run() 주석을 해제해주세요.")