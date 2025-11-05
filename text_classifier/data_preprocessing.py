import pandas as pd
import numpy as np
import os
import json
import zipfile
import re
from glob import glob
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

AFFECT_Y_CLASSES = ['우울/무기력', '불안/초조', '감정조절문제']
RISK_TARGET_COLS = ["사건구체성", "자서전적기억", "시간적구체성", "공간적구체성"]
REPEAT_LABEL     = "같은말반복"
ALL_STORY_TARGET_COLS = RISK_TARGET_COLS + [REPEAT_LABEL]

# --- 감성 분류 데이터 전처리 ---

def load_wellness_data(path: str) -> pd.DataFrame:
    """웰니스 대화 데이터 로드 및 전처리"""
    new_df = pd.read_excel(path)
    new_df.drop(columns=['챗봇'], inplace=True)

    depression_lethargy = ['감정/우울감', '감정/무력감', '감정/의욕상실', '감정/기분저하', '증상/무기력']
    anxiety_agitation = ['감정/불안감', '감정/걱정', '감정/초조함', '감정/긴장', '증상/공황발작']
    emotional_dysregulation = ['감정/감정조절이상', '감정/분노', '감정/화', '감정/짜증', '증상/공격적성향', '감정/예민함']
    target_categories = depression_lethargy + anxiety_agitation + emotional_dysregulation

    filtered_df = new_df[new_df['구분'].isin(target_categories)].copy()

    category_map = {}
    for item in depression_lethargy: category_map[item] = '우울/무기력'
    for item in anxiety_agitation: category_map[item] = '불안/초조'
    for item in emotional_dysregulation: category_map[item] = '감정조절문제'

    filtered_df['대분류'] = filtered_df['구분'].map(category_map)
    filtered_df = filtered_df.rename(columns={'유저': 'text'})[['text', '대분류']].copy()
    filtered_df['talk_id'] = np.nan
    return filtered_df

def load_emotion_data(path: str) -> pd.DataFrame:
    """감성 대화 말뭉치 로드 (60대 이상 & E66 필터링)"""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    def join_text(content: dict, max_turns: int = 50) -> str:
        utts = []
        for i in range(1, max_turns + 1):
            for k in (f"HS{i:02d}", f"SS{i:02d}"):
                v = content.get(k, "")
                if isinstance(v, str) and v.strip(): utts.append(v.strip())
        return " ".join(utts)

    def derive_age_gender(human_codes):
        age_code = next((code for code in human_codes if isinstance(code, str) and code.startswith("A")), None)
        AGE_MAP_KO = {"A01": "청소년", "A02": "청년", "A03": "중년", "A04": "60대 이상"}
        return AGE_MAP_KO.get(age_code, None)

    def flatten_record(rec: dict) -> dict:
        emotion = rec.get("profile", {}).get("emotion", {})
        talk = rec.get("talk", {})
        text = join_text(talk.get("content", {}))
        age = derive_age_gender(rec.get("profile", {}).get("persona", {}).get("human", []))
        tid = (talk.get("id", {}) or {}).get("talk-id")
        return {"talk_id": tid, "text": text, "emotion": emotion.get("type"), "person.age": age}

    rows = [flatten_record(r) for r in (data if isinstance(data, list) else [data])]
    df = pd.DataFrame(rows)
    df_e66_elderly = df[
        (df["emotion"].astype(str).str.strip().str.upper() == "E66") &
        (df["person.age"] == "60대 이상")
    ].copy()
    df_e66_elderly = df_e66_elderly[['talk_id', 'text']].copy()
    df_e66_elderly['대분류'] = np.nan
    return df_e66_elderly

def preprocess_affect_data(wellness_df: pd.DataFrame, emotion_df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """감성 데이터 결합, 라벨링, 학습/검증 분할"""
    combined_df = pd.concat([wellness_df, emotion_df], ignore_index=True)
    combined_df['text'] = combined_df['text'].fillna('').str.strip()
    combined_df = combined_df[combined_df['text']!=''] \
                             .drop_duplicates(subset=['text'], keep='first') \
                             .reset_index(drop=True)

    def to_list(x):
        if isinstance(x, list): return [str(s).strip() for s in x if str(s).strip()]
        s = str(x).strip("[](){} ").strip().strip("'").strip('"') if not pd.isna(x) else ""
        if not s: return []
        parts = re.split(r'[,\;|]+', s)
        return [q for p in parts for q in p.strip().strip("'").strip('"').split() if q]

    combined_df['labels_list'] = combined_df['대분류'].apply(to_list)

    norm_map = {"우울/ 무기력": "우울/무기력", "불안 /초조": "불안/초조", "불안 / 초조": "불안/초조",
                "감정 조절 문제": "감정조절문제", "감정조절 문제": "감정조절문제"}
    def normalize_labels(L):
        return [norm_map.get(t.replace('\u00a0',' ').strip(), t.replace('\u00a0',' ').strip()) for t in L]

    combined_df['labels_list'] = combined_df['labels_list'].apply(normalize_labels)

    y_cols = [f'y_{c}' for c in AFFECT_Y_CLASSES]
    for c in AFFECT_Y_CLASSES:
        combined_df[f'y_{c}'] = combined_df['labels_list'].apply(lambda L: 1 if c in L else 0).astype('uint8')

    combined_df['is_labeled'] = combined_df['labels_list'].apply(lambda L: len(L) > 0)
    train_pool = combined_df[combined_df['is_labeled']].copy()

    X = train_pool['text'].values
    Y = train_pool[y_cols].values

    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr_idx, va_idx = next(msss.split(X, Y))
    train_df = train_pool.iloc[tr_idx].reset_index(drop=True)
    val_df   = train_pool.iloc[va_idx].reset_index(drop=True)

    return train_df, val_df

# --- 스토리 구술 데이터 전처리 ---

def unzip_story_data(zip_path: str, extract_dir: str, unzipped_root: str):
    """고령자 스토리 구술 데이터 압축 해제"""
    if not os.path.exists(extract_dir) or not os.listdir(extract_dir):
        print(f"압축 해제: {zip_path} → {extract_dir}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

    train_zip_dir = os.path.join(extract_dir, "025.고령자 근현대 경험 기반 스토리 구술 데이터/3.개방데이터/1.데이터/Training/02.라벨링데이터")
    val_zip_dir = os.path.join(extract_dir, "025.고령자 근현대 경험 기반 스토리 구술 데이터/3.개방데이터/1.데이터/Validation/02.라벨링데이터")
    train_zips = glob(os.path.join(train_zip_dir, "*.zip"))
    val_zips = glob(os.path.join(val_zip_dir, "*.zip"))

    unzipped_train_dir = os.path.join(unzipped_root, "train")
    unzipped_val_dir = os.path.join(unzipped_root, "val")
    os.makedirs(unzipped_train_dir, exist_ok=True)
    os.makedirs(unzipped_val_dir, exist_ok=True)

    def unzip_all(zip_list, extract_root):
        for zip_path in zip_list:
            try:
                extract_path = os.path.join(extract_root, os.path.splitext(os.path.basename(zip_path))[0])
                if not os.path.exists(extract_path) or not os.listdir(extract_path):
                    with zipfile.Zipfile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_path)
            except Exception as e:
                print(f"오류 발생: {zip_path}, {e}")

    unzip_all(train_zips, unzipped_train_dir)
    unzip_all(val_zips, unzipped_val_dir)
    return {"train": unzipped_train_dir, "val": unzipped_val_dir}

def load_story_data(root_dir: str) -> pd.DataFrame:
    """압축 해제된 JSON 파일에서 스토리 데이터 로드"""
    json_paths = sorted(glob(os.path.join(root_dir, "**/*.json"), recursive=True))
    data_list = []
    for path in json_paths:
        with open(path, 'r', encoding='utf-8') as f:
            try: js = json.load(f)
            except: continue
        if not isinstance(js, dict): continue
        qa = js.get("qa", [])
        text_content = " ".join([item.get("answer", "") for item in qa if isinstance(item, dict)]) if isinstance(qa, list) else ""
        data_list.append({
            "text": text_content,
            "label_1": js.get("label_1", [{}]),
            "label_2": js.get("label_2", [{}]),
            "qualityPoint": js.get("qualityPoint", None),
        })
    return pd.DataFrame(data_list)

def attach_story_targets(df: pd.DataFrame, target_cols: list) -> pd.DataFrame:
    """스토리 데이터에 이진 라벨 (구체성, 반복) 부착"""
    df = df.copy()

    def extract_binary_variables(row, target_cols_list):
        label_1 = row.get("label_1", [{}])[0] if isinstance(row.get("label_1"), list) and len(row.get("label_1")) > 0 and isinstance(row.get("label_1")[0], dict) else {}
        label_2 = row.get("label_2", [{}])[0] if isinstance(row.get("label_2"), list) and len(row.get("label_2")) > 0 and isinstance(row.get("label_2")[0], dict) else {}
        out = {}
        for k in target_cols_list:
            v = label_1.get(k, 0) if k in RISK_TARGET_COLS else label_2.get(k, 0)
            try: out[k] = int(v) if isinstance(v, bool) else int(float(v))
            except Exception: out[k] = 0
        return pd.Series(out, index=target_cols_list)

    feats = df.apply(lambda r: extract_binary_variables(r, target_cols), axis=1)
    for c in target_cols: df[c] = feats[c].astype("float32")

    df = df[df["text"].apply(lambda x: isinstance(x, str) and len(str(x).strip()) > 0)].reset_index(drop=True)
    return df

def preprocess_story_data(zip_path: str, extract_dir: str, unzipped_root: str):
    """스토리 데이터 전처리 및 구체성/반복 모델 학습 데이터 분리"""
    unzipped_dirs = unzip_story_data(zip_path, extract_dir, unzipped_root)
    train_df_raw = load_story_data(unzipped_dirs["train"])
    val_df_raw = load_story_data(unzipped_dirs["val"])

    # 1. 반복 라벨 데이터프레임 (5개 라벨 포함)
    train_df_repeat = attach_story_targets(train_df_raw.copy(), ALL_STORY_TARGET_COLS)
    val_df_repeat   = attach_story_targets(val_df_raw.copy(), ALL_STORY_TARGET_COLS)

    # 2. 위험지표(구체성 4개) 라벨 데이터프레임 (QualityPoint 정규화 포함)
    def squash_qualitypoint(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        qcols = [c for c in df.columns if str(c).strip() == "qualityPoint"]
        if not qcols:
            df["qualityPoint"] = np.nan
            return df
        qvals = pd.to_numeric(df[qcols[-1]], errors="coerce").astype("float32")
        df["qualityPoint"] = qvals.values
        return df

    train_df_risk = squash_qualitypoint(train_df_raw.copy())
    val_df_risk   = squash_qualitypoint(val_df_raw.copy())

    train_df_risk = attach_story_targets(train_df_risk, RISK_TARGET_COLS)
    val_df_risk   = attach_story_targets(val_df_risk, RISK_TARGET_COLS)

    # QualityPoint 정규화 (mean, std는 train 데이터에서 계산)
    q_mean = float(train_df_risk["qualityPoint"].mean(skipna=True))
    q_std  = float(train_df_risk["qualityPoint"].std(skipna=True))
    q_std = q_std if np.isfinite(q_std) and q_std != 0.0 else 1.0

    for df in (train_df_risk, val_df_risk):
        df["qualityPoint_std"] = ((df["qualityPoint"].fillna(q_mean) - q_mean) / q_std).astype("float32")

    return (train_df_repeat, val_df_repeat), (train_df_risk, val_df_risk)