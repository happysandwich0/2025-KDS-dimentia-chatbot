import json
import os, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from demrisk_encoders import *
import warnings
try:
    from sklearn.exceptions import InconsistentVersionWarning  
except Exception:
    class InconsistentVersionWarning(Warning):
        """Fallback when sklearn doesn't expose this symbol."""
        pass

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
from torch.serialization import add_safe_globals


EXPECTED_Q12 = np.array([93, 86, 79, 72, 65], dtype=float)

def mmse13_to_feature_row(mmse_13):
    """
    mmse_13: array-like(13,) 또는 dict.
      순서 가정: [Q01..Q08, Q12_1..Q12_5]
      값 예시: Q01~Q08은 1/2, Q12_*는 정답 숫자
    반환: pd.DataFrame (1행), 학습 때와 동일한 컬럼 순서
      [Q01_bin..Q08_bin,
       Q12_1_correct, Q12_1_abs_err, ..., Q12_5_correct, Q12_5_abs_err,
       Q12_correct_count, Q12_abs_err_mean]
    """
    # 1) dict로 표준화
    if not isinstance(mmse_13, dict):
        keys = [f"Q0{i}" for i in range(1, 9)] + [f"Q12_{i}" for i in range(1, 6)]
        vals = np.asarray(mmse_13, dtype=float).tolist()
        if len(vals) != 13:
            raise ValueError(f"MMSE 길이가 13이 아닙니다: {len(vals)}")
        mmse_13 = dict(zip(keys, vals))
    df = pd.DataFrame([mmse_13])

    # 2) Q01~Q08 → _bin
    q01_08_cols = [f"Q0{i}" for i in range(1, 9)]
    out1 = pd.DataFrame(index=df.index)
    for c in q01_08_cols:
        x = pd.to_numeric(df.get(c), errors='coerce')
        out1[c + "_bin"] = np.where(x == 2, 1.0, np.where(x == 1, 0.0, np.nan))

    # 3) Q12 → correct/abs_err + 집계 (컬럼 생성 순서 주의: correct → abs_err 순으로 1~5 반복)
    q12_cols = [f"Q12_{i}" for i in range(1, 6)]
    out2 = pd.DataFrame(index=df.index)
    corr_cols, err_cols = [], []
    for i, c in enumerate(q12_cols):
        x = pd.to_numeric(df.get(c), errors='coerce')
        c_corr = f"{c}_correct"
        c_err  = f"{c}_abs_err"
        out2[c_corr] = (x == EXPECTED_Q12[i]).astype(float)
        out2[c_err]  = (x - EXPECTED_Q12[i]).abs()
        corr_cols.append(c_corr); err_cols.append(c_err)
    out2["Q12_correct_count"] = out2[corr_cols].sum(axis=1)
    out2["Q12_abs_err_mean"]  = out2[err_cols].mean(axis=1)

    # 4) 최종 DataFrame (학습과 동일한 열 순서로 결합)
    ordered_cols = (
        [f"Q0{i}_bin" for i in range(1, 9)] +
        sum([[f"Q12_{i}_correct", f"Q12_{i}_abs_err"] for i in range(1, 6)], []) +
        ["Q12_correct_count", "Q12_abs_err_mean"]
    )
    X_row = pd.concat([out1, out2], axis=1)[ordered_cols]
    return X_row


# === 시계열 전처리(프로젝트 코드와 동일 로직) ==================
def _preprocess_activity(x):
    arr = np.asarray(x, dtype=float)
    if arr.size != 288: raise ValueError(f"activity len={arr.size}, need 288")
    arr = np.nan_to_num(arr, nan=1.0)
    return np.clip(arr, 0, 5).astype(int)

def _preprocess_met(x):
    arr = np.asarray(x, dtype=float)
    if arr.size != 288: raise ValueError(f"met len={arr.size}, need 288")
    arr = np.nan_to_num(arr, nan=1.2)
    return np.clip(arr, 0.8, 15.0)

def _preprocess_sleep_hr(x):
    arr = np.asarray(x, dtype=float)
    if arr.size != 288: raise ValueError(f"hr len={arr.size}, need 288")
    arr = np.where((np.isnan(arr)) | (arr <= 0), 60.0, arr)
    return np.clip(arr, 40, 120)

def _preprocess_hypnogram(x):
    arr = np.asarray(x, dtype=float)
    if arr.size != 288: raise ValueError(f"hypno len={arr.size}, need 288")
    arr = np.nan_to_num(arr, nan=4.0)
    return np.clip(arr, 1, 4).astype(int)

def _preprocess_rmssd(x):
    arr = np.asarray(x, dtype=float)
    if arr.size != 288: raise ValueError(f"rmssd len={arr.size}, need 288")
    arr = np.where((np.isnan(arr)) | (arr <= 0), 30.0, arr)
    return np.clip(arr, 0, 200)

def _preprocess_daily16(x):
    arr = np.asarray(x, dtype=float)
    if arr.size != 16: raise ValueError(f"daily len={arr.size}, need 16")
    return np.nan_to_num(arr, nan=0.0)

EXPECTED_Q12 = np.array([93, 86, 79, 72, 65], dtype=float)

def mmse13_to_feature_row(mmse_13):
    """
    mmse_13: array-like(13,) 또는 dict.
      순서 가정: [Q01..Q08, Q12_1..Q12_5]
      값 예시: Q01~Q08은 1/2, Q12_*는 정답 숫자
    반환: pd.DataFrame (1행), 학습 때와 동일한 컬럼 순서
      [Q01_bin..Q08_bin,
       Q12_1_correct, Q12_1_abs_err, ..., Q12_5_correct, Q12_5_abs_err,
       Q12_correct_count, Q12_abs_err_mean]
    """
    # 1) dict로 표준화
    if not isinstance(mmse_13, dict):
        keys = [f"Q0{i}" for i in range(1, 9)] + [f"Q12_{i}" for i in range(1, 6)]
        vals = np.asarray(mmse_13, dtype=float).tolist()
        if len(vals) != 13:
            raise ValueError(f"MMSE 길이가 13이 아닙니다: {len(vals)}")
        mmse_13 = dict(zip(keys, vals))
    df = pd.DataFrame([mmse_13])

    # 2) Q01~Q08 → _bin
    q01_08_cols = [f"Q0{i}" for i in range(1, 9)]
    out1 = pd.DataFrame(index=df.index)
    for c in q01_08_cols:
        x = pd.to_numeric(df.get(c), errors='coerce')
        out1[c + "_bin"] = np.where(x == 2, 1.0, np.where(x == 1, 0.0, np.nan))

    # 3) Q12 → correct/abs_err + 집계 (컬럼 생성 순서 주의: correct → abs_err 순으로 1~5 반복)
    q12_cols = [f"Q12_{i}" for i in range(1, 6)]
    out2 = pd.DataFrame(index=df.index)
    corr_cols, err_cols = [], []
    for i, c in enumerate(q12_cols):
        x = pd.to_numeric(df.get(c), errors='coerce')
        c_corr = f"{c}_correct"
        c_err  = f"{c}_abs_err"
        out2[c_corr] = (x == EXPECTED_Q12[i]).astype(float)
        out2[c_err]  = (x - EXPECTED_Q12[i]).abs()
        corr_cols.append(c_corr); err_cols.append(c_err)
    out2["Q12_correct_count"] = out2[corr_cols].sum(axis=1)
    out2["Q12_abs_err_mean"]  = out2[err_cols].mean(axis=1)

    # 4) 최종 DataFrame (학습과 동일한 열 순서로 결합)
    ordered_cols = (
        [f"Q0{i}_bin" for i in range(1, 9)] +
        sum([[f"Q12_{i}_correct", f"Q12_{i}_abs_err"] for i in range(1, 6)], []) +
        ["Q12_correct_count", "Q12_abs_err_mean"]
    )
    X_row = pd.concat([out1, out2], axis=1)[ordered_cols]
    return X_row


# === 시계열 전처리(프로젝트 코드와 동일 로직) ==================
def _preprocess_activity(x):
    arr = np.asarray(x, dtype=float)
    if arr.size != 288: raise ValueError(f"activity len={arr.size}, need 288")
    arr = np.nan_to_num(arr, nan=1.0)
    return np.clip(arr, 0, 5).astype(int)

def _preprocess_met(x):
    arr = np.asarray(x, dtype=float)
    if arr.size != 288: raise ValueError(f"met len={arr.size}, need 288")
    arr = np.nan_to_num(arr, nan=1.2)
    return np.clip(arr, 0.8, 15.0)

def _preprocess_sleep_hr(x):
    arr = np.asarray(x, dtype=float)
    if arr.size != 288: raise ValueError(f"hr len={arr.size}, need 288")
    arr = np.where((np.isnan(arr)) | (arr <= 0), 60.0, arr)
    return np.clip(arr, 40, 120)

def _preprocess_hypnogram(x):
    arr = np.asarray(x, dtype=float)
    if arr.size != 288: raise ValueError(f"hypno len={arr.size}, need 288")
    arr = np.nan_to_num(arr, nan=4.0)
    return np.clip(arr, 1, 4).astype(int)

def _preprocess_rmssd(x):
    arr = np.asarray(x, dtype=float)
    if arr.size != 288: raise ValueError(f"rmssd len={arr.size}, need 288")
    arr = np.where((np.isnan(arr)) | (arr <= 0), 30.0, arr)
    return np.clip(arr, 0, 200)

def _preprocess_daily16(x):
    arr = np.asarray(x, dtype=float)
    if arr.size != 16: raise ValueError(f"daily len={arr.size}, need 16")
    return np.nan_to_num(arr, nan=0.0)

# ================= 모델 로드 유틸 =================

def load_state_model_from_path(model_path, device=None):
    """
    다음 케이스를 모두 처리:
      A) 모델 객체 통째 저장(torch.save(model, ...))  ← PyTorch 2.6에서 안전 정책 필요
      B) dict에 'state_dict'/'model_state_dict'/'module'/'net' 등으로 저장
      C) 순수 state_dict(dict of tensors) 저장
    반환: (model, 'full' | 'state_dict')
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일이 없습니다: {model_path}")

    real_path = os.path.realpath(model_path)

    # 1) 가장 안전한 경로: tensor들만 로드 시도
    try:
        obj = torch.load(real_path, map_location=device, weights_only=True)
    except Exception:
        # 2) 모델 객체가 피클로 저장된 경우: 우리가 신뢰하는 클래스만 allowlist에 추가 후 재시도
        try:
            add_safe_globals([DementiaClassificationModel])
        except Exception:
            pass
        try:
            obj = torch.load(real_path, map_location=device, weights_only=True)
        except Exception:
            # 3) 마지막 수단: 파일을 '신뢰'할 때만 사용하세요 (任意 코드 실행 위험)
            obj = torch.load(real_path, map_location=device, weights_only=False)

    # 4) 로드 결과 해석
    if isinstance(obj, nn.Module):
        model = obj.to(device).eval()
        return model, 'full'

    if isinstance(obj, dict):
        # dict 안에서 state_dict 찾아보기
        cand_keys = ['state_dict', 'model_state_dict', 'module', 'net', 'model']
        state = None
        for k in cand_keys:
            if k in obj:
                state = obj[k].state_dict() if isinstance(obj[k], nn.Module) else obj[k]
                break
        if state is None and all(torch.is_tensor(v) for v in obj.values()):
            state = obj  # 순수 state_dict

        if state is None:
            raise RuntimeError("체크포인트 형식을 해석할 수 없습니다.")

        model = DementiaClassificationModel(num_classes=2).to(device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        model.eval()
        return model, 'state_dict'

    raise RuntimeError(f"알 수 없는 체크포인트 형식: {type(obj)}")




# ----------------- 최종 앙상블 예측기 -----------------
class DementiaRiskPredictor:
    """
    - TS 모델: best_dementia_model.pth (TorchScript .pt/.pth or pickled nn.Module)
      * forward 시그니처는 다음 인자를 받는다고 가정:
        (activity_seq, met_5min, sleep_hr_seq, sleep_hypno_seq, sleep_rmssd_seq, daily16)
        각 텐서는 배치 1, 길이 288(시계열) / 16(요약) 형태
    - MMSE 모델: mmse_rf.pkl (scikit-learn Pipeline)
    - 최종 확률: ts_weight * p_ts + (1 - ts_weight) * p_mmse
    """
    def __init__(self,
                 ts_model_path="best_dementia_model.pt",
                 mmse_model_path="mmse_rf.pkl",
                 ts_weight=0.6,
                 threshold=0.5,
                 device=None):
        self.device = device or (torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.ts_weight = float(ts_weight)
        self.threshold = float(threshold)

        if not os.path.exists(ts_model_path):
            raise FileNotFoundError(f"TS 모델 파일을 찾을 수 없습니다: {ts_model_path}")
        self.ts_model, self.ts_kind = load_state_model_from_path(ts_model_path, self.device)

        if not os.path.exists(mmse_model_path):
            raise FileNotFoundError(f"MMSE 모델 파일을 찾을 수 없습니다: {mmse_model_path}")
        with open(mmse_model_path, "rb") as f:
            self.mmse_clf = pickle.load(f)

    @torch.inference_mode()
    def predict_one(self,
                    activity_seq, met_5min, sleep_hr_seq, sleep_hypno_seq, sleep_rmssd_seq,
                    daily16, mmse13):
        # ---- TS 확률
        a  = torch.LongTensor(_preprocess_activity(activity_seq)).unsqueeze(0).to(self.device)
        m  = torch.FloatTensor(_preprocess_met(met_5min)).unsqueeze(0).to(self.device)
        hr = torch.FloatTensor(_preprocess_sleep_hr(sleep_hr_seq)).unsqueeze(0).to(self.device)
        hy = torch.LongTensor(_preprocess_hypnogram(sleep_hypno_seq)).unsqueeze(0).to(self.device)
        rm = torch.FloatTensor(_preprocess_rmssd(sleep_rmssd_seq)).unsqueeze(0).to(self.device)
        d  = torch.FloatTensor(_preprocess_daily16(daily16)).unsqueeze(0).to(self.device)

        logits = self.ts_model(a, m, hr, hy, rm, d)
        if logits.dim() == 2 and logits.size(-1) == 2:
            p_ts = float(F.softmax(logits, dim=1)[0, 1].item())
        elif logits.dim() == 2 and logits.size(-1) == 1:
            p_ts = float(torch.sigmoid(logits)[0, 0].item())
        else:
            # TorchScript 모델의 반환이 dict 등인 경우 키 추정 (선택)
            try:
                p_ts = float(logits["prob_risk"])
            except Exception:
                raise RuntimeError("TS 모델 출력 형식을 해석할 수 없습니다. (2-class logits 또는 1-logit sigmoid 필요)")

        # ---- MMSE 확률
        X_row = mmse13_to_feature_row(mmse13)  # (1, F)
        p_mmse = float(self.mmse_clf.predict_proba(X_row)[:, 1][0])

        # ---- 앙상블
        w = self.ts_weight
        p_final = float(np.clip(w * p_ts + (1.0 - w) * p_mmse, 0.0, 1.0))
        y_pred  = int(p_final >= self.threshold)
        return {
            "risk_probability": p_final,
            "risk_class": "Risk" if y_pred == 1 else "Normal",
            # "confidence": float(abs(p_final - 0.5) * 2.0),
            # "details": {
            #     "p_ts": p_ts, "p_mmse": p_mmse,
            #     "ts_weight": w, "threshold": self.threshold,
            #     "ts_model_format": self.ts_kind
            # }
        }

# ----------------- 사용 예시 -----------------
# === JSON config에서 한 케이스를 읽어오는 헬퍼 ===
def load_case_from_json(config_path, case_id):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # 케이스 선택
    if isinstance(cfg, dict) and "cases" in cfg:
        items = cfg["cases"]
        match = next((it for it in items if str(it.get("id")) == str(case_id)), None)
        if match is None:
            raise KeyError(f"config.cases에서 id='{case_id}'를 찾지 못했습니다.")
        data = match
    elif isinstance(cfg, dict) and case_id in cfg:
        data = cfg[case_id]
    else:
        raise ValueError("config 포맷이 올바르지 않습니다. 'cases' 배열 또는 최상위 키로 케이스를 제공하세요.")

    # 필수 키 확인
    fields = ["activity_seq","met_5min","sleep_hr_seq","sleep_hypno_seq","sleep_rmssd_seq","daily16","mmse13"]
    missing = [k for k in fields if k not in data]
    if missing:
        raise KeyError(f"config에 누락된 키: {missing}")

    # numpy 배열로 변환 + 길이 검증 + 결측치 처리
    def to_arr(k, length, dtype=float):
        arr = np.asarray(data[k], dtype=dtype)
        
        # 길이 조정
        if arr.size != length:
            if arr.size < length:
                padding = np.full(length - arr.size, np.nan, dtype=dtype)
                arr = np.concatenate([arr, padding])
            elif arr.size > length:
                arr = arr[:length]
        return arr

    return {
        "activity_seq":    to_arr("activity_seq",    288, float),
        "met_5min":        to_arr("met_5min",        288, float),
        "sleep_hr_seq":    to_arr("sleep_hr_seq",    288, float),
        "sleep_hypno_seq": to_arr("sleep_hypno_seq", 288, float),
        "sleep_rmssd_seq": to_arr("sleep_rmssd_seq", 288, float),
        "daily16":         to_arr("daily16",          16, float),
        "mmse13":          to_arr("mmse13",           13, float),
    }


# ----------------- 사용 예시 -----------------
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    predictor = DementiaRiskPredictor(
        # 파일 위치에 맞게 경로 조정
        ts_model_path=os.path.join(BASE_DIR, "best_dementia_model_full.pth"),
        mmse_model_path=os.path.join(BASE_DIR, "mmse_rf.pkl"),
        ts_weight=0,
        threshold=0.45
    )

    # JSON 설정에서 두 케이스를 읽어 순차 예측
    CONFIG_PATH = os.path.join(BASE_DIR, "user_config2.json")  # ← JSON 설정 파일명
    sample_risk   = load_case_from_json(CONFIG_PATH, "risk")
    sample_normal = load_case_from_json(CONFIG_PATH, "normal")
    out_risk   = predictor.predict_one(**sample_risk)
    out_normal = predictor.predict_one(**sample_normal)
    print("[risk]  ", out_risk)
    print("[normal]", out_normal)