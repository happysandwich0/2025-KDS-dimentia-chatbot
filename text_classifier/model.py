import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.metrics import classification_report, roc_curve, precision_recall_curve, f1_score
from typing import Tuple, Dict

def _tune_thresholds_by_pr(y_true_bin: np.ndarray, y_prob_mat: np.ndarray, label_names: list) -> list:
    """Precision-Recall Curve 기반 최적 F1-score 임계값 튜닝"""
    best_thr = []
    for j, _ in enumerate(label_names):
        y_true = y_true_bin[:, j]; y_prob = y_prob_mat[:, j]
        if y_true.max() == y_true.min(): best_thr.append(0.5); continue
        p, r, t = precision_recall_curve(y_true, y_prob)
        if len(t) == 0: best_thr.append(0.5); continue
        f1s = [f1_score(y_true, (y_prob >= thr).astype(int), zero_division=0) for thr in t]
        best_thr.append(float(t[int(np.argmax(f1s))]))
    return best_thr

def _tune_thresholds_by_roc(y_true_bin: np.ndarray, y_prob_mat: np.ndarray) -> list:
    """ROC Curve (TPR-FPR 최대) 기반 최적 임계값 튜닝 (이진 분류용)"""
    best_thr = []
    for j in range(y_prob_mat.shape[1]):
        y_true = y_true_bin[:, j]; y_prob = y_prob_mat[:, j]
        if y_true.max() == y_true.min(): best_thr.append(0.5); continue
        fpr, tpr, t = roc_curve(y_true, y_prob)
        best_thr.append(float(t[np.argmax(tpr - fpr)]) if len(t) else 0.5)
    return best_thr


# ==============================================================================
## 1. 감성 분류 (Affect)
# ==============================================================================

class AffectSentDataset(Dataset):
    def __init__(self, df, y_cols, tokenizer, max_len):
        self.texts = df['text'].fillna("").tolist()
        self.labels = df[y_cols].astype('float32').values
        self.max_len = max_len
        self.tokenizer = tokenizer
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        enc = self.tokenizer(self.texts[i], padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt")
        attn = enc["attention_mask"].squeeze(0)
        token_len = (attn.sum().float()/self.max_len)
        return {
            "input_ids": enc["input_ids"].squeeze(0), "attention_mask": attn,
            "labels": torch.tensor(self.labels[i], dtype=torch.float32), "token_len": token_len
        }

class AffectMultiLabelHead(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        h = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(h + 1, num_labels) # CLS + token_len
    def forward(self, input_ids, attention_mask, token_len):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:,0]
        x = torch.cat([cls, token_len.unsqueeze(1)], dim=1)
        return self.classifier(self.dropout(x))

def train_affect_model(train_df, val_df, config: Dict, device: torch.device):
    """감성 분류 모델 학습 및 최적 임계값 튜닝"""
    Y_COLS = config['Y_COLS']
    MAX_LEN, BATCH, EPOCHS, LR = config['MAX_LEN'], config['BATCH'], config['EPOCHS'], config['LR']
    MODEL_NAME, CKPT_PATH = config['MODEL_NAME'], config['CKPT_PATH']
    LABEL_NAMES = [c.replace('y_','') for c in Y_COLS]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AffectMultiLabelHead(MODEL_NAME, num_labels=len(Y_COLS)).to(device)

    train_loader = DataLoader(AffectSentDataset(train_df, Y_COLS, tokenizer, MAX_LEN), batch_size=BATCH, shuffle=True, num_workers=2)
    val_loader   = DataLoader(AffectSentDataset(val_df, Y_COLS, tokenizer, MAX_LEN), batch_size=BATCH*2, shuffle=False, num_workers=2)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    def train_one_epoch():
        model.train(); total=0.0
        for b in train_loader:
            optimizer.zero_grad()
            logits = model(b["input_ids"].to(device), b["attention_mask"].to(device), b["token_len"].to(device))
            loss = criterion(logits, b["labels"].to(device))
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); total += loss.item()
        return total/len(train_loader)

    @torch.no_grad()
    def evaluate(thresholds=None, return_raw=False):
        model.eval(); total=0.0; P, Y = [], []
        for b in val_loader:
            logits = model(b["input_ids"].to(device), b["attention_mask"].to(device), b["token_len"].to(device))
            loss = criterion(logits, b["labels"].to(device)); total += loss.item()
            P.append(torch.sigmoid(logits).cpu().numpy()); Y.append(b["labels"].cpu().numpy())
        P = np.vstack(P); Y = np.vstack(Y)

        thr = np.asarray(thresholds)[None, :] if thresholds is not None else 0.5
        preds = (P >= thr).astype(int)

        rep = classification_report(Y, preds, target_names=LABEL_NAMES, output_dict=True, zero_division=0)
        if return_raw: return total/len(val_loader), rep, P, Y
        return total/len(val_loader), rep

    best_f1 = -1.0; best_thr = None

    for epoch in range(1, EPOCHS+1):
        tr = train_one_epoch()
        if best_thr is None:
            va, rep, P, Y = evaluate(return_raw=True, thresholds=None)
            best_thr = _tune_thresholds_by_roc(Y, P)
            va, rep = evaluate(thresholds=best_thr)
        else:
            va, rep = evaluate(thresholds=best_thr)

        macro_f1 = rep["macro avg"]["f1-score"]
        print(f"[E{epoch}] train {tr:.4f} | val {va:.4f} | macroF1 {macro_f1:.4f}")

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save({"model_state": model.state_dict(), "model_name": MODEL_NAME, "classes": LABEL_NAMES, "thresholds": best_thr, "y_cols": Y_COLS}, CKPT_PATH)
            print(f"→ Best updated ({best_f1:.4f}) saved to {CKPT_PATH}")

    return tokenizer


# ==============================================================================
## 2. 같은말반복 (Repeat)
# ==============================================================================

class RepeatDataset(Dataset):
    def __init__(self, df, y_col, tokenizer, max_len):
        self.texts = df["text"].astype(str).tolist()
        self.labels = df[y_col].astype(float).values
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.y_col = y_col
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt")
        attn = enc["attention_mask"].squeeze(0)
        token_len = attn.sum().float() / self.max_len
        return {
            "input_ids": enc["input_ids"].squeeze(0), "attention_mask": attn,
            "token_len": token_len, "label": torch.tensor(self.labels[idx], dtype=torch.float),
        }

class RepeatBinaryHead(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        h = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(h + 1, 1) # CLS + token_len
    def forward(self, input_ids, attention_mask, token_len):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        x = torch.cat([cls, token_len.unsqueeze(1)], dim=1)
        return self.classifier(self.dropout(x)).squeeze(1)

def train_repeat_model(train_df, val_df, config: Dict, device: torch.device):
    """같은말반복 이진 분류 모델 학습 (Weighted Sampler 사용)"""
    LABEL = config['LABEL']
    MAX_LEN, BATCH, EPOCHS, LR = config['MAX_LEN'], config['BATCH'], config['EPOCHS'], config['LR']
    MODEL_NAME, CKPT_PATH = config['MODEL_NAME'], config['CKPT_PATH']

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = RepeatBinaryHead(MODEL_NAME).to(device)

    # Weighted Random Sampler (데이터 불균형 해소)
    y = train_df[LABEL].astype(int).values
    pos_n, neg_n = (y==1).sum(), (y==0).sum()
    w_pos = 0.5 / max(pos_n,1); w_neg = 0.5 / max(neg_n,1)
    weights = np.where(y==1, w_pos, w_neg)
    sampler = WeightedRandomSampler(weights=torch.DoubleTensor(weights), num_samples=len(y), replacement=True)

    train_ds = RepeatDataset(train_df, y_col=LABEL, tokenizer=tokenizer, max_len=MAX_LEN)
    val_ds   = RepeatDataset(val_df, y_col=LABEL, tokenizer=tokenizer, max_len=MAX_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH, sampler=sampler, shuffle=False, num_workers=2)
    val_loader   = DataLoader(val_ds, batch_size=BATCH*2, shuffle=False, num_workers=2)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    @torch.no_grad()
    def evaluate(thr=0.5, return_raw=False):
        model.eval(); total = 0.0; P=[]; Y=[]
        for b in val_loader:
            logits = model(b["input_ids"].to(device), b["attention_mask"].to(device), b["token_len"].to(device))
            loss = criterion(logits, b["label"].to(device)); total += loss.item()
            P.append(torch.sigmoid(logits).cpu().numpy()); Y.append(b["label"].cpu().numpy())
        P = np.concatenate(P); Y = np.concatenate(Y)
        preds = (P >= thr).astype(int)
        rep = classification_report(Y, preds, target_names=[f"{LABEL}=1",f"{LABEL}=0"], output_dict=True, zero_division=0)
        if return_raw: return total/len(val_loader), rep, P, Y
        return total/len(val_loader), rep

    best_f1, best_thr = -1.0, 0.5
    for epoch in range(1, EPOCHS+1):
        model.train(); total=0.0
        for b in train_loader:
            optimizer.zero_grad()
            logits = model(b["input_ids"].to(device), b["attention_mask"].to(device), b["token_len"].to(device))
            loss = criterion(logits, b["label"].to(device)); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); total += loss.item()
        tr_loss = total/len(train_loader)

        if epoch == 1:
            va, rep, P, Y = evaluate(return_raw=True)
            best_thr = _tune_thresholds_by_roc(Y[:, None], P[:, None])[0]

        va_loss, rep = evaluate(thr=best_thr)
        macro_f1 = rep["macro avg"]["f1-score"]
        print(f"[E{epoch}] train {tr_loss:.4f} | val {va_loss:.4f} | thr {best_thr:.3f} | macroF1 {macro_f1:.4f}")

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save({"model_state": model.state_dict(), "model_name": MODEL_NAME, "label": LABEL, "threshold": best_thr, "max_len": MAX_LEN}, CKPT_PATH)
            print(f"→ Best updated ({best_f1:.4f}) saved to {CKPT_PATH}")

    return tokenizer


# ==============================================================================
## 3. 위험지표 (Risk / 4개 구체성)
# ==============================================================================

class RiskStoryDataset(Dataset):
    def __init__(self, df, tokenizer, label_cols, max_len):
        self.texts = df["text"].astype(str).tolist()
        self.labels = df[label_cols].astype(float).values
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_cols = label_cols
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt")
        attn = enc["attention_mask"].squeeze(0)
        token_len = attn.sum().float() / self.max_len
        return {
            "input_ids": enc["input_ids"].squeeze(0), "attention_mask": attn,
            "labels": torch.tensor(self.labels[idx], dtype=torch.float), "token_len": token_len,
        }

class RiskMultiLabelHead(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        h = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(h + 1, num_labels) # CLS + token_len
    def forward(self, input_ids, attention_mask, token_len):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        x = torch.cat([cls, token_len.unsqueeze(1)], dim=1)
        return self.classifier(self.dropout(x))

def train_risk_model(train_df, val_df, config: Dict, device: torch.device):
    """스토리 위험지표(4개 구체성) 다중 라벨 분류 모델 학습"""
    TARGET_COLS = config['TARGET_COLS']
    MAX_LEN, BATCH, EPOCHS, LR = config['MAX_LEN'], config['BATCH'], config['EPOCHS'], config['LR']
    MODEL_NAME, CKPT_PATH = config['MODEL_NAME'], config['CKPT_PATH']

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = RiskMultiLabelHead(MODEL_NAME, num_labels=len(TARGET_COLS)).to(device)

    train_ds = RiskStoryDataset(train_df, tokenizer, TARGET_COLS, MAX_LEN)
    val_ds   = RiskStoryDataset(val_df, tokenizer, TARGET_COLS, MAX_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    def train_one_epoch():
        model.train(); total=0.0
        for b in train_loader:
            optimizer.zero_grad()
            logits = model(b["input_ids"].to(device), b["attention_mask"].to(device), b["token_len"].to(device))
            loss = criterion(logits, b["labels"].to(device)); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); total += loss.item()
        return total/len(train_loader)

    @torch.no_grad()
    def validate(thresholds=None, return_raw=False):
        model.eval(); total=0.0; probs_list, labels_list = [], []
        for b in val_loader:
            logits = model(b["input_ids"].to(device), b["attention_mask"].to(device), b["token_len"].to(device))
            loss = criterion(logits, b["labels"].to(device)); total += loss.item()
            probs_list.append(torch.sigmoid(logits).cpu().numpy()); labels_list.append(b["labels"].cpu().numpy())
        P = np.vstack(probs_list); Y = np.vstack(labels_list)

        thr = np.asarray(thresholds)[None, :] if thresholds is not None else 0.5
        preds = (P >= thr).astype(int)

        rep = classification_report(Y, preds, target_names=TARGET_COLS, output_dict=True, zero_division=0)
        if return_raw: return total/len(val_loader), rep, P, Y
        return total/len(val_loader), rep

    best_macro_f1 = -1.0; best_thresholds = None

    for epoch in range(1, EPOCHS+1):
        tr_loss = train_one_epoch()
        if best_thresholds is None:
            va_loss, rep, P, Y = validate(return_raw=True)
            tuned = _tune_thresholds_by_pr(Y, P, TARGET_COLS) # F1-score 기반 튜닝
            va_loss, rep = validate(thresholds=tuned); cur_thresholds = tuned
        else:
            va_loss, rep = validate(thresholds=best_thresholds); cur_thresholds = best_thresholds

        macro_f1 = rep["macro avg"]["f1-score"]
        print(f"[E{epoch}] train {tr_loss:.4f} | val {va_loss:.4f} | macroF1 {macro_f1:.4f}")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_thresholds = cur_thresholds
            torch.save({"model_state": model.state_dict(), "model_name": MODEL_NAME, "target_cols": TARGET_COLS, "thresholds": best_thresholds, "max_len": MAX_LEN}, CKPT_PATH)
            print(f"→ Best updated ({best_macro_f1:.4f}) saved to {CKPT_PATH}")

    return tokenizer