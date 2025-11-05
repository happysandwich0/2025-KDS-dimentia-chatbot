import torch
import torch.nn as nn
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
import json
import os
from tqdm import tqdm

from anomaly_detection.model import MultiLabelClassifier, AutoEncoder


def load_models_with_finetuning_check(label_model_path, autoencoder_model_path, model_name, target_cols, device):
    """
    학습된 라벨 예측 모델과 개인화(fine-tuned)된 오토인코더 모델이 있으면 로드, 없으면 기본 모델을 로드.
    """
    num_labels = len(target_cols)
    label_model = MultiLabelClassifier(model_name, num_labels).to(device)
    autoencoder_model = AutoEncoder(input_dim=num_labels, latent_dim=32).to(device)

    try:
        checkpoint = torch.load(label_model_path, map_location=device, weights_only=False)
        label_model.load_state_dict(checkpoint['model_state'])
        label_model.eval()
    except Exception as e:
        print(f"'{label_model_path}' 모델 로드 오류. {e}")
        return None, None, None

    user_model_path = "autoencoder_for_user.pth"
    user_threshold_path = "user_threshold.txt"
    if os.path.exists(user_model_path):
        try:
            autoencoder_model.load_state_dict(torch.load(user_model_path, map_location=device))
            autoencoder_model.eval()
            with open(user_threshold_path, "r") as f:
                threshold = float(f.read())
            print(f"개인화된 오토인코더 모델을 '{user_model_path}'에서 로드.")
            return label_model, autoencoder_model, threshold
        except Exception as e:
            print(f"개인화된 오토인코더 모델 로드 오류, 기존 모델로 대체. {e}")
    
    try:
        autoencoder_model.load_state_dict(torch.load(autoencoder_model_path, map_location=device))
        autoencoder_model.eval()
        initial_threshold = 0.057396
        print(f"기본 오토인코더 모델을 '{autoencoder_model_path}'에서 로드.")
        return label_model, autoencoder_model, initial_threshold
    except Exception as e:
        print(f" '{autoencoder_model_path}' 기본 오토인코더 모델 로드 오류. {e}")
        return None, None, None


def classify_and_detect_outlier(text, label_model, autoencoder_model, tokenizer, label_cols, threshold, device):
    """
    입력된 텍스트에 대해 라벨을 예측하고 이상치 여부를 판별.
    """
    max_len = 512
    encoding = tokenizer(
        [text],
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    token_len = torch.sum(attention_mask, dim=1).to(torch.float32) / max_len
    token_len = token_len.to(device)

    with torch.no_grad():
        label_outputs = label_model(input_ids=input_ids, attention_mask=attention_mask, token_len=token_len)
        probabilities = torch.sigmoid(label_outputs).cpu().numpy()
        
        print("\n--- 디버깅 정보: 라벨 예측 확률 (Probabilities) ---")
        prob_dict = {col: prob for col, prob in zip(label_cols, probabilities[0])}
        print(prob_dict)
        print("--------------------------------------------------")
        binary_preds = (probabilities >= 0.5).astype(int)
        
    predicted_labels_tensor = torch.tensor(binary_preds, dtype=torch.float32).to(device)
    bce_loss = nn.BCELoss(reduction='none')
    
    with torch.no_grad():
        recon_output, _ = autoencoder_model(predicted_labels_tensor)
        recon_error = torch.mean(bce_loss(recon_output, predicted_labels_tensor), dim=1).cpu().numpy()[0]
    
    is_outlier = recon_error > threshold
    
    predicted_labels_dict = {col: int(binary_preds[0][i]) for i, col in enumerate(label_cols)}
    return {
        "text": text,
        "predicted_labels": predicted_labels_dict,
        "reconstruction_error": float(recon_error),
        "is_outlier": bool(is_outlier),
        "threshold": float(threshold)
    }

def finetune_user_autoencoder(user_data_path="user_data.json", num_features=7, epochs=100, device="cpu"):
    """
    JSON 파일에 저장된 사용자 데이터를 바탕으로 오토인코더를 fine-tuning함
    ** 개인화(fine-tuned)된 이상치 탐지 모델을 생성 **
    """
    if not os.path.exists(user_data_path):
        print(f"Error: '{user_data_path}' 파일이 존재하지 않음.")
        return None

    with open(user_data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if len(data) < 10:
        print(f"Warning: 사용자 데이터가 10개 미만이므로 fine-tuning을 건너뜀. (현재 {len(data)}개)")
        return None

    user_labels = [list(entry['predicted_labels'].values()) for entry in data]
    user_labels_tensor = torch.tensor(user_labels, dtype=torch.float32).to(device)

    autoencoder_model = AutoEncoder(input_dim=num_features, latent_dim=32).to(device)
    try:
        autoencoder_model.load_state_dict(torch.load("autoencoder_model.pth", map_location=device))
        print("기존 오토인코더 모델을 로드하여 fine-tuning을 시작.")
    except Exception as e:
        print(f"기존 오토인코더 모델 로드 실패. {e}")
        return None

    # fine-tuning 시작
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(autoencoder_model.parameters(), lr=1e-4)

    dataset = torch.utils.data.TensorDataset(user_labels_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(user_labels_tensor), shuffle=True)
    
    for epoch in tqdm(range(epochs), desc="Fine-tuning AutoEncoder"):
        for batch in loader:
            inputs = batch[0]
            outputs, _ = autoencoder_model(inputs)
            loss = criterion(outputs, inputs)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    new_model_path = "autoencoder_for_user.pth"
    torch.save(autoencoder_model.state_dict(), new_model_path)
    print(f"Fine-tuning된 오토인코더 모델이 '{new_model_path}'에 저장됨.")
    
    bce_loss = nn.BCELoss(reduction='none')
    with torch.no_grad():
        recon, _ = autoencoder_model(user_labels_tensor)
        bce_errors = bce_loss(recon, user_labels_tensor)
        recon_error_normal = torch.mean(bce_errors, dim=1).numpy()
    mean_err = np.mean(recon_error_normal)
    std_err = np.std(recon_error_normal)
    new_threshold = mean_err + 1.96 * std_err

    with open("user_threshold.txt", "w") as f:
        f.write(str(new_threshold))

    print(f"새로운 사용자 데이터에 대한 이상치 임계값({new_threshold:.6f})이 저장됨.")
    
    return new_model_path

def save_to_json_and_finetune(data_to_save, file_path="user_data.json"):
    """
    데이터를 JSON 파일에 추가하고, 10개 이상 쌓이면 fine-tuning 함수를 호출.
    ** 개인화된 이상치 탐지 모델에서 판별 수행 **
    """
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = []

    data.append(data_to_save)

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"'{data_to_save['text']}'에 대한 분석 결과가 '{file_path}'에 저장됨. (총 {len(data)}개)")

    if len(data) >= 10:
        finetune_user_autoencoder()
