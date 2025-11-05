import torch
import torch.nn as nn
from transformers import AutoModel
import torch.utils.data as Data
import numpy as np
from tqdm import tqdm

class MultiLabelClassifier(nn.Module):
    """
    사전 학습이 완료된 멀티 라벨 분류기 모델을 로드
    user data 텍스트의 **사건, 공간, 시간 구체성**을 판별하여 라벨링함

    사용 모델 : model_name = "beomi/KcELECTRA-base"
    모델 경로 : label_model_path = "pret_multilabel.pt"
    """
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.backbone.config.hidden_size + 1, num_labels)

    def forward(self, input_ids, attention_mask, token_len):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        token_len_tensor = token_len.unsqueeze(1)
        combined_features = torch.cat((pooled_output, token_len_tensor), dim=1)
        logits = self.classifier(self.dropout(combined_features))
        return logits
    
class RepeatBinaryHead(nn.Module):
    """
    사전 학습이 완료된 단일 라벨 분류기 모델을 로드
    user data 텍스트의 **같은 말 반복 여부**를 판별하여 라벨링함

    사용 모델 : MODEL_NAME = "monologg/koelectra-base-v3-discriminator"
    모델 경로 : label_model_path = "pret_repeat.pt"
    """
    def __init__(self, model_name):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        h = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(h + 1, 1)   # + token_len 
    def forward(self, input_ids, attention_mask, token_len, quality):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]         
        x = torch.cat([cls, token_len.unsqueeze(1), quality.unsqueeze(1)], dim=1)
        return self.classifier(self.dropout(x)).squeeze(1)
    
class MultiLabelHead(nn.Module):
    """
    사전 학습이 완료된 다중 라벨 분류기 모델을 로드
    user data 텍스트의 **감성**를 판별하여 라벨링함

    사용 모델 : MODEL_NAME = "klue/roberta-base"
    모델 경로 : label_model_path = "pret_emotion.pt"
    """
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        h = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(h + 1, num_labels)  # +1: token_len 보
    def forward(self, input_ids, attention_mask, token_len):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:,0]
        x = torch.cat([cls, token_len.unsqueeze(1)], dim=1)
        return self.classifier(self.dropout(x))

class AutoEncoder(nn.Module):
    """
    이상치 탐지를 위한 오토인코더 모델.
    """
    def __init__(self, input_dim, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32), 
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out, z

def predict_batch_multilabel(texts, model_name, model_path, tokenizer, device, max_len=512, batch_size=32, num_labels=4):
    """
    멀티라벨 분류기(4개 라벨)로 텍스트 리스트에 대해 배치 단위로 라벨을 예측
    """
    model = MultiLabelClassifier(model_name, num_labels).to(device)
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        state_dict = checkpoint['model_state']
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"모델 가중치 로드 중 오류 발생. {e}")
        return None
    
    model.eval()
    all_preds = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Prediction MultiLabel"):
        batch_texts = texts[i:i+batch_size]
        encoding = tokenizer(
            batch_texts,
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
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_len=token_len)
            probabilities = torch.sigmoid(outputs).cpu().numpy()
            binary_preds = (probabilities >= 0.5).astype(int)
            all_preds.extend(binary_preds)
            
    return np.array(all_preds)

def predict_batch_repeat(texts, model_name, model_path, tokenizer, device, max_len=512, batch_size=32):
    """
    단일 라벨 분류기로 텍스트 리스트에 대해 배치 단위로 **같은 말 반복 여부** 라벨을 예측
    """
    model = RepeatBinaryHead(model_name).to(device)
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        state_dict = checkpoint['model_state']
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"모델 가중치 로드 중 오류 발생. {e}")
        return None
    
    model.eval()
    all_preds = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Prediction Repeat"):
        batch_texts = texts[i:i+batch_size]
        encoding = tokenizer(
            batch_texts,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        token_len = torch.sum(attention_mask, dim=1).to(torch.float32) / max_len
        token_len = token_len.to(device)

        # 배치 크기 조절
        current_batch_size = len(batch_texts)
        current_quality_dummy = torch.ones(current_batch_size, device=device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_len=token_len)
            probabilities = torch.sigmoid(outputs).cpu().numpy()
            binary_preds = (probabilities >= 0.5).astype(int)
            all_preds.extend(binary_preds)
            
    return np.array(all_preds).reshape(-1, 1)

def predict_batch_emotion(texts, model_name, model_path, tokenizer, device, max_len=512, batch_size=32, num_labels=3):
    """
    멀티 라벨 분류기로 텍스트 리스트에 대해 배치 단위로 **감성** 라벨을 예측
    """
    model = MultiLabelHead(model_name, num_labels).to(device)
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        state_dict = checkpoint['model_state']
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"모델 가중치 로드 중 오류 발생. {e}")
        return None
    
    model.eval()
    all_preds = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Prediction Emotion"):
        batch_texts = texts[i:i+batch_size]
        encoding = tokenizer(
            batch_texts,
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
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_len=token_len)
            probabilities = torch.sigmoid(outputs).cpu().numpy()
            binary_preds = (probabilities >= 0.5).astype(int)
            all_preds.extend(binary_preds)
            
    return np.array(all_preds)

def train_autoencoder(data_tensor, num_features, epochs=50):
    """
    정상 데이터를 사용하여 오토인코더 모델을 학습
    """
    model = AutoEncoder(input_dim=num_features, latent_dim=32)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    dataset = Data.TensorDataset(data_tensor)
    loader = Data.DataLoader(dataset, batch_size=64, shuffle=True)
    
    print("오토인코더 모델 학습 시작")
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs = batch[0]
            outputs, _ = model(inputs)
            loss = criterion(outputs, inputs)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    
    return model

def detect_outliers(model, all_data, normal_data):
    """
    오토인코더를 사용하여 이상치를 탐지하고 재구성 오차, 이상치 여부, 임계값을 반환
    """
    model.eval()
    bce_loss = nn.BCELoss(reduction='none')

    with torch.no_grad():
        recon, _ = model(all_data)
        bce_errors = bce_loss(recon, all_data)
        recon_error_all = torch.mean(bce_errors, dim=1).numpy()
    
        recon_normal, _ = model(normal_data)
        normal_bce_errors = bce_loss(recon_normal, normal_data)
        recon_error_normal = torch.mean(normal_bce_errors, dim=1).numpy()

    mean_err = np.mean(recon_error_normal)
    std_err = np.std(recon_error_normal)
    threshold = mean_err + 1.96 * std_err

    outliers = recon_error_all > threshold
    return recon_error_all, outliers, threshold