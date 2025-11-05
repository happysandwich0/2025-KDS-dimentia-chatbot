import math
import torch
from torch import nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]
    
class ActivityEncoder(nn.Module):
    def __init__(self, vocab_size=7, embed_dim=16, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1d = nn.Conv1d(embed_dim, 32, kernel_size=5, padding=2)
        self.lstm = nn.LSTM(32, hidden_dim, batch_first=True, bidirectional=True)
        self.output_proj = nn.Linear(hidden_dim * 2, 128)

    def forward(self, x):
        # x: (batch, 288)
        embedded = self.embedding(x)  # (batch, 288, 16)
        conv_out = F.relu(self.conv1d(embedded.transpose(1,2)))  # (batch, 32, 288)
        lstm_out, _ = self.lstm(conv_out.transpose(1,2))  # (batch, 288, 128)
        return self.output_proj(lstm_out.mean(dim=1))  # Global average pooling
    

class METEncoder(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.res_blocks = nn.ModuleList([
            self._make_res_block(hidden_dim) for _ in range(3)
        ])
        self.attention = nn.MultiheadAttention(hidden_dim, 8, batch_first=True)
        self.output_proj = nn.Linear(hidden_dim, 128)

    def _make_res_block(self, dim):
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.LayerNorm(dim),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        # x: (batch, 288, 1)
        x = self.input_proj(x.unsqueeze(-1))

        for res_block in self.res_blocks:
            x = x + res_block(x)

        attn_out, _ = self.attention(x, x, x)
        return self.output_proj(attn_out.mean(dim=1))
    

class SleepHREncoder(nn.Module):
    def __init__(self, d_model=64, nhead=8, num_layers=4):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead,
                                                   dim_feedforward=256,
                                                   dropout=0.1,
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_proj = nn.Linear(d_model, 128)

    def forward(self, x):
        # x: (batch, 288)
        x = self.input_proj(x.unsqueeze(-1))  # (batch, 288, d_model)
        x = self.pos_encoding(x)
        transformer_out = self.transformer(x)
        return self.output_proj(transformer_out.mean(dim=1))
    

import torch

class HypnogramEncoder(nn.Module):
    def __init__(self, num_states=5, embed_dim=32, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(num_states, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim + 1, hidden_dim, 2,
                          batch_first=True, bidirectional=True)
        self.transition_proj = nn.Linear(embed_dim, 1)
        self.output_proj = nn.Linear(hidden_dim * 2, 128)

    def forward(self, x):
        # x: (batch, 288)
        embedded = self.embedding(x.long())  # (batch, 288, embed_dim)

        # 상태 전환 정보 추가
        transitions = torch.diff(x.float(), dim=1, prepend=x[:, :1])
        transitions = (transitions != 0).float().unsqueeze(-1)

        combined = torch.cat([embedded, transitions], dim=-1)
        gru_out, _ = self.gru(combined)
        return self.output_proj(gru_out.mean(dim=1))
    

class RMSSDEncoder(nn.Module):
    def __init__(self, wavelet_features=64, cnn_channels=[32, 64, 128]):
        super().__init__()
        # 웨이블릿 변환을 시뮬레이션하는 다중 스케일 CNN
        self.multi_scale_convs = nn.ModuleList([
            nn.Conv1d(1, 16, kernel_size=k, padding=k//2)
            for k in [3, 7, 15, 31]  # 다양한 시간 스케일
        ])

        conv_layers = []
        in_ch = 64  # 16 * 4 scales
        for out_ch in cnn_channels:
            conv_layers.extend([
                nn.Conv1d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(2)
            ])
            in_ch = out_ch

        self.conv_layers = nn.Sequential(*conv_layers)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.output_proj = nn.Linear(cnn_channels[-1], 128)

    def forward(self, x):
        # x: (batch, 288)
        x = x.unsqueeze(1)  # (batch, 1, 288)

        # 다중 스케일 특성 추출
        multi_scale_features = []
        for conv in self.multi_scale_convs:
            features = F.relu(conv(x))
            multi_scale_features.append(features)

        combined = torch.cat(multi_scale_features, dim=1)  # (batch, 64, 288)
        conv_out = self.conv_layers(combined)
        pooled = self.adaptive_pool(conv_out).squeeze(-1)
        return self.output_proj(pooled)
    

class DailySummaryEncoder(nn.Module):
    def __init__(self, input_dim=16, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 128)
        )

    def forward(self, x):
        return self.encoder(x)
    


### 시계열 예측 모델 클래스 ###
class DementiaClassificationModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # 시계열 인코더들 (기존과 동일)
        self.activity_encoder = ActivityEncoder()
        self.met_encoder = METEncoder()
        self.sleep_hr_encoder = SleepHREncoder()
        self.hypnogram_encoder = HypnogramEncoder()
        self.rmssd_encoder = RMSSDEncoder()

        # 하루요약통계 인코더
        self.daily_summary_encoder = DailySummaryEncoder()

        # Attention 메커니즘
        self.attention = nn.MultiheadAttention(128, 8, batch_first=True)

        # 개선된 분류기 (과적합 방지)
        total_features = 128 * 6
        self.classifier = nn.Sequential(
            nn.Linear(total_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),  # 더 강한 드롭아웃

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.4),

            nn.Linear(128, 64),  # 더 작은 레이어
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, num_classes)  # 마지막 레이어만 클래스 개수
        )

    def forward(self, activity_seq, met_5min, sleep_hr_seq, sleep_hypno_seq,
                sleep_rmssd_seq, daily_summary):
        # 시계열 데이터 인코딩
        activity_features = self.activity_encoder(activity_seq)
        met_features = self.met_encoder(met_5min)
        sleep_hr_features = self.sleep_hr_encoder(sleep_hr_seq)
        hypno_features = self.hypnogram_encoder(sleep_hypno_seq)
        rmssd_features = self.rmssd_encoder(sleep_rmssd_seq)

        # 하루요약통계 인코딩
        daily_features = self.daily_summary_encoder(daily_summary)

        # 모든 특성 결합
        seq = torch.stack([
            activity_features, met_features, sleep_hr_features,
            hypno_features, rmssd_features, daily_features
        ], dim=1)  # (B, 6, 128)

        attn_out, _ = self.attention(seq, seq, seq)  # (B, 6, 128)
        all_features = attn_out.reshape(attn_out.size(0), -1)  # (B, 6*128)

        output = self.classifier(all_features)
        return output