import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer
import numpy as np
import os
import json
from tqdm import tqdm

from data_preprocessing import (
    set_korean_font,
    load_and_preprocess_data,
    plot_results,
    extract_outlier_utterances_and_labels,
    save_report_to_json
)
from anomaly_detection.model import (
    MultiLabelClassifier,
    RepeatBinaryHead,
    MultiLabelHead,
    AutoEncoder,
    predict_batch_multilabel,
    predict_batch_repeat,
    predict_batch_emotion,
    train_autoencoder,
    detect_outliers
)

def main():
    """
    전체 데이터 처리 및 모델 학습, 이상치 탐지 과정을 실행하는 메인 함수.

    1) 데이터 및 모델 로드
    2) 기존 데이터에 대한 구체성 라벨 / 같은 말 반복 라벨 / 감성 라벨 tagging
    3) 새로운 데이터에 대한 구체성 라벨 / 같은 말 반복 라벨 / 감성 라벨 tagging

    """
    set_korean_font()
    
    specific_labels = [
        "label_1_사건구체성", "label_1_자서전적기억", "label_1_시간적구체성", "label_1_공간적구체성"
    ]
    repeat_label = ['label_2_같은말반복']
    emotion_labels = ['우울/무기력','불안/초조','감정조절문제']
    
    all_target_cols = specific_labels + repeat_label + emotion_labels

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    #device = torch.device("cpu")
    print(f"현재 사용 가능한 디바이스: {device}")
    
    # 사용한 Tokenizer 모델 로드
    tokenizer_multi = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base-v2022")
    tokenizer_repeat = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
    tokenizer_emotion = AutoTokenizer.from_pretrained("klue/roberta-base")

    data_org, df_nm = load_and_preprocess_data(
        org_path='merged.parquet',
        json_path='018.감성대화 - 응답/감성대화말뭉치(최종데이터)_Training.json',
        chunk_size=25
    )
    
    org_processed_path = 'temp/data_org_processed.parquet'
    if os.path.exists(org_processed_path):
        print("\n저장된 기존 데이터 예측 파일을 로드.")
        data_org = pd.read_parquet(org_processed_path)
    else:
        print("\n기존 데이터에 대한 **감성** 라벨 예측을 시작.")
        predicted_emotion_labels_org = predict_batch_emotion(
            texts=data_org['A_list'].tolist(),
            model_name="klue/roberta-base",
            model_path="pret_emotion.pt", 
            tokenizer=tokenizer_emotion,
            device=device,
            num_labels=len(emotion_labels)
        )
        if predicted_emotion_labels_org is not None:
            data_org[emotion_labels] = predicted_emotion_labels_org
            
            if not os.path.exists('temp'): os.makedirs('temp')
            data_org.to_parquet(org_processed_path)
            print(f"중간 결과가 '{org_processed_path}'에 저장.")
        else:
            print("기존 데이터에 대한 **감성** 라벨 분류기 예측 실패.")
            return


    nm_processed_path = 'temp/df_nm_processed.parquet'
    if os.path.exists(nm_processed_path):
        print("\n저장된 새로운 데이터(user) 예측 파일을 로드.")
        df_nm = pd.read_parquet(nm_processed_path)
    else:
        print("\n새로운 데이터에 대한 **구체성** 라벨 예측을 시작.")
        
        predicted_specific_labels = predict_batch_multilabel(
            texts=df_nm['utterance'].tolist(),
            model_name="beomi/KcELECTRA-base-v2022",
            model_path="pret_multilabel.pt",
            tokenizer=tokenizer_multi,
            device=device,
            num_labels=len(specific_labels)
        )
        if predicted_specific_labels is not None:
            df_nm[specific_labels] = predicted_specific_labels
        else:
            print("새로운 데이터에 대한 **구체성** 라벨 분류기 예측 실패.")
            return

        predicted_repeat_label = predict_batch_repeat(
            texts=df_nm['utterance'].tolist(),
            model_name="monologg/koelectra-base-v3-discriminator",
            model_path="pret_repeat.pt", 
            tokenizer=tokenizer_repeat,
            device=device
        )
        if predicted_repeat_label is not None:
            df_nm[repeat_label] = predicted_repeat_label
        else:
            print("새로운 데이터에 대한 **같은 말 반복** 라벨 분류기 예측 실패.")
            return
        
        predicted_emotion_labels = predict_batch_emotion(
            texts=df_nm['utterance'].tolist(),
            model_name="klue/roberta-base",
            model_path="pret_emotion.pt", 
            tokenizer=tokenizer_emotion,
            device=device,
            num_labels=len(emotion_labels)
        )
        if predicted_emotion_labels is not None:
            df_nm[emotion_labels] = predicted_emotion_labels
            
            # 중간 결과 저장
            if not os.path.exists('temp'): os.makedirs('temp')
            df_nm.to_parquet(nm_processed_path)
        else:
            print("새로운 데이터에 대한 **감성** 라벨 분류기 예측 실패.")
            return


    df_nm['normal'] = 1
    data_org_renamed = data_org.rename(columns={'A_list': 'utterance'})
    data_org_renamed['normal'] = 0
    
    cols_to_merge = all_target_cols + ['utterance', 'normal']
    merged_df = pd.concat([
        df_nm[cols_to_merge].reset_index(drop=True), 
        data_org_renamed[cols_to_merge].reset_index(drop=True)
    ], ignore_index=True)

    print("\n최종 병합된 데이터프레임의 head:")
    print(merged_df.head())
    print("\n최종 데이터프레임의 shape:", merged_df.shape)
    
    plot_results(merged_df, all_target_cols, 'initial_analysis')

    X_tensor = torch.tensor(merged_df[all_target_cols].to_numpy(), dtype=torch.float32)
    X_normal = X_tensor[merged_df['normal'] == 1]
    
    num_features = X_tensor.shape[1]
    model_ae = train_autoencoder(X_normal, num_features, epochs=50)
    torch.save(model_ae.state_dict(), 'autoencoder_model.pth')
    print("모델 가중치 'autoencoder_model.pth'에 저장 완료.")

    recon_error, outliers, threshold = detect_outliers(model_ae, X_tensor, X_normal)
    
    merged_df['recon_error'] = recon_error
    merged_df['is_outlier'] = outliers.astype(int)

    outlier_ratio = np.mean(outliers)
    
    outlier_samples = extract_outlier_utterances_and_labels(merged_df, all_target_cols)
    
    report_data = {
        "metadata": {
            "threshold": float(threshold),
            "outlier_ratio": float(outlier_ratio) * 100,
            "total_samples": len(merged_df),
            "total_outliers": int(np.sum(outliers))
        },
        "outlier_samples": outlier_samples
    }
    save_report_to_json(report_data)

    plot_results(merged_df, all_target_cols, 'final_analysis')


if __name__ == '__main__':
    main()