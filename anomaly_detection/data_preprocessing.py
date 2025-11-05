import pandas as pd
import json
import re
import os
from collections import defaultdict
import platform
import matplotlib.pyplot as plt
import seaborn as sns

def set_korean_font():
    system = platform.system()
    if system == "Windows":
        plt.rcParams['font.family'] = 'Malgun Gothic'
    elif system == "Darwin":  # MacOS
        plt.rcParams['font.family'] = 'AppleGothic'
    else:  # Linux
        plt.rcParams['font.family'] = 'NanumGothic'
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_style("whitegrid", rc={"font.family": plt.rcParams['font.family']})

def load_and_preprocess_data(org_path, json_path, chunk_size=5):
    """
    기존 데이터(train set from AI-hub)와 새로운 데이터(user data)를 로드하고 전처리
    
    1) 기존 데이터 감성 라벨 부재. 0으로 초기화.
    2) 기존 데이터 중 A04(청년)로 마킹된 페르소나 로드. 이후 이를 정상군으로 명명.
       ** 청년의 대화는 발화 구체성이 담보되어 있음을 가정 **
    3) 새로운 데이터는 발화가 5개 이상일 경우, chunk_size개씩 묶어서 여러 개의 행으로 생성
       마지막 남은 발화 묶음의 발화 수가 3개 이하인 경우 버림
       ** 발화 수 3개 이하의 짧은 대화는 라벨링과 이상치 판별이 어려울 것을 고려함 **

    """
    org = pd.read_parquet(org_path)
    org = org[(org['qualityPoint']==2)]

    def extract_all_answers(text):
        if pd.isna(text):
            return ""
        text = text.replace('\xa0', ' ')
        answers = re.findall(r"A:\s*(.*)", text)
        return "".join([a.strip() for a in answers])

    org["A_list"] = org["qa_combined"].apply(extract_all_answers)
    org_target_cols = [
        "label_1_사건구체성", "label_1_자서전적기억", "label_1_시간적구체성", "label_1_공간적구체성", 
        'label_2_같은말반복'
    ]

    # 기존 데이터에는 감성 라벨이 없으므로, 0으로 초기화

    new_emotion_labels = ['우울/무기력','불안/초조','감정조절문제']
    for col in new_emotion_labels:
        if col not in org.columns:
            org[col] = 0

    target_cols_full = org_target_cols + ['A_list']
    data = org.loc[:, target_cols_full + ['jsonId', 'qa_combined']]
    
    with open(json_path, "r", encoding="utf-8") as f:
        df_json = json.load(f)
    
    # 기존 데이터 중 A04(청년)로 마킹된 페르소나만 불러옴

    persona_hs = defaultdict(list)
    for entry in df_json:
        human_list = entry["profile"]["persona"].get("human", [])
        if "A04" in human_list:
            continue
        persona_id = entry["profile"]["persona-id"]
        content = entry["talk"]["content"]
        for k, v in content.items():
            if k.startswith("HS") and v.strip():
                processed_utterance = v.strip().replace('\xa0', ' ')
                persona_hs[persona_id].append(processed_utterance)
    
    result_list = []
    for pid, utterances in persona_hs.items():
        if len(utterances) >= 4:
            num_chunks = len(utterances) // chunk_size
            for i in range(num_chunks):
                chunk = utterances[i * chunk_size:(i + 1) * chunk_size]
                result_list.append({
                    "persona_id": pid,
                    "utterance": " ".join(chunk)
                })
            
            remaining_utterances = utterances[num_chunks * chunk_size:]
            if len(remaining_utterances) >= 4:
                 result_list.append({
                    "persona_id": pid,
                    "utterance": " ".join(remaining_utterances)
                })

    df_new = pd.DataFrame(result_list)
    print(f"새로운 데이터프레임의 행 수: {len(df_new)}")
    return data, df_new

def plot_results(merged_df, target_cols, file_prefix):
    """
    이상치 탐지 결과를 시각화하고 파일로 저장
    """
    if not os.path.exists('output'):
        os.makedirs('output')

    plot_data_label = pd.DataFrame()
    for col in target_cols:
        normal_ratio = merged_df[merged_df['normal'] == 1][col].mean() * 100
        abnormal_ratio = merged_df[merged_df['normal'] == 0][col].mean() * 100
        temp_df = pd.DataFrame({
            'Label': [col, col],
            'Category': ['Normal (1)', 'Abnormal (0)'],
            'Ratio': [normal_ratio, abnormal_ratio]
        })
        plot_data_label = pd.concat([plot_data_label, temp_df], ignore_index=True)

    if not plot_data_label.empty:
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Label', y='Ratio', hue='Category', data=plot_data_label, palette='viridis')
        plt.title('정상/비정상 데이터 그룹별 라벨 분포 (비율)', fontsize=16)
        plt.xlabel('라벨', fontsize=12)
        plt.ylabel('긍정 라벨(1) 비율 (%)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='카테고리')
        plt.tight_layout()
        plt.savefig(f'output/{file_prefix}_label_distribution.png')
        plt.close()
    else:
        print("라벨 분포 데이터를 생성 불가.")

    if 'is_outlier' in merged_df.columns:
        plot_data_outlier = merged_df.groupby('normal')['is_outlier'].mean().reset_index()
        plot_data_outlier['is_outlier'] *= 100
        plot_data_outlier['normal_label'] = plot_data_outlier['normal'].map({1: '정상 (1)', 0: '비정상 (0)'})
        
        plt.figure(figsize=(8, 6))
        sns.barplot(x='normal_label', y='is_outlier', data=plot_data_outlier, palette='viridis')
        plt.title('정상/비정상 태그 그룹별 이상치 비율', fontsize=16)
        plt.xlabel('정상 태그', fontsize=12)
        plt.ylabel('이상치 비율 (%)', fontsize=12)
        plt.xticks()
        plt.tight_layout()
        plt.savefig(f'output/{file_prefix}_outlier_distribution.png')
        plt.close()
    else:
        print("'is_outlier' 컬럼 오류.")


def extract_outlier_utterances_and_labels(df, target_cols, num_samples=5):
    """
    정상/비정상 그룹별로 이상치로 분류된 발화문과 태깅된 라벨을 반환.
    """
    def format_outlier_data(outliers_df):
        outlier_list = []
        for _, row in outliers_df.iterrows():
            labeled_cols = [col for col in target_cols if row[col] == 1]
            outlier_list.append({
                "utterance": row['utterance'],
                "predicted_labels": labeled_cols if labeled_cols else "(태깅된 라벨 없음)"
            })
        return outlier_list

    normal_outliers_df = df[(df['normal'] == 1) & (df['is_outlier'] == 1)].head(num_samples)
    abnormal_outliers_df = df[(df['normal'] == 0) & (df['is_outlier'] == 1)].head(num_samples)

    return {
        "normal_group_outliers": format_outlier_data(normal_outliers_df),
        "abnormal_group_outliers": format_outlier_data(abnormal_outliers_df)
    }

def save_report_to_json(report_data, file_path='output/analysis_report.json'):
    """
    분석 보고서를 JSON 파일로 저장
    """
    if not os.path.exists('output'):
        os.makedirs('output')

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, ensure_ascii=False, indent=4)
    
    print("--------------------------------------------------")
    print("임계값:", report_data['metadata']['threshold'])
    print("전체 데이터 이상치 비율:", f"{report_data['metadata']['outlier_ratio']:.2f}%")
    print("--------------------------------------------------")

