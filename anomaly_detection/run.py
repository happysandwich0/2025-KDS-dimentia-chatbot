import torch
import warnings
from transformers import AutoTokenizer

warnings.filterwarnings('ignore', category=UserWarning, message='You are using `torch.load` with `weights_only=False`')

from anomaly_detection.interact import (
    load_models_with_finetuning_check,
    classify_and_detect_outlier,
    save_to_json_and_finetune
)

if __name__ == '__main__':
    model_name = "beomi/KcELECTRA-base"
    label_model_path = "best.pt"
    autoencoder_model_path = "autoencoder_model.pth"
    user_data_path = "user_data.json"
    target_cols = [
        "label_1_사건구체성", "label_1_자서전적기억", "label_1_시간적구체성", "label_1_공간적구체성", 
        'label_2_주제이탈', 'label_2_같은말반복', 'label_2_과도한흥분'
    ]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("--- 텍스트 이상치 탐지 시스템 시작 ---")
    
    label_model, autoencoder_model, threshold = load_models_with_finetuning_check(
        label_model_path, autoencoder_model_path, model_name, target_cols, device
    )

    # 예시
    if label_model and autoencoder_model:
        test_texts = [
            "어제 퇴근하고 집에 가다가 갑자기 횡단보도에서 차에 치일 뻔 했어. 너무 놀라서 지금도 심장이 벌렁거려. 다행히 다친 곳은 없어서 안심이야. 이제 다시는 그 길로 안 다닐 거야.",
            "나 요즘 너무 행복해! 내가 한 달 뒤면 사랑하는 사람과 결혼하거든! 그래서 앞으로 한 달 동안은 사람도 많이 만나게 될 것 같아!",
            "직장에서 누명을 쓰고 잘렸어. 화가 나서 밤에 잠이 안 와. 회사에 달려가서 사장 멱살이라도 잡고 싶어. 이렇게 경우 없이 그러는 게 어딨어.",
            "요즘 취업이 잘 안 돼서 자신감이 없어졌는지 사람 대할 때 뭔가 계속 불안하고 초조해. 뭐라고 말로 설명하기 힘든데 그냥 우울증인 거 같기도 하고.",
            "오늘은 출근 시간이 한 시간 늦어져서 느긋해. 만족스러워. 우리 회사는 한 달에 한 번 오후 출근을 할 수 있어. 그 외에도 직원 복지를 신경쓰는 부분이 많은 것 같아서 좋아. 맞아. 내가 만족한 만큼 열심히해서 성과를 내려고 해.",
            "내 소득이 너무 낮아서 고민이야 미래도 없어. 남자친구한테 매일 얻어먹는 것도 미안해. 날 받아주는 직장도 없어. 내 미래가 암흑이 되면 어떡해?",
            "나는 성공하고 싶어갖고 이 자기 최면 거는 거에 프로그램을 해서 자 저한테 처음에는 이게 학회가 두 가지가 있어요. 레드썬 하는 가리키는 그런 학회가 있고 다른 학회에 있는 그 원장님 상담학 박사님이라서 ****** 박사님이 있거든요.",
            "내가 아는 여자가 있는데 그 여자가 잘난 남자를 만난다네. 그 여자 수준이랑 맞지 않는데 말이지. 친구랑 전화해서 공감을 얻었지. 아내랑 싸웠어. 기분이 안 좋네.",
            "성공하면은요, 평범해 평범해서 성공이라는 단어가 저가 떠오르게 저하고는 연결이 안 되는 거죠. 예 저한테 가만 놔 갖고 고시 고시라고 할 거 같으면 우리 아까 말씀드렸듯이 뭐 학교 다닐 때 공부도 안 하고 그랬는데 지금 공무원이래서 고맙다 싶어요.",
            "결혼을 하려고 하는데 남자 친구네 부모님이 나를 마음에 안 들어 하시는 것 같아. 그래서 어떻게 해야 할지 고민이야.",
            "나 지금 너무너무 행복해! 내가 한 달 뒤면 사랑하는 사람과 결혼하거든! 그래서 앞으로 한 달 동안은 사람도 많이 만나게 될 것 같아!",
        ]

        for text in test_texts:
            analysis_result = classify_and_detect_outlier(
                text, label_model, autoencoder_model, tokenizer, target_cols, threshold, device
            )
            
            # 분석 결과 출력
            print("\n" + "="*50)
            print("단일 텍스트 분석 결과")
            print("="*50)
            print(f"입력 텍스트: {analysis_result['text']}")
            print(f"예측 라벨: {analysis_result['predicted_labels']}")
            print(f"재구성 오차: {analysis_result['reconstruction_error']:.6f}")
            print(f"이상치 임계값: {analysis_result['threshold']:.6f}")
            print(f"이상치 여부: {'이상치로 판단됨' if analysis_result['is_outlier'] else '정상 데이터로 판단됨'}")
            
            save_to_json_and_finetune(analysis_result, user_data_path)
    else:
        print("\n모델 로드에 실패했습니다. 파일을 확인하고 다시 시도해 주세요.")
