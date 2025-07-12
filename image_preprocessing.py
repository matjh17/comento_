from datasets import load_dataset
from PIL import Image, ImageEnhance
import numpy as np
import os
import cv2
import random

# 출력 폴더 생성
os.makedirs("preprocessed_samples", exist_ok=True)

# 데이터셋 로드
dataset = load_dataset("ethz/food101", split="train[:5]")  # 샘플 5개만 사용


def preprocess_image(pil_image):
    # 1. 크기 조정 (224x224)
    img = pil_image.resize((224, 224))

    # 2. Grayscale 변환
    img = img.convert("L")  # 흑백으로 변환

    # 3. Normalize (0~1)
    img_np = np.array(img).astype(np.float32) / 255.0

    # 4. 노이즈 제거 (Blur 필터 적용)
    img_np = cv2.GaussianBlur(img_np, (5, 5), 0)

    # 5. 데이터 증강
    # 랜덤 좌우 반전
    if random.random() > 0.5:
        img_np = cv2.flip(img_np, 1)

    # 랜덤 회전 (-15도 ~ +15도)
    angle = random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((112, 112), angle, 1.0)
    img_np = cv2.warpAffine(img_np, M, (224, 224), borderMode=cv2.BORDER_REFLECT)

    # 랜덤 밝기 변화
    factor = random.uniform(0.8, 1.2)
    img_np = np.clip(img_np * factor, 0, 1)

    return img_np


def is_outlier(img_np):
    # 너무 어두운 이미지 필터링 (평균 밝기 < 0.1)
    if img_np.mean() < 0.1:
        return True

    # 객체 크기 기반 필터링 (가장 밝은 픽셀 기준의 면적이 작을 경우)
    thresh = (img_np > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return True
    largest_area = max([cv2.contourArea(c) for c in contours])
    if largest_area < 500:  # 객체가 너무 작으면 제거
        return True

    return False

# 전처리 및 저장
for i, sample in enumerate(dataset):
    pil_image = sample['image']
    processed = preprocess_image(pil_image)

    if is_outlier(processed):
        print(f"Image {i} skipped (outlier)")
        continue

    save_path = f"preprocessed_samples/img_{i}.png"
    cv2.imwrite(save_path, (processed * 255).astype(np.uint8))
    print(f"Saved: {save_path}")
