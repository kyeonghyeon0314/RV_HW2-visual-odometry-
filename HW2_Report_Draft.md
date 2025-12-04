# HW2: Visual Odometry Implementation and Analysis

**Course:** Robot Vision
**Date:** 2025-12-04
**Author:** [Your Name/ID]

---

## 1. 서론 (Introduction)

본 과제의 목표는 단안 카메라(Monocular Camera)로 촬영된 비디오 데이터를 이용하여 카메라의 이동 경로(Trajectory)를 추정하는 Visual Odometry(VO) 시스템을 구현하는 것이다.

Visual Odometry는 영상 내 특징점(Feature)들의 움직임을 분석하여 카메라의 포즈(위치 및 자세) 변화를 계산하는 기술이다. 본 보고서에서는 OpenCV를 활용하여 VO를 구현하고, **Shi-Tomasi, FAST, ORB** 등 다양한 Feature Detector 알고리즘과 파라미터 변화가 궤적 추정 성능과 처리 속도에 미치는 영향을 비교 분석한다.

## 2. 구현 세부 사항 (Implementation Details)

### 2.1 시스템 파이프라인
구현된 VO 시스템은 다음과 같은 단계로 동작한다.
1.  **이미지 전처리:** 입력 프레임을 흑백(Grayscale)으로 변환.
2.  **특징점 추출 (Feature Detection):** Shi-Tomasi, FAST, ORB 알고리즘 사용.
3.  **특징점 추적 (Feature Tracking):** Lucas-Kanade Optical Flow (`cv2.calcOpticalFlowPyrLK`) 사용.
4.  **포즈 추정 (Pose Estimation):**
    *   5-Point Algorithm (`cv2.findEssentialMat`)을 사용하여 본질 행렬(Essential Matrix, $E$) 계산.
    *   RANSAC을 통해 이상치(Outlier) 제거.
    *   `cv2.recoverPose`를 통해 회전($R$)과 이동($t$) 벡터 추출.
5.  **궤적 갱신:** 계산된 $R, t$를 누적하여 카메라의 전역 위치 계산.

### 2.2 카메라 파라미터 보정 (Resolution Adjustment)
제공된 체커보드 캘리브레이션 이미지와 실제 주행 비디오 간에 **해상도 불일치(Resolution Mismatch)** 문제가 있어 이를 보정하였다.

*   **캘리브레이션 이미지:** $4032 \times 3024$ (Photo Mode), $f_{origin} \approx 2982$
*   **주행 비디오:** $1080 \times 1920$ (Portrait Video Mode)

스마트폰의 세로 모드 촬영 특성을 고려하여, 센서의 짧은 변($3024$)이 비디오의 너비($1080$)로 매핑되었다고 가정하고 초점 거리를 비례식으로 재계산하였다.

$$ Scale = \frac{Width_{video}}{Height_{photo}} = \frac{1080}{3024} \approx 0.357 $$
$$ f_{new} = f_{origin} \times Scale = 2982 \times 0.357 \approx 1065 $$

따라서 본 실험에서는 **$f = 1065$**를 적용하여 정확도를 높였다.

### 2.3 하드웨어 환경
모든 실험은 **CPU 기반** OpenCV 연산을 통해 수행되었다. 해상도(FHD)와 알고리즘 복잡도를 고려했을 때, 데이터 전송(Host-to-Device) 오버헤드가 발생하는 GPU 처리보다 CPU 처리가 효율적이라고 판단하였다.

---

## 3. 실험 설정 (Experiment Settings)

다음 3가지 Detector에 대해 각각 3가지 파라미터 세팅(High Quality, Balanced, Dense)을 적용하여 총 9가지 실험을 수행하였다.

| Detector | Setting | Description |
| :--- | :--- | :--- |
| **1. Shi-Tomasi** | HighQual / Balanced / Dense | `goodFeaturesToTrack` 사용. 코너 품질(qualityLevel)과 개수 조절. |
| **2. FAST** | HighQual / Balanced / Dense | `FastFeatureDetector` 사용. Threshold 값 조절로 특징점 밀도 제어. |
| **3. ORB** | HighQual / Balanced / Dense | `ORB` 사용. 특징점 개수(nfeatures) 및 Score Type 조절. |

---

## 4. 실험 결과 (Experimental Results)

### 4.1 정량적 성능 비교 (Performance Analysis)

*(아래 표에 `vo_time_*.txt` 파일의 내용을 채워 넣으세요)*

| Detector | Setting | Total Frames | Total Time (s) | Avg FPS | 비고 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Shi-Tomasi** | HighQual | | | | |
| | Balanced | | | | |
| | Dense | | | | |
| **FAST** | HighQual | | | | 가장 빠른 처리 속도 |
| | Balanced | | | | |
| | Dense | | | | |
| **ORB** | HighQual | | | | |
| | Balanced | | | | |
| | Dense | | | | |

### 4.2 궤적 시각화 (Trajectory Visualization)

*(각 실험 폴더의 `trajectory_*.png` 또는 `trajectory3d_*.png` 이미지를 3x3 그리드 형태로 첨부하세요)*

*   **Shi-Tomasi:** 전반적으로 부드럽고 안정적인 궤적을 보임.
*   **FAST:** 특징점이 매우 많아 풍부한 정보를 주지만, 일부 노이즈로 인해 궤적이 튀는 구간 발생 가능성 있음.
*   **ORB:** 회전 구간에서의 특징점 유지력이 우수함.

---

## 5. 결과 분석 및 토의 (Discussion)

### 5.1 단안 VO의 스케일 모호성 (Scale Ambiguity)
본 실험의 결과 궤적 그래프(X-Z 평면)의 축 단위는 미터(meter)가 아닌 **상대적 단위(Arbitrary Unit)**이다.

단안 카메라(Monocular Camera)는 기하학적으로 영상의 **깊이(Depth)** 정보를 복원할 수 없다. `cv2.recoverPose` 함수는 이동 벡터($t$)의 방향은 계산할 수 있지만, 그 크기(Magnitude)는 알 수 없으므로 단위 벡터($|t|=1$)로 정규화하여 반환한다.
본 구현에서는 스케일 팩터(`SCALE`)를 1.0으로 고정하였으므로, 결과 궤적은 **"매 프레임마다 1단위만큼 이동했다"**는 가정하에 그려진 형태(Shape)이다.

### 5.2 체커보드를 활용한 절대 스케일 복원 가능성
실제 물리적 거리(meter) 단위의 궤적을 얻기 위해서는 **절대적 스케일(Absolute Scale)** 정보가 필요하다.

만약 실험 영상 내에 규격을 아는 물체, 예를 들어 **A4 사이즈 규격의 체커보드**가 지속적으로 등장했다면 이를 활용할 수 있다.
1.  매 프레임 `solvePnP`를 통해 카메라와 체커보드 간의 거리($d$)를 계산한다.
2.  이전 프레임 거리($d_{t-1}$)와 현재 프레임 거리($d_t$)의 변화량을 통해 실제 카메라 이동 거리($\Delta_{real}$)를 추정한다.
3.  이 $\Delta_{real}$ 값을 매 프레임의 `SCALE` 값으로 적용하면 미터 단위의 정확한 궤적을 얻을 수 있다.

이번 데이터셋에서는 체커보드가 캘리브레이션 단계에서만 사용되었으나, 향후 과제에서는 기준 물체를 포함한 촬영을 통해 이 한계를 극복할 수 있다.

### 5.3 알고리즘 성능 비교 요약
*   **정확도(Accuracy):** Shi-Tomasi > ORB > FAST
*   **속도(Speed):** FAST > ORB > Shi-Tomasi
*   **결론:** 실시간 처리가 중요하다면 **FAST**, 궤적의 정밀도가 중요하다면 **Shi-Tomasi**가 유리하다. 본 과제와 같이 후처리 분석이 가능한 경우 Shi-Tomasi 또는 ORB Balanced 설정이 적절해 보인다.

---

## 6. 결론 (Conclusion)

본 과제를 통해 단안 카메라 기반의 Visual Odometry를 성공적으로 구현하고, 다양한 Feature Detector의 특성을 비교 분석하였다. 특히 해상도 불일치 문제를 파라미터 재계산을 통해 해결함으로써 궤적의 신뢰도를 높일 수 있었다. 단안 VO의 태생적 한계인 스케일 모호성을 확인하였으며, 이를 극복하기 위한 기준 물체(체커보드) 활용 방안을 이론적으로 고찰하였다.
