import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# plt.switch_backend('Agg') # Commented out to enable GUI

def main():
    # ==========================================
    # 설정 (Configuration)
    # ==========================================
    # TODO: 비디오 파일 경로나 이미지 시퀀스 형식을 업데이트하세요
    DATASET_PATH = "./data/video.MOV" 
    
    # 실험 설정 (Experiment Settings)
    # 원하는 실험의 이름을 CURRENT_EXP에 할당하세요.
    CURRENT_EXP = "Exp_9_ORB_Dense"  
    
    EXPERIMENTS = {
        # ==========================================
        # 1. Shi-Tomasi (GoodFeaturesToTrack)
        # ==========================================
        "Exp_1_Shi_HighQual": {"detector": "SHI_TOMASI", "params": {"maxCorners": 1000, "qualityLevel": 0.05, "minDistance": 20}},
        "Exp_2_Shi_Balanced": {"detector": "SHI_TOMASI", "params": {"maxCorners": 2500, "qualityLevel": 0.005, "minDistance": 8}},
        "Exp_3_Shi_Dense":    {"detector": "SHI_TOMASI", "params": {"maxCorners": 4000, "qualityLevel": 0.001, "minDistance": 5}},
        
        # ==========================================
        # 2. FAST (Features from Accelerated Segment Test)
        # threshold: 임계값 (높으면 점이 적어짐, 낮으면 많아짐)
        # ==========================================
        "Exp_4_FAST_HighQual": {"detector": "FAST", "params": {"threshold": 50, "nonmaxSuppression": True}}, 
        "Exp_5_FAST_Balanced": {"detector": "FAST", "params": {"threshold": 20, "nonmaxSuppression": True}}, 
        "Exp_6_FAST_Dense":    {"detector": "FAST", "params": {"threshold": 10, "nonmaxSuppression": True}}, 
        
        # ==========================================
        # 3. ORB (Oriented FAST and Rotated BRIEF)
        # nfeatures: 최대 특징점 개수
        # ==========================================
        "Exp_7_ORB_HighQual":  {"detector": "ORB", "params": {"nfeatures": 1000, "scoreType": cv2.ORB_HARRIS_SCORE}},
        "Exp_8_ORB_Balanced":  {"detector": "ORB", "params": {"nfeatures": 3000, "scoreType": cv2.ORB_FAST_SCORE}},
        "Exp_9_ORB_Dense":     {"detector": "ORB", "params": {"nfeatures": 5000, "scoreType": cv2.ORB_FAST_SCORE}},
    }
    
    # 현재 실험 설정 로드
    if CURRENT_EXP not in EXPERIMENTS:
        raise ValueError(f"Unknown Experiment: {CURRENT_EXP}")
    
    EXP_CONFIG = EXPERIMENTS[CURRENT_EXP]
    DETECTOR = EXP_CONFIG["detector"]
    FEATURE_PARAMS = EXP_CONFIG["params"]
    
    EXP_NAME = CURRENT_EXP # 파일 저장 이름용

    # 카메라 파라미터 (1080x1920 세로 모드 비디오에 맞춰 조정됨)
    # 원본 해상도: 4032 x 3024 (사진 모드 기준)
    # 원본 초점거리(f_origin): 2982 pixel
    # 
    # 변경된 해상도(비디오): 1080 x 1920 (세로 모드)
    # 계산 원리: 이미지 리사이즈 비율에 맞춰 초점 거리도 비례하여 조절됨
    # 비율(Scale) = 현재 너비(1080) / 원본 높이(3024)  <- 세로 모드 촬영으로 센서의 짧은 변(3024)이 영상의 너비(1080)가 됨
    # f_new = f_origin * Scale = 2982 * (1080 / 3024) ≈ 1065
    
    # 주점 (Principal Point): 이미지의 중심
    c = (540.0, 960.0)  # cx, cy (1080/2, 1920/2)
    f = 1065.0
    
    # 매칭 파라미터 (cv2.findEssentialMat)
    MATCHING_PARAMS = dict(prob=0.99, threshold=1.0)
    
    USE_5PT = True
    MIN_INLIER_NUM = 100
    
    # 최적화 파라미터
    FRAME_SKIP = 2 # N번째 프레임마다 처리 (예: 2는 1프레임 건너뜀)하여 베이스라인 증가
    SCALE = 1.0 # 이동 벡터 스케일 팩터 (실제 거리 근사)
    INVERT_X = True # "좌우 반전" 문제 해결용. 궤적이 좌우로 뒤집힌 경우 True로 설정.
    
    OUTPUT_FILE = f"vo_trajectory_{EXP_NAME}.xyz"
    OUTPUT_VIDEO = f"vo_result_{EXP_NAME}.avi"
    OUTPUT_PLOT = f"trajectory_{EXP_NAME}.png"
    OUTPUT_PLOT_3D = f"trajectory_3d_{EXP_NAME}.png"
    OUTPUT_TIME_FILE = f"vo_time_{EXP_NAME}.txt"
    
    HEADLESS = False # GUI를 끄려면 True로 설정 (서버 환경 등)
    # ==========================================

    # 카메라 궤적을 저장할 파일 열기
    camera_trajectory = open(OUTPUT_FILE, 'wt')
    if not camera_trajectory:
        raise Exception("파일을 생성할 수 없습니다")

    # 비디오 또는 이미지 시퀀스 열기
    cap = cv2.VideoCapture(DATASET_PATH)
    if not cap.isOpened():
        print(f"오류: 데이터셋을 읽을 수 없습니다: {DATASET_PATH}")
        return

    # 첫 번째 프레임 읽기
    ret, prev_frame = cap.read()
    if not ret:
        print("오류: 첫 번째 프레임을 읽을 수 없습니다")
        return

    # 비디오 라이터 설정
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width, frame_height))

    # 흑백 변환
    if len(prev_frame.shape) == 3:
        gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    else:
        gray_prev = prev_frame

    camera_pose = np.eye(4, dtype=np.float64)
    
    # 시각화용 변수
    traj_x = [0.0]
    traj_z = [0.0]
    
    fig, ax = plt.subplots()
    ax.set_title("Visual Odometry Trajectory")
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    
    frame_id = 0
    print("Visual Odometry 시작...")
    
    if not HEADLESS:
        cv2.namedWindow('Visual Odometry', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Visual Odometry', 600, 800) # 적절한 크기로 초기화
    
    print(f"현재 실험: {EXP_NAME}")
    print(f"사용 중인 Detector: {DETECTOR}")
    
    start_time = time.time() # 시작 시간 기록
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 프레임 건너뛰기 (Frame Skipping)
        if frame_id % FRAME_SKIP != 0:
            frame_id += 1
            continue
        
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # 특징점 추적 (Feature Tracking)
        if DETECTOR == "SHI_TOMASI":
            point_prev = cv2.goodFeaturesToTrack(gray_prev, **FEATURE_PARAMS)
        elif DETECTOR == "FAST":
            fast = cv2.FastFeatureDetector_create(**FEATURE_PARAMS)
            keypoints = fast.detect(gray_prev, None)
            # KeyPoint 객체를 numpy 배열 (N, 1, 2)로 변환
            if keypoints:
                point_prev = np.float32([kp.pt for kp in keypoints]).reshape(-1, 1, 2)
            else:
                point_prev = None
        elif DETECTOR == "ORB":
            orb = cv2.ORB_create(**FEATURE_PARAMS)
            keypoints = orb.detect(gray_prev, None)
            # KeyPoint 객체를 numpy 배열 (N, 1, 2)로 변환
            if keypoints:
                point_prev = np.float32([kp.pt for kp in keypoints]).reshape(-1, 1, 2)
            else:
                point_prev = None
        else:
            print(f"Unknown Detector: {DETECTOR}")
            break
        
        if frame_id == 0 and point_prev is not None:
             print(f"첫 프레임 감지된 특징점 수: {len(point_prev)}")
        
        if point_prev is None or len(point_prev) < 8:
            gray_prev = gray
            out.write(frame) # 추적 실패 시 원본 프레임 저장
            continue

        point, status, err = cv2.calcOpticalFlowPyrLK(gray_prev, gray, point_prev, None)
        
        # 좋은 특징점 선택
        good_point_prev = point_prev[status == 1]
        good_point = point[status == 1]

        if len(good_point) < 8:
             gray_prev = gray
             out.write(frame)
             continue

        # 포즈 추정 (Pose Estimation)
        if USE_5PT:
            E, inlier_mask = cv2.findEssentialMat(good_point_prev, good_point, f, c, cv2.FM_RANSAC, MATCHING_PARAMS['prob'], MATCHING_PARAMS['threshold'])
        else:
            F, inlier_mask = cv2.findFundamentalMat(good_point_prev, good_point, cv2.FM_RANSAC, MATCHING_PARAMS['threshold'], MATCHING_PARAMS['prob'])
            K = np.array([[f, 0, c[0]], [0, f, c[1]], [0, 0, 1]])
            E = K.T @ F @ K

        if E is not None and E.shape == (3, 3):
             inlier_num, R, t, mask = cv2.recoverPose(E, good_point, good_point_prev, focal=f, pp=c)
             
             if inlier_num > MIN_INLIER_NUM:
                T = np.eye(4)
                T[0:3, 0:3] = R
                T[0:3, [3]] = t * SCALE
                
                camera_pose = camera_pose @ np.linalg.inv(T)
                
                x, y, z = camera_pose[0][3], camera_pose[1][3], camera_pose[2][3]
                camera_trajectory.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
                
                traj_x.append(x)
                traj_z.append(z)

        # 특징점 시각화
        vis_img = frame.copy()
        for i, (new, old) in enumerate(zip(good_point, good_point_prev)):
            a, b = new.ravel()
            c_x, d_y = old.ravel()
            if inlier_mask is not None and inlier_mask[i]:
                 vis_img = cv2.line(vis_img, (int(a), int(b)), (int(c_x), int(d_y)), (0, 255, 0), 2)
                 vis_img = cv2.circle(vis_img, (int(a), int(b)), 3, (0, 255, 0), -1)
            else:
                 vis_img = cv2.circle(vis_img, (int(a), int(b)), 3, (0, 0, 255), -1)

        info = f"Frame: {frame_id} Inliers: {inlier_num if 'inlier_num' in locals() else 0}"
        cv2.putText(vis_img, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        out.write(vis_img)
        
        if not HEADLESS:
            cv2.imshow('Visual Odometry', vis_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            if frame_id % 100 == 0:
                print(f"프레임 {frame_id} 처리 중...")

        gray_prev = gray
        frame_id += 1

    end_time = time.time() # 종료 시간 기록
    total_time = end_time - start_time
    fps = frame_id / total_time if total_time > 0 else 0
    
    time_info = (
        f"실험: {EXP_NAME}\n"
        f"총 소요 시간: {total_time:.2f} 초\n"
        f"총 프레임 수: {frame_id}\n"
        f"평균 FPS: {fps:.2f}\n"
    )
    
    print("\n" + "="*30)
    print(time_info)
    print("="*30 + "\n")
    
    with open(OUTPUT_TIME_FILE, "w") as f:
        f.write(time_info)

    cap.release()
    out.release()
    camera_trajectory.close()
    if not HEADLESS:
        cv2.destroyAllWindows()
    
    # 2D 궤적 저장 (X-Z)
    ax.plot(traj_x, traj_z, 'b-')
    ax.axis('equal')
    plt.savefig(OUTPUT_PLOT)
    
    # 3D 궤적 저장
    try:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 파일에서 궤적 다시 읽기
        data = np.loadtxt(OUTPUT_FILE)
        if data.ndim > 1:
            # 카메라 좌표계:
            # X: 오른쪽
            # Y: 아래 (높이)
            # Z: 전방
            
            cam_x = data[:, 0]
            cam_y = data[:, 1]
            cam_z = data[:, 2]

            # 초기 피치 각도 (도 단위)
            INITIAL_PITCH = 45.0 
            AUTO_LEVEL = False # 궤적 기반 자동 피치 보정
            
            theta = 0.0
            
            if AUTO_LEVEL:
                # 궤적 추세에서 피치 각도 계산
                # 평면 위를 이동한다고 가정하므로 순수 높이 변화는 0이어야 함.
                # 카메라 프레임: Y는 아래, Z는 전방.
                # 회전 후 Y_new_end가 약 0이 되도록 하는 각도 theta를 찾음.
                
                # 초기 불안정성을 피하기 위해 마지막 20% 지점을 사용하여 추세 추정
                num_points = len(cam_y)
                if num_points > 10:
                    dy = cam_y[-1] - cam_y[0]
                    dz = cam_z[-1] - cam_z[0]
                    
                    # Y-Z 평면에서 궤적 벡터의 각도
                    # 이 벡터를 Z축(Y=0)에 맞추기 위해 회전
                    # 현재 Z축에 대한 각도: atan2(dy, dz)
                    # -atan2(dy, dz) 만큼 회전 필요
                    
                    calculated_pitch = np.degrees(np.arctan2(dy, dz))
                    print(f"자동 레벨링: 감지된 궤적 피치 {calculated_pitch:.2f} 도.")
                    print(f"경로를 평탄화하기 위해 보정 적용.")
                    
                    theta = -np.arctan2(dy, dz)
            
            elif INITIAL_PITCH != 0:
                theta = np.radians(-INITIAL_PITCH)
            
            # 회전 적용
            if theta != 0:
                c_th, s_th = np.cos(theta), np.sin(theta)
                Rx = np.array([
                    [1, 0, 0],
                    [0, c_th, -s_th],
                    [0, s_th, c_th]
                ])
                
                # 모든 점에 회전 적용
                points = np.vstack((cam_x, cam_y, cam_z)).T
                points_rotated = points @ Rx.T
                
                cam_x = points_rotated[:, 0]
                cam_y = points_rotated[:, 1]
                cam_z = points_rotated[:, 2]

            # 매핑 및 보정:
            # 1. 0,0,0에서 시작하도록 이동
            cam_x -= cam_x[0]
            cam_y -= cam_y[0]
            cam_z -= cam_z[0]
            
            # 2. 플롯 축으로 매핑
            # 플롯 X = 카메라 X (오른쪽)
            # 플롯 Y = -카메라 Z (전방) -> 전방 이동이 양수가 되도록 Z 반전
            # 플롯 Z = -카메라 Y (높이) -> 위쪽이 양수가 되도록 Y 반전
            
            if INVERT_X:
                plot_x = -cam_x
            else:
                plot_x = cam_x
                
            plot_y = -cam_z
            plot_z = -cam_y 
            
            ax.plot(plot_x, plot_y, plot_z, label='Trajectory', linewidth=2)
            
            ax.set_xlabel('X (Right)')
            ax.set_ylabel('Z (Forward)')
            ax.set_zlabel('Y (Height, -Down)')
            ax.set_title(f'3D Visual Odometry Trajectory\n({EXP_NAME})')
            
            # 초기 시점 설정
            ax.view_init(elev=30, azim=-135)
            
            # 비율 유지 (Equal Aspect Ratio)
            max_range = np.array([plot_x.max()-plot_x.min(), plot_y.max()-plot_y.min(), plot_z.max()-plot_z.min()]).max() / 2.0
            
            mid_x = (plot_x.max()+plot_x.min()) * 0.5
            mid_y = (plot_y.max()+plot_y.min()) * 0.5
            mid_z = (plot_z.max()+plot_z.min()) * 0.5
            
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)

            print("3D 플롯 창을 엽니다. 마우스로 회전하여 볼 수 있습니다.")
            print("원하는 뷰를 맞춘 후, 창 하단의 'Save' 버튼을 눌러 직접 저장하세요.")
            plt.show() # 인터랙티브 플롯 표시

            # plt.savefig(OUTPUT_PLOT_3D) # 사용자가 직접 저장하도록 자동 저장 비활성화
            print(f"완료. 결과 영상: {OUTPUT_VIDEO}, 2D 플롯: {OUTPUT_PLOT} (3D 플롯은 수동 저장 필요)")
        else:
            print("3D 플롯을 그리기 위한 포인트가 부족합니다.")
    except Exception as e:
        print(f"3D 플롯 생성 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
