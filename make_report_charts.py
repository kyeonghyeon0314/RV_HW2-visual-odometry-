import os
import glob
import matplotlib.pyplot as plt
import numpy as np

# 한글 폰트 설정 (필요시 시스템 폰트에 맞게 조정, 여기서는 영문으로 진행)
plt.rcParams['axes.unicode_minus'] = False

RESULT_DIR = "HW2/Result"
OUTPUT_DIR = os.path.join(RESULT_DIR, "Report_Assets")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 데이터 구조: { 'Algorithm': { 'Setting': { 'fps': float, 'traj': [[x,y,z], ...] } } }
data = {}

# 1. 데이터 파싱
print("데이터 로딩 중...")
for category_dir in sorted(os.listdir(RESULT_DIR)): # 1SHI, 2FAST, 3ORB
    cat_path = os.path.join(RESULT_DIR, category_dir)
    if not os.path.isdir(cat_path) or category_dir == "Report_Assets":
        continue
    
    algo_name = category_dir[1:] # Remove number prefix (1SHI -> SHI)
    data[algo_name] = {}
    
    for exp_dir in sorted(os.listdir(cat_path)): # EXP1, EXP2...
        exp_path = os.path.join(cat_path, exp_dir)
        if not os.path.isdir(exp_path):
            continue
            
        # 파일 찾기
        time_files = glob.glob(os.path.join(exp_path, "vo_time_*.txt"))
        traj_files = glob.glob(os.path.join(exp_path, "vo_trajectory_*.xyz"))
        
        if not time_files or not traj_files:
            continue
            
        # 설정 이름 추출 (파일명에서) e.g., ..._Shi_HighQual.txt
        setting_name = "Unknown"
        fname = os.path.basename(time_files[0])
        parts = fname.replace(".txt", "").split("_")
        if len(parts) >= 5:
            setting_name = parts[-1] # HighQual, Balanced, Dense
            
        # FPS 읽기
        fps = 0
        with open(time_files[0], 'r', encoding='utf-8') as f:
            for line in f:
                if "FPS" in line:
                    try:
                        fps = float(line.split(":")[-1].strip())
                    except:
                        pass
        
        # Trajectory 읽기
        traj = []
        with open(traj_files[0], 'r') as f:
            for line in f:
                coords = [float(x) for x in line.strip().split()]
                if len(coords) == 3:
                    traj.append(coords)
        
        data[algo_name][setting_name] = {
            'fps': fps,
            'traj': np.array(traj)
        }

print(f"데이터 로드 완료: {list(data.keys())}")

# 2. FPS 비교 그래프 (Bar Chart)
print("FPS 비교 그래프 생성 중...")
plt.figure(figsize=(10, 6))

algos = sorted(data.keys())
settings = ["HighQual", "Balanced", "Dense"] # 순서 고정
bar_width = 0.25
index = np.arange(len(settings))

colors = {'SHI': 'skyblue', 'FAST': 'salmon', 'ORB': 'lightgreen'}

for i, algo in enumerate(algos):
    fps_values = []
    for setting in settings:
        val = data.get(algo, {}).get(setting, {}).get('fps', 0)
        fps_values.append(val)
    
    plt.bar(index + i * bar_width, fps_values, bar_width, label=algo, color=colors.get(algo, 'gray'))

plt.xlabel('Settings', fontsize=12)
plt.ylabel('Average FPS', fontsize=12)
plt.title('Performance Comparison (FPS) by Algorithm & Setting', fontsize=14)
plt.xticks(index + bar_width, settings)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "comparison_fps.png"), dpi=300)
plt.close()

# 3. 경로 비교 그래프 (Trajectory X-Z Plane) - 설정별로 묶어서
print("경로 비교 그래프 생성 중...")

# 색상 및 스타일
algo_styles = {'SHI': '-', 'FAST': '--', 'ORB': '-.'}
algo_colors = {'SHI': 'blue', 'FAST': 'red', 'ORB': 'green'}

for setting in settings:
    plt.figure(figsize=(8, 8))
    
    for algo in algos:
        traj_data = data.get(algo, {}).get(setting, {}).get('traj')
        if traj_data is not None and len(traj_data) > 0:
            # X-Z plane (Top-down view usually)
            x = traj_data[:, 0]
            z = traj_data[:, 2] 
            
            plt.plot(x, z, label=f"{algo}", 
                     color=algo_colors.get(algo, 'black'), 
                     linestyle=algo_styles.get(algo, '-'), linewidth=1.5)
    
    plt.title(f'Trajectory Comparison - {setting}', fontsize=14)
    plt.xlabel('X (axis)', fontsize=12)
    plt.ylabel('Z (axis)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.axis('equal') # 비율 유지
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"comparison_traj_{setting}.png"), dpi=300)
    plt.close()

# 4. 전체 경로 비교 (복잡할 수 있으니 대표적으로 Balanced만 하나 더 크게)
plt.figure(figsize=(10, 10))
for algo in algos:
    traj_data = data.get(algo, {}).get('Balanced', {}).get('traj')
    if traj_data is not None:
         plt.plot(traj_data[:, 0], traj_data[:, 2], label=f"{algo} (Balanced)", linewidth=2)

plt.title('Trajectory Overview (Balanced Mode)', fontsize=16)
plt.xlabel('X', fontsize=12)
plt.ylabel('Z', fontsize=12)
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.savefig(os.path.join(OUTPUT_DIR, "comparison_traj_overview.png"), dpi=300)
plt.close()

print(f"모든 그래프가 {OUTPUT_DIR} 에 저장되었습니다.")
