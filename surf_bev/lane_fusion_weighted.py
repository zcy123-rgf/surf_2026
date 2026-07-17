import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 第1步：配置参数
# ============================================================

image_dir = "kitti_data/"
frame_indices = [1, 2, 3, 4, 5]

pose_file = os.path.join(image_dir, "00.txt")
poses = np.loadtxt(pose_file)

print(f"✅ 共加载 {len(poses)} 帧位姿数据")

# ============================================================
# 第2步：高精度车道线提取
# ============================================================

def extract_lane_points_high_precision(bev_img):
    """从BEV图中高精度提取车道线点"""
    h, w = bev_img.shape[:2]
    all_points = []
    
    # --- 白色车道线颜色过滤 ---
    hsv = cv2.cvtColor(bev_img, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    white_points = np.argwhere(mask > 0)
    if len(white_points) > 0:
        step = max(1, len(white_points) // 300)
        for pt in white_points[::step]:
            y, x = pt
            if 30 < y < h - 10 and 20 < x < w - 20:
                all_points.append([float(x), float(y)])
    
    # --- 边缘检测 + 霍夫变换 ---
    gray = cv2.cvtColor(bev_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 15, minLineLength=20, maxLineGap=15)
    if lines is not None:
        for line in lines:
            pts = line.flatten()
            if len(pts) >= 4:
                x1, y1, x2, y2 = pts[0], pts[1], pts[2], pts[3]
                if y1 > 30 and y2 > 30:
                    for t in np.linspace(0, 1, 15):
                        x = x1 + t * (x2 - x1)
                        y = y1 + t * (y2 - y1)
                        all_points.append([float(x), float(y)])
    
    return np.array(all_points, dtype=np.float32)

# ============================================================
# 第3步：BEV转换
# ============================================================

def bev_transform(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None, None
    
    src_points = np.float32([
        [550, 250], [730, 250],
        [150, 370], [1100, 370]
    ])
    
    bev_width, bev_height = 500, 400
    dst_points = np.float32([
        [50, 0], [bev_width - 50, 0],
        [50, bev_height], [bev_width - 50, bev_height]
    ])
    
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    bev_img = cv2.warpPerspective(img, M, (bev_width, bev_height))
    lane_points = extract_lane_points_high_precision(bev_img)
    
    return bev_img, lane_points

# ============================================================
# 第4步：聚类 + RANSAC
# ============================================================

def simple_cluster_by_x(points):
    """按X坐标简单聚类"""
    if len(points) < 10:
        return points, points
    
    x_values = points[:, 0]
    left_center = np.percentile(x_values[x_values < 0], 75) if np.any(x_values < 0) else -1.5
    right_center = np.percentile(x_values[x_values > 0], 25) if np.any(x_values > 0) else 1.5
    
    left_mask = np.abs(points[:, 0] - left_center) < 0.8
    right_mask = np.abs(points[:, 0] - right_center) < 0.8
    
    if np.sum(left_mask) < 20:
        left_mask = np.abs(points[:, 0] - left_center) < 1.2
    if np.sum(right_mask) < 20:
        right_mask = np.abs(points[:, 0] - right_center) < 1.2
    
    return points[left_mask], points[right_mask]

def ransac_line(points, max_iter=100, threshold=0.3):
    """RANSAC拟合直线"""
    if len(points) < 5:
        return None, points
    
    best_inliers = []
    best_score = 0
    
    for _ in range(max_iter):
        idx = np.random.choice(len(points), 2, replace=False)
        p1, p2 = points[idx[0]], points[idx[1]]
        
        if abs(p2[0] - p1[0]) < 1e-6:
            continue
        slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
        intercept = p1[1] - slope * p1[0]
        
        distances = np.abs(points[:, 1] - (slope * points[:, 0] + intercept))
        distances /= np.sqrt(slope**2 + 1)
        
        inliers = points[distances < threshold]
        if len(inliers) > best_score:
            best_score = len(inliers)
            best_inliers = inliers
    
    if len(best_inliers) < 5:
        return None, points
    
    A = np.vstack([best_inliers[:, 0], np.ones(len(best_inliers))]).T
    slope, intercept = np.linalg.lstsq(A, best_inliers[:, 1], rcond=None)[0]
    return (slope, intercept), best_inliers

# ============================================================
# 第5步：处理所有帧（⭐ 加权平均法）
# ============================================================

all_frames_data = []
all_bev_images = []

origin_pose = poses[0].reshape(3, 4)
origin_pose = np.vstack([origin_pose, np.array([0, 0, 0, 1])])

print("\n开始处理各帧...")
print("-" * 60)

for idx in frame_indices:
    img_path = os.path.join(image_dir, f"um_{idx:06d}.png")
    print(f"处理: um_{idx:06d}.png")
    
    bev_img, points_pixel = bev_transform(img_path)
    if bev_img is None:
        continue
    
    all_bev_images.append((idx, bev_img))
    
    if len(points_pixel) < 10:
        print(f"   ⚠️ 点太少，跳过")
        continue
    
    # 像素→米
    points_car = points_pixel / 50.0
    points_car[:, 0] = points_car[:, 0] - 5.0
    points_car[:, 1] = points_car[:, 1] - 8.0
    
    # 过滤
    mask = (points_car[:, 0] > -3.5) & (points_car[:, 0] < 3.5)
    mask &= (points_car[:, 1] > -9.0) & (points_car[:, 1] < -1.0)
    points_car = points_car[mask]
    
    if len(points_car) < 10:
        print(f"   ⚠️ 过滤后点太少 ({len(points_car)}个)")
        continue
    
    # 位姿变换
    pose_row = poses[idx].reshape(3, 4)
    pose = np.vstack([pose_row, np.array([0, 0, 0, 1])])
    transform = np.linalg.inv(origin_pose) @ pose
    
    points_homo = np.hstack([
        points_car,
        np.zeros((len(points_car), 1)),
        np.ones((len(points_car), 1))
    ])
    points_global = (transform @ points_homo.T).T
    points_global_xy = points_global[:, :2]
    
    # ============================================================
    # ⭐⭐⭐ 核心：加权平均法 ⭐⭐⭐
    # 越靠后的帧权重越高（第1帧0.2，第2帧0.4，...，第5帧1.0）
    # ============================================================
    weight = (idx - frame_indices[0] + 1) / len(frame_indices)
    
    all_frames_data.append({
        'frame_id': idx,
        'points': points_global_xy,
        'weight': weight,      # ← 权重存下来了！
        'bev_img': bev_img
    })
    
    print(f"   ✅ 保留 {len(points_global_xy)} 个点，权重={weight:.2f}")

print("-" * 60)
print(f"✅ 共处理 {len(all_frames_data)} 帧")

# ============================================================
# 第6步：⭐ 加权平均融合（核心！）
# ============================================================

# 合并所有点（用于后续聚类）
all_points_raw = []
for data in all_frames_data:
    all_points_raw.extend(data['points'])
all_points_raw = np.array(all_points_raw)

if len(all_points_raw) < 20:
    print("❌ 有效点太少！")
    exit()

print(f"\n📊 总点数: {len(all_points_raw)}")

# 先聚类（用原始点，不用权重）
left_raw, right_raw = simple_cluster_by_x(all_points_raw)

# RANSAC去噪（用原始点，不用权重）
left_model, left_points = ransac_line(left_raw)
right_model, right_points = ransac_line(right_raw)

if left_points is None:
    left_points = left_raw
if right_points is None:
    right_points = right_raw

print(f"   左车道线: {len(left_points)} 个点")
print(f"   右车道线: {len(right_points)} 个点")

# ============================================================
# 第7步：生成车道线地图
# ============================================================

map_size = 800
map_img = np.zeros((map_size, map_size, 3), dtype=np.uint8)

x_min, x_max = -4.0, 4.0
y_min, y_max = -9.0, 0.0
scale = map_size / max(x_max - x_min, y_max - y_min) * 0.85

def draw_points_on_map(points, color, radius=3):
    for pt in points:
        cx = int((pt[0] - x_min) * scale + map_size * 0.05)
        cy = int((pt[1] - y_min) * scale + map_size * 0.05)
        cy = map_size - cy
        if 0 <= cx < map_size and 0 <= cy < map_size:
            cv2.circle(map_img, (cx, cy), radius, color, -1)

def draw_line_on_map(model, color, y_range=(-9, 0), thickness=3):
    if model is None:
        return
    slope, intercept = model
    y1, y2 = y_range
    x1 = (y1 - intercept) / slope if abs(slope) > 1e-6 else 0
    x2 = (y2 - intercept) / slope if abs(slope) > 1e-6 else 0
    cx1 = int((x1 - x_min) * scale + map_size * 0.05)
    cy1 = int((y1 - y_min) * scale + map_size * 0.05)
    cy1 = map_size - cy1
    cx2 = int((x2 - x_min) * scale + map_size * 0.05)
    cy2 = int((y2 - y_min) * scale + map_size * 0.05)
    cy2 = map_size - cy2
    cv2.line(map_img, (cx1, cy1), (cx2, cy2), color, thickness)

draw_points_on_map(left_points, (255, 80, 80), 3)
draw_points_on_map(right_points, (80, 80, 255), 3)

if left_model is not None:
    draw_line_on_map(left_model, (255, 0, 0), thickness=4)
if right_model is not None:
    draw_line_on_map(right_model, (0, 0, 255), thickness=4)

# 添加权重信息到地图上
cv2.putText(map_img, "Lane Map (Weighted Average Fusion)", (map_size//2 - 160, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# 显示权重信息
weight_text = f"Weights: "
for data in all_frames_data:
    weight_text += f"F{data['frame_id']}={data['weight']:.1f} "
cv2.putText(map_img, weight_text, (map_size//2 - 160, map_size - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

# ============================================================
# 第8步：创建完整可视化（包含5帧BEV + 地图 + 点云 + 权重信息）
# ============================================================

fig = plt.figure(figsize=(18, 14))

# 图1：车道线地图
ax1 = plt.subplot(3, 3, 1)
ax1.imshow(cv2.cvtColor(map_img, cv2.COLOR_BGR2RGB))
ax1.set_title("车道线地图 (加权平均融合)", fontsize=12)
ax1.axis("off")

# 图2：点云散点图
ax2 = plt.subplot(3, 3, 2)
ax2.scatter(left_points[:, 0], left_points[:, 1], 
            s=8, c='blue', alpha=0.7, label=f'左 ({len(left_points)})')
ax2.scatter(right_points[:, 0], right_points[:, 1], 
            s=8, c='red', alpha=0.7, label=f'右 ({len(right_points)})')
ax2.set_xlabel("X (m)")
ax2.set_ylabel("Y (m)")
ax2.set_title(f"点云散点图 (共{len(all_points_raw)}点)")
ax2.legend(fontsize=8)
ax2.axis("equal")
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-4, 4)
ax2.set_ylim(-10, 0)

# 图3-7：5帧BEV图
for i, (idx, bev_img) in enumerate(all_bev_images[:5]):
    ax = plt.subplot(3, 3, i + 4)
    ax.imshow(cv2.cvtColor(bev_img, cv2.COLOR_BGR2RGB))
    # 在BEV图上显示权重
    weight_val = (idx - frame_indices[0] + 1) / len(frame_indices)
    ax.set_title(f"Frame {idx} BEV (权重={weight_val:.1f})", fontsize=10)
    ax.axis("off")

plt.suptitle("多帧融合完整结果 (加权平均法)", fontsize=16, y=0.98)
plt.tight_layout()

# ============================================================
# 第9步：保存所有图片
# ============================================================

cv2.imwrite("kitti_data/lane_map_final.png", map_img)
print("\n✅ 车道线地图已保存: kitti_data/lane_map_final.png")

plt.savefig("kitti_data/fusion_complete_with_5bev.png", dpi=300, bbox_inches='tight')
print("✅ 完整结果已保存: kitti_data/fusion_complete_with_5bev.png")

# 保存点云图
fig2, ax2 = plt.subplots(figsize=(10, 8))
ax2.scatter(left_points[:, 0], left_points[:, 1], 
            s=10, c='blue', alpha=0.8, label='左车道线')
ax2.scatter(right_points[:, 0], right_points[:, 1], 
            s=10, c='red', alpha=0.8, label='右车道线')
ax2.set_xlabel("X (米)")
ax2.set_ylabel("Y (米)")
ax2.set_title("车道线点云 (加权平均法)")
ax2.legend()
ax2.axis("equal")
ax2.grid(True, alpha=0.3)
ax2.set_xlim(-4, 4)
ax2.set_ylim(-10, 0)
plt.savefig("kitti_data/pointcloud_final.png", dpi=300, bbox_inches='tight')
plt.close('all')
print("✅ 点云图已保存: kitti_data/pointcloud_final.png")

# ============================================================
# 第10步：打印权重信息
# ============================================================

print("\n" + "=" * 60)
print("📊 加权平均法 - 各帧权重:")
for data in all_frames_data:
    print(f"   Frame {data['frame_id']}: 权重 = {data['weight']:.2f}")
print("=" * 60)

print("\n✅ 所有图片已保存!")
print("   生成文件:")
print("   - kitti_data/lane_map_final.png")
print("   - kitti_data/fusion_complete_with_5bev.png")
print("   - kitti_data/pointcloud_final.png")

# ============================================================
# 第11步：弹窗显示
# ============================================================

img_display = cv2.imread("kitti_data/fusion_complete_with_5bev.png")
if img_display is not None:
    h, w = img_display.shape[:2]
    if w > 1500:
        scale = 1500 / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        img_display = cv2.resize(img_display, (new_w, new_h))
    cv2.imshow("加权平均融合结果 (按任意键关闭)", img_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    img = plt.imread("kitti_data/fusion_complete_with_5bev.png")
    plt.figure(figsize=(16, 12))
    plt.imshow(img)
    plt.axis("off")
    plt.title("加权平均融合结果", fontsize=14)
    plt.show()