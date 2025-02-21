import cv2
import mediapipe as mp
import numpy as np

# 初始化 MediaPipe Pose 模块
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 定义具体动作的判断函数
def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

def is_bingbu_baoquan(left_shoulder, right_shoulder, left_wrist, right_wrist, left_ankle, right_ankle):
    # 双脚间距判断阈值
    foot_distance_threshold = 0.05
    # 双手位置与肩部的相对位置判断阈值
    hand_shoulder_distance_threshold = 0.1

    # 计算双脚的水平间距
    foot_distance = abs(left_ankle.x - right_ankle.x)

    # 计算双手与肩部的水平和垂直相对位置
    left_hand_shoulder_distance_x = abs(left_wrist.x - left_shoulder.x)
    left_hand_shoulder_distance_y = abs(left_wrist.y - left_shoulder.y)
    right_hand_shoulder_distance_x = abs(right_wrist.x - right_shoulder.x)
    right_hand_shoulder_distance_y = abs(right_wrist.y - right_shoulder.y)

    # 判断双脚是否并拢，双手是否靠近肩部
    if foot_distance < foot_distance_threshold and \
            left_hand_shoulder_distance_x < hand_shoulder_distance_threshold and \
            left_hand_shoulder_distance_y < hand_shoulder_distance_threshold and \
            right_hand_shoulder_distance_x < hand_shoulder_distance_threshold and \
            right_hand_shoulder_distance_y < hand_shoulder_distance_threshold:
        return True
    return False

def is_gongbu_chongquan(left_knee, right_knee, left_wrist, right_wrist, left_hip, right_hip, landmarks):
    # 弓步角度判断阈值
    knee_angle_threshold = 120
    # 冲拳位置判断阈值
    punch_distance_threshold = 0.1
    # 弓步腿的膝盖角度判断阈值
    front_knee_angle_threshold = 120
    back_knee_angle_threshold = 160

    # 获取左右脚踝的实际地标对象
    left_ankle = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
    right_ankle = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]

    # 计算左右腿的角度
    left_leg_angle = calculate_angle(left_hip, left_knee, left_ankle)
    right_leg_angle = calculate_angle(right_hip, right_knee, right_ankle)

    # 判断是否为左弓步
    if left_leg_angle < front_knee_angle_threshold and right_leg_angle > back_knee_angle_threshold:
        # 判断左拳是否收抱于腰间
        if left_wrist.y > left_hip.y and abs(left_wrist.x - left_hip.x) < punch_distance_threshold:
            # 判断右拳是否向前平冲
            if right_wrist.x > right_hip.x + punch_distance_threshold and right_wrist.y < right_shoulder.y:
                return True

    # 判断是否为右弓步
    elif right_leg_angle < front_knee_angle_threshold and left_leg_angle > back_knee_angle_threshold:
        # 判断右拳是否收抱于腰间
        if right_wrist.y > right_hip.y and abs(right_wrist.x - right_hip.x) < punch_distance_threshold:
            # 判断左拳是否向前平冲
            if left_wrist.x < left_hip.x - punch_distance_threshold and left_wrist.y < left_shoulder.y:
                return True

    return False

# 定义动作检测函数
def detect_action(landmarks):
    # 获取关键点坐标
    left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_elbow = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
    right_elbow = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
    left_wrist = landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
    left_hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
    left_knee = landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
    right_knee = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]

    # 判断动作
    left_ankle = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
    right_ankle = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
    if is_gongbu_chongquan(left_knee, right_knee, left_wrist, right_wrist, left_hip, right_hip, landmarks):
        return "弓步冲拳"
    elif is_bingbu_baoquan(left_shoulder, right_shoulder, left_wrist, right_wrist, left_ankle, right_ankle):
        return "起势-并步抱拳"
    # 添加其他动作的判断逻辑
    else:
        return "连接动作"

# 打开视频文件
cap = cv2.VideoCapture('StandardSubActions/T001A002.mp4')

# 获取视频的宽度、高度和帧率
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 创建视频写入器
output_file = 'output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

# 定义动作列表
actions = [
    "起势-并步抱拳", "弓步冲拳", "弹踢冲拳", "马步冲拳", "弓步冲拳", "弹踢冲拳", "马步冲拳",
    "弓步左右推掌", "按拦推掌", "上架蹬踢", "马步推掌", "弓步左右推掌", "按拦推掌", "上架蹬踢",
    "马步推掌", "弓步双冲拳", "抱拳蹬踢", "弓步冲拳", "马步架打", "弓步双冲拳", "抱拳蹬踢",
    "弓步冲拳", "马步架打", "弓步左右冲拳", "回身弓步冲拳", "翻身劈砸", "弹踢冲拳", "马步冲拳",
    "弓步左右冲拳", "回身弓步冲拳", "翻身劈砸", "弹踢冲拳", "马步冲拳", "收势-并步抱拳"
]

# 创建 Pose 实例
with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:
    current_action = None
    action_start_frame = None
    action_end_frame = None
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 将帧从 BGR 转换为 RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 处理帧并获取姿态检测结果
        results = pose.process(image)

        # 将图像转换回 BGR 格式以便 OpenCV 显示
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            # 绘制骨骼点
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0)),  # 骨骼点颜色
                mp_drawing.DrawingSpec(color=(255, 255, 255))  # 连接线颜色
            )

            # 检测当前动作
            detected_action = detect_action(results.pose_landmarks)
            if detected_action != current_action:
                if current_action is not None:
                    action_end_frame = frame_count
                    print(f"Action: {current_action}, Frames: {action_start_frame}-{action_end_frame}")
                current_action = detected_action
                action_start_frame = frame_count

        # 写入处理后的帧到输出视频
        out.write(image)

        # 显示结果
        cv2.imshow('MediaPipe Pose', image)

        # 按下 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()