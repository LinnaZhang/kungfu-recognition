import cv2
import mediapipe as mp

# 初始化 MediaPipe Pose 模块
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils  # 用于绘制骨骼点

# 打开视频文件
cap = cv2.VideoCapture('input_video.mp4')

# 获取视频的宽度、高度和帧率
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 创建视频写入器
output_file = 'output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 定义编码格式
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

# 创建 Pose 实例
with mp_pose.Pose(
    static_image_mode=False,  # 设置为 False 以处理视频流
    model_complexity=1,       # 模型复杂度（0:轻量，1:中等，2:高）
    min_detection_confidence=0.5,  # 检测置信度阈值
    min_tracking_confidence=0.5    # 跟踪置信度阈值
) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # 视频结束

        # 将帧从 BGR 转换为 RGB（MediaPipe 需要 RGB 格式）
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 处理帧并获取姿态检测结果
        results = pose.process(image)

        # 将图像转换回 BGR 格式以便 OpenCV 显示
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 如果检测到姿态，绘制骨骼点
        if results.pose_landmarks:
            # 绘制骨骼点
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                # 为每个骨骼点设置不同的颜色
                color = (int(255 * (idx % 3) / 2), int(255 * ((idx + 1) % 3) / 2), int(255 * ((idx + 2) % 3) / 2))
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2),  # 骨骼点颜色和样式
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1))  # 连接线颜色和样式

            # 写入处理后的帧到输出视频
            out.write(image)

        # 显示结果
        # cv2.imshow('MediaPipe Pose', image)

        # 按下 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()