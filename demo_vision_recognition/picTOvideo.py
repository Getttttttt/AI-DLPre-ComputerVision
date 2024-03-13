import cv2
import os

# 图片文件夹路径
img_folder_path = './something_else/annotated_videos/22983'

# 视频输出路径
video_output_path = './something_else/annotated_videos/22983.mp4'

# 获取图片文件列表
img_files = sorted([f for f in os.listdir(img_folder_path) if f.endswith('.jpg')])

# 读取第一张图片获取尺寸
img = cv2.imread(os.path.join(img_folder_path, img_files[0]))
height, width, _ = img.shape

# 设置视频编码器和输出文件
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(video_output_path, fourcc, 10.0, (width, height))

# 遍历图片并写入视频
for img_file in img_files:
    img_path = os.path.join(img_folder_path, img_file)
    img = cv2.imread(img_path)
    video_writer.write(img)

# 关闭视频写入器
video_writer.release()
