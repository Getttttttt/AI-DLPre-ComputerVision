## Part2 计算机视觉的识别

### 单目视觉 Monocular Vision

#### 定义

单目视觉是指主要依靠单个摄像机或传感器来获取图像信息，并通过计算机视觉技术对这些图像进行处理和分析的过程。

#### 应用场景

虽然单目视觉无法直接提供深度信息，但它在处理二维图像、进行图像识别和分类等方面非常有效。

1. **自动驾驶**：在自动驾驶领域中用于道路检测、车辆检测、行人识别等任务，帮助车辆感知周围环境，做出相应的驾驶决策。
2. **人脸识别**： 人脸检测、识别和表情分析，应用于安防监控、门禁系统、人机交互等场景。
3. **医学影像分析**：可用于图像分割、病灶检测、医学图像配准等任务，帮助医生进行疾病诊断和治疗。
4. **工业质检**：可以用于产品质量检测、缺陷检测、尺寸测量等任务，提高生产线的效率和产品质量。
5. **相机标定：**基于单目视觉，捕捉图像中的畸变参数，从而对图像进行畸变校正，以获得更为准确的视觉数据。

#### 技术原理

单目视觉的核心原理是从二维图像中提取有用信息，用于理解和分析三维世界。通常涉及以下步骤：

1. **图像捕获**：相机捕获现实世界的二维表示。
2. **特征提取**：基于识别算法识别图像中的关键特征点或边缘。包括图像中的线条、角点、轮廓等。
3. **特征匹配与跟踪**：在连续的图像帧中跟踪特征点，以了解物体或场景的动态变化。
4. **三维场景恢复**：由于单目视觉只依靠单个摄像机，因此单目视觉不能直接测量深度。但是，可以通过其他方法如运动视差、尺度不变特征变换（SIFT）等来间接推断深度信息。
5. **图像理解**：应用特定的算法（如物体检测、图像分类、机器学习或深度学习算法）来解释图像内容，并对图像中的目标进行检测和定位，如目标物体、人脸等。
6. **决策与应用**：根据目标检测和识别的结果，做出相应的决策或应用，如自动驾驶中的车辆控制、人脸识别中的身份认证等。

单目视觉的重要原理是**针孔相机模型**，用于描述三维世界如何向二维图像上映射。具体数学计算公式如下：
$$
p=K[R|t]P
$$

- P表示三维世界中的一个点。
- p是该点在图像平面上的投影。
- K是内部参数矩阵，包含焦距和主点坐标。
- R和t是相机的旋转和平移向量，代表外部参数。

#### Demo

**1. 相机标定**

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 棋盘格参数

cross_points = (9, 6)
square_size = 1.0  # 假设棋盘格每个方块的大小为1.0单位

# 加载图像

image_path = 'Chessboard_Photo.jpg'
image = cv2.imread(image_path)
image = cv2.resize(image, (int(image.shape[1] / 2), int(image.shape[0] / 2)))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 寻找棋盘格角点

ret, corners = cv2.findChessboardCorners(gray, cross_points, None)

# 如果找到足够的角点，则进行标定
if ret == True:
    # 准备对象点
    objp = np.zeros((cross_points[0] * cross_points[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:cross_points[0], 0:cross_points[1]].T.reshape(-1, 2)
    objp *= square_size
# 将对象点和图像点放入数组中
objpoints = []  # 真实世界中的点
imgpoints = []  # 图像中的点

objpoints.append(objp)
imgpoints.append(corners)

# 进行相机标定
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# 标记角点并显示
img = cv2.drawChessboardCorners(image.copy(), cross_points, corners, int(ret))

# 透视变换
# 原始图像尺寸
h, w = image.shape[:2]

# 获取角点的坐标
top_left, top_right, bottom_right, bottom_left = corners[0][0], corners[8][0], corners[-1][0], corners[-9][0]
pts1 = np.float32([top_left, top_right, bottom_right, bottom_left])

# 计算四个角点到图像边缘的最大距离
maxDistToLeftEdge = max(top_left[0], bottom_left[0])
maxDistToRightEdge = max(w - top_right[0], w - bottom_right[0])
maxDistToTopEdge = max(top_left[1], top_right[1])
maxDistToBottomEdge = max(h - bottom_left[1], h - bottom_right[1])

# 使用最大距离来定义目标图像的大小
maxWidth =int(w + maxDistToLeftEdge + maxDistToRightEdge)
maxHeight =int( h + maxDistToTopEdge + maxDistToBottomEdge)

# 计算目标点的坐标，使整张图片在透视变换后能居中显示
pts2 = np.float32([
    [maxDistToLeftEdge, maxDistToTopEdge],
    [maxWidth - maxDistToRightEdge - 1, maxDistToTopEdge],
    [maxWidth - maxDistToRightEdge - 1, maxHeight - maxDistToBottomEdge - 1],
    [maxDistToLeftEdge, maxHeight - maxDistToBottomEdge - 1]
])

# 获取透视变换矩阵
M = cv2.getPerspectiveTransform(pts1, pts2)

# 应用透视变换
dst_perspective = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

# 显示原图及透视变换后的图片
plt.figure(figsize=(8, 6))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original Image with Corners')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(dst_perspective, cv2.COLOR_BGR2RGB))
plt.title('Perspective Transform')

plt.show()
else:
    print("找不到足够的角点，请检查图片是否适合棋盘格标定。")
```

![img](https://img-blog.csdnimg.cn/b51753d02577401bae24712cc60b3229.png)

### 双目视觉 Binocular Stereo Vision

#### 定义

双目视觉，又称立体视觉，利用两个摄像机从略微不同的角度捕捉图像，模拟人类的双眼视觉机制，从而获得场景的深度信息。

#### 应用场景

这种方法在获取深度信息和三维场景重建方面比单目视觉有着显著优势。

1. **深度感知与三维重建**：可用于测量场景中物体的距离和深度，从而实现深度感知。
2. **立体成像和虚拟现实**基于双目视觉，可以创建逼真的三维立体图像，用于虚拟现实VR中的沉浸式体验，如虚拟游戏、模拟培训等。
3. **机器人导航**：利用深度信息帮助机器人进行空间感知和路径规划，特别适用于自动化和工业机器人。
4. **增强现实AR**：结合真实场景和计算机生成的图像，需要精确的空间和深度信息以提供更真实的体验。
5. **自动驾驶汽车**：为自动驾驶系统提供深度感知能力，用于障碍物检测、车道识别和环境理解。

#### 技术原理

双目视觉的核心在于利用两个相机的视差来计算深度信息。视差是指同一物体在两个相机视角中的图像位置差异。

物体点P的深度Z的数学计算公式如下：
$$
Z=b×f/d
$$
其中，d是两个相机成像平面上对应点p1和p2之间的视差，f为相机的焦距，d为两相机之间的基线距离。

![img](https://img-blog.csdnimg.cn/img_convert/e8d2228b770d48b5d830ee57f3b36421.png)

### 图像分类与目标检测

#### 理论框架

##### CNN

原理：利用卷积操作、池化操作和非线性激励函数构建多层神经网络，以从原始图像数据中提取特征并进行分类。

一个卷积神经网络主要由以下5层组成：

- 数据输入层 Input layer：输入图像等信息

- 卷积计算层 CONV layer：一个降维的过程，通过卷积核的不停移动计算，提取图像中最有用的特征。基于卷积核计算得到的新的二维矩阵，被称为特征图。
  $$
  (f∗g)(x,y)=∑_{i,~j}f(i,j)g(x−i,y−j)
  $$
  其中，f是输入的图像的像素值，g是卷积核，(f∗g)(x,y)是卷积核在输入图像上的一个位置的输出值。

- ReLU激励层 ReLU layer：把卷积层输出结果做非线性映射，因为卷积层的计算是一种线性计算，对非线性情况无法很好拟合。
  $$
  ReLU(x)=max(0,x)
  $$

- 池化层  Pooling layer：池化操作用于减少特征图的空间大小，并保留最重要的特征。常用的池化操作包括最大池化（Max Pooling）和平均池化（Average Pooling）。
  $$
  Max Pooling(x,y)=max_{i,~j}f(x+i,y+j)
  $$
  其中，f是输入的特征图，(x,y)是输出特征图中的一个位置。

- 全连接层 FC layer：用于将特征图转换为输出类别的概率分布。全连接层中的每个神经元都与前一层中的所有神经元相连。

![img](https://img-blog.csdnimg.cn/20210703003217329.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0V6cmExOTkx,size_16,color_FFFFFF,t_70)

##### R-CNN

R-CNN是一种目标检测算法，通过首先生成候选区域（region proposals），然后对这些候选区域进行分类和边界框回归来检测图像中的目标。通过利用深度学习中的卷积神经网络来提取图像特征，并结合传统的机器学习方法（如SVM）来进行目标检测，从而在目标检测任务上取得了较好的性能。

其核心思想是将目标检测任务分解为两个子任务：目标定位（Localization）和目标分类（Classification）。

- **候选区域生成**： R-CNN首先利用区域建议方法（如Selective Search或EdgeBoxes）生成一系列候选区域，这些候选区域可能包含了图像中可能存在的目标。
- **特征提取**： 对于每个候选区域，R-CNN使用预训练的卷积神经网络（通常是在大规模图像数据集上预训练的CNN模型，如AlexNet、VGG、ResNet等）提取特征。对于每个候选区域，将其调整为固定大小，并输入到CNN中，从而获得与每个区域相关联的特征向量。
- **目标分类**： 特征向量传递到一个线性支持向量机（SVM）分类器，用于将候选区域分类为目标类别（如人、汽车、狗等）或背景类别。
- **边界框回归**： 同时，为了更准确地定位目标，R-CNN使用回归器对每个候选区域的边界框进行微调，以使其更加精确地拟合目标的真实位置。
- **损失函数**： 对于目标分类任务，R-CNN通常使用多类别交叉熵损失函数。对于边界框回归任务，可以使用平滑的L1损失函数。

![img](https://bbsmax.ikafan.com/static/L3Byb3h5L2h0dHBzL2ltZzIwMTguY25ibG9ncy5jb20vYmxvZy80Mzk3NjEvMjAxOTAyLzQzOTc2MS0yMDE5MDIxNTE1NDQxMzkxOS0xODMwNDEzMTQ4LmpwZw==.jpg)

##### YOLO

YOLO（You Only Look Once）是一种快速、实时的目标检测算法，其核心思想是将目标检测任务转化为单个神经网络的回归问题，同时在图像中直接预测边界框和类别概率。

- **单阶段检测**： YOLO是一种单阶段（single-stage）检测器，与传统的两阶段（two-stage）方法（如R-CNN系列）不同，它不需要生成候选区域，而是直接在整个图像上执行目标检测和分类任务。
- **网格划分**： YOLO将输入图像分为固定大小的网格（例如 7x7 或 13x13），每个网格负责检测该网格内的目标。每个网格预测固定数量的边界框和类别概率。
- **边界框预测**： 每个网格预测多个边界框，每个边界框由5个参数表示：边界框中心的坐标（x, y）、边界框的宽度和高度（w, h）以及目标存在的置信度（confidence）。
- **类别预测**： 每个边界框还预测一个类别概率分布，表示该边界框中包含不同类别目标的可能性。通常使用 softmax 函数将原始输出转换为概率分布。
- **损失函数**： YOLO使用综合的损失函数来同时优化边界框坐标和类别概率。损失函数包括两部分：
  - **边界框损失**：使用均方误差（Mean Squared Error，MSE）来衡量预测边界框与真实边界框之间的差异。
  - **分类损失**：使用交叉熵损失函数来衡量预测类别概率与真实类别标签之间的差异。
- **非最大抑制**： 为了消除重叠的边界框，并保留具有最高置信度的目标边界框，YOLO使用非最大抑制（Non-Maximum Suppression，NMS）算法对预测的边界框进行筛选和合并。

![img](https://img-blog.csdnimg.cn/b64a9e304d824e5eb2f530eb1163ce40.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAR3V5Y3lubm5ubg==,size_20,color_FFFFFF,t_70,g_se,x_16)

##### RNNs

RNNs是一种专门用于处理序列数据的神经网络结构，在网络的隐藏层之间引入循环连接，使得网络可以对序列数据进行处理，同时在不同时间步共享权重参数，允许信息在网络内部传递循环。这种设计使得RNN能够捕捉序列数据中的时间依赖关系，对于自然语言处理、时间序列预测等任务非常有效。

RNN的基本结构包括一个输入层、一个隐藏层和一个输出层。隐藏层的输出不仅会传递到输出层，还会反馈到隐藏层的输入中，形成循环连接，这使得网络在处理序列数据时具有记忆能力。

假设在时间步 *t*，RNN的输入是 *xt*，隐藏层的状态是 *ht*，输出是 *yt*。RNN的状态转移公式可以表示为：
$$
h_t=σ(W_{hx}x_t+W_{hh}h_{t−1}+b_h)
$$
其中，*W*是权重矩阵，*σ*是激活函数，通常为*tanh*或*ReLU*。

隐藏层的输出*yt*可以根据隐藏状态*ht*计算得到：
$$
y 
t
​
 =σ(W_{yh}
 h 
t
​
 +b_y
 )
$$
![img](https://img-blog.csdn.net/20180310144117576?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvVHdUNTIwTHk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

##### BiLSTM

双向长短期记忆网络（Bidirectional Long Short-Term Memory，BiLSTM）是一种循环神经网络（RNN）的变种，它在每个时间步使用两个方向的隐藏状态，从而更好地捕获序列数据中的长期依赖关系。BiLSTM结合了正向（forward）和反向（backward）两个方向的信息，从而提高了对序列数据的建模能力。

![img](https://img1.baidu.com/it/u=3257795394,2809064521&fm=253&fmt=auto&app=138&f=JPEG?w=1267&h=500)

#### 应用案例分析：HOI (Human Object Interacting)

HOI的目的是理解场景中人物关系，包括人和物的定位识别，以及交互的判别。人物交互理解对于关系学习，场景理解和动作理解都具有重要的意义。人物交互（Human-Object Interaction）最早来源于动作理解，相关人员发现人周边的被人交互的物体对于人的动作识别能够提供很强的判别信息。![](D:\大学\人工智能与深度学习\课前展示\hotr_pipeline.png)

##### 简单的识别功能

1. 基础图像识别模型

https://github.com/xinyu1205/recognize-anything.git

- **卓越的图像识别能力**：RAM++是下一代RAM，可以**高精度识别任何类别**，包括**预定义的常见类别和多样化的开放集类别**。RAM++ 在常见标签类别、不常见标签类别和人机交互短语方面优于现有的 SOTA 图像基础识别模型。
  - RAM++是一种增强型的记忆增强神经网络（Memory Augmented Neural Network）。RAM++结合了神经网络和外部记忆存储器，其中外部记忆存储器充当了网络的额外存储空间，可以存储大量信息。在每个时间步，RAM++将当前输入和网络的内部状态传递给控制器（Controller），控制器可以读取、写入和更新外部记忆存储器，并生成输出。这样，网络在处理序列数据时不仅可以利用内部状态，还可以利用外部记忆存储器中的信息，从而更有效地捕捉序列中的长期依赖关系和复杂模式。
- **强大的视觉语义分析**：将 Tag2Text 和 RAM 与本地化模型（Grounding-DINO 和 SAM）相结合，并在Grounded-SAM项目中开发了强大的视觉语义分析管道。
  - Tag2Text通过对视觉语言模型引入图片标记任务（类似于给一个图片打个多个与图片有关的label）来指导模型更好的学习视觉-语言特征。
    - ![tag2text_framework](D:\大学\人工智能与深度学习\课前展示\AI-DLPre-ComputerVision\demo_HOI\recognize-anything\images\tag2text_framework.png)
  - RAM结构上与 Tag2Text 相似，Tag2Text 有3个分支，tagging，generation 和 alignment；RAM 只保留了 Tagging 和 Generation 两个，其中 Tagging 分支用来多tags推理，完成识别任务；Generation用来做 image caption任务；Tag2Text 中的alignment是做 Visual-Language Features学习的，在这里被移除了。
    - ![ram_plus_framework](D:\大学\人工智能与深度学习\课前展示\AI-DLPre-ComputerVision\demo_HOI\recognize-anything\images\ram_plus_framework.jpg)

- 识别效果对比：

![ram_grounded_sam](D:\大学\人工智能与深度学习\课前展示\ram_grounded_sam.jpg)

![tagging_results](D:\大学\人工智能与深度学习\课前展示\tagging_results.jpg)

2. Something-else数据集

使用Something-Something数据集，对视频中人与物体交互中每个对象和手的每帧边界框注释。

展示视频：对应asset中的“862” “57082” “22983”

![0027](D:\大学\人工智能与深度学习\课前展示\AI-DLPre-ComputerVision\demo_HOI\something_else\annotated_videos\862\0027.jpg)

![model](D:\大学\人工智能与深度学习\课前展示\AI-DLPre-ComputerVision\demo_HOI\something_else\figures\model.png)

##### 人与物体的交互判断

HOI模型可以对图片中的人和物体进行识别，并根据方向、位置等特征信息判断人和物体之间的交互关系。

![01. data_acquisition](D:\大学\人工智能与深度学习\课前展示\01. data_acquisition.jpg)

1. DRG---对人和物体之间的对偶关系进行识别

- 利用抽象的空间语义表示来描述每个人物对，并通过双重关系图（一个以人为中心，一个以对象为中心）聚合场景的上下文信息，使用对偶关系图有效地捕捉了场景中的判别线索，以解决局部预测的歧义。
  ![teaser](D:\大学\人工智能与深度学习\课前展示\teaser.png)

![img](https://www.chengao.vision/DRG/files/iterative_update.png)

![img](https://www.chengao.vision/DRG/files/HICO-DET.jpg)

- https://github.com/vt-vl-lab/DRG.git

2. Bongard-HOI

- Introduction：Bongard-HOI 是一个挑战模型视觉推理能力的数据集，旨在从自然图像中学习和识别人与物体的交互 (HOI)。 它受到经典 Bongard 问题的启发，该问题需要少量概念学习和上下文相关推理。 该数据集包含 1,200 个少样本实例，每个实例由六张图像组成：三张正图和三张负图。 正图像具有共同的 HOI 概念，而负图像与正图像仅在动作标签上有所不同。 该数据集还具有多个测试集，训练和测试 HOI 概念之间有不同程度的重叠，以衡量模型的泛化性能。

![overview.png](https://github.com/NVlabs/Bongard-HOI/blob/master/assets/overview.png?raw=true)

- https://github.com/NVlabs/Bongard-HOI.git

3. iCAN

- 通过以实例为中心的注意力网络实现人物交互检测
- https://github.com/vt-vl-lab/iCAN.git

（如果word里的动图不行的话，asset里面也存了：“chatting.gif”）

![HOI](D:\大学\人工智能与深度学习\课前展示\HOI.gif)

4. DiffHOI和SynHOI

- **DiffHOI**：第一个利用生成和代表性功能来提升HOI任务性能的框架。
- **SynHOI**：一个平衡类别、大规模、高多样性的合成 HOI 数据集。

（如果word里的动图不行的话，asset里面也存了：“SynHOI_vis.gif”）

###### ![img](https://github.com/IDEA-Research/DiffHOI/raw/master/assets/SynHOI_vis.gif)

- [GitHub - IDEA-Research/DiffHOI: Official implementation of the paper "Boosting Human-Object Interaction Detection with Text-to-Image Diffusion Model"](https://github.com/IDEA-Research/DiffHOI)

##### 人物交互的分类识别

基于对于人与物体交互的检测，对交互类型进行分类识别。

1. AVA

- AVA数据集: HOIs (human-object, human-human), and pose (body motion) actions

![image-20240312001514763](C:\Users\rita\AppData\Roaming\Typora\typora-user-images\image-20240312001514763.png)

- https://research.google.com/ava/

##### 基于HOI的3D重建与生成

1. ParaHome：将日常家庭活动参数化，实现人与物交互的 3D 生成建模

- https://jlogkim.github.io/parahome/

- 视频演示：https://www.youtube.com/embed/HeXqiK0eGec?si=mtAmctx0JHHYD6Ac （asset中的“3D还原”）

  <video src="D:\大学\人工智能与深度学习\课前展示\3D还原.mp4"></video>

![预告片图片](https://jlogkim.github.io/parahome/static/images/teaser.jpg)

