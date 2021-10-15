# Face-Detection
仅使用 15 行 Python 代码就可以在你的系统上实现面部检测！

**🌊 作者主页：[海拥](https://haiyong.blog.csdn.net/)
🌊 作者简介：🏆CSDN全栈领域优质创作者、🥇HDZ核心组成员、🥈蝉联C站周榜前十
🌊 粉丝福利：[粉丝群](https://app.yinxiang.com/fx/8aa8eb1b-7d45-4793-a160-b990d9da2e75) 每周送四本书，每月送各种小礼品(搪瓷杯、抱枕、鼠标垫、马克杯等)**

<a href="#jump99"><font size="5" color="#03a9f4"><b><u>直接跳到末尾</u></b></font></a>  **去领资料**

无论你是最近开始探索OpenCV还是已经使用它很长一段时间，在任何一种情况下，你都一定遇到过“人脸检测”这个词。随着机器变得越来越智能，它们模仿人类行为的能力似乎也在增加，而人脸检测就是人工智能的进步之一。

所以今天，我们将快速了解一下面部检测是什么，为什么它很有用，以及如何仅用 15 行代码就可以在你的系统上实际实现面部检测！

让我们从了解面部检测开始。

## 什么是人脸检测？

人脸检测是一种基于人工智能的计算机技术，能够识别和定位数码照片和视频中人脸的存在。简而言之，机器检测图像或视频中人脸的能力。

由于人工智能的重大进步，现在可以检测图像或视频中的人脸，无论光照条件、肤色、头部姿势和背景如何。

人脸检测是几个人脸相关应用程序的起点，例如人脸识别或人脸验证。如今，大多数数码设备中的摄像头都利用人脸检测技术来检测人脸所在的位置并相应地调整焦距。

那么人脸检测是如何工作的呢？
很高兴你问了！任何人脸检测应用程序的主干都是一种算法（机器遵循的简单分步指南），可帮助确定图像是正图像（有脸的图像）还是负图像（没有人脸的图像）。

为了准确地做到这一点，算法在包含数十万张人脸图像和非人脸图像的海量数据集上进行了训练。这种经过训练的机器学习算法可以检测图像中是否有人脸，如果检测到人脸，还会放置一个边界框。

## 使用 OpenCV 进行人脸检测
计算机视觉是人工智能中最令人兴奋和最具挑战性的任务之一，有几个软件包可用于解决与计算机视觉相关的问题。OpenCV 是迄今为止最流行的用于解决基于计算机视觉的问题的开源库。

OpenCV 库的下载量超过1800 万次，活跃的用户社区拥有 47000 名成员。它拥有 2500 种优化算法，包括一整套经典和最先进的计算机视觉和机器学习算法，使其成为机器学习领域最重要的库之一。

图像中的人脸检测是一个简单的 3 步过程：

## 第一步：安装并导入open-cv模块：

```python
pip install opencv-python
```

```python
import cv2
import matplotlib.pyplot as plt # 用于绘制图像
```

## 第 2 步：将 XML 文件加载到系统中
下载 Haar-cascade Classifier XML 文件并将其加载到系统中：

Haar-cascade Classifier 是一种机器学习算法，我们用大量图像训练级联函数。根据不同的目标对象有不同类型的级联分类器，这里我们将使用考虑人脸的分类器将其识别为目标对象。

你可以[点击此处](https://github.com/wanghao221/Face-Detection/blob/main/haarcascade_frontalface_default.xml)找到用于人脸检测的经过训练的分类器 XML 文件


```python
# 加载级联
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
```
## 第 3 步：检测人脸并在其周围绘制边界框
使用 Haar-cascade 分类器中的 detectMultiScale() 函数检测人脸并在其周围绘制边界框：

```python
# 读取输入图像
img = cv2.imread('test.png')

# 检测人脸
faces = face_cascade.detectMultiScale(image = img, scaleFactor = 1.1, minNeighbors = 5)

# 在人脸周围绘制边界框
for (x, y, w, h) in faces:
      cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

# 显示图像中检测到的人脸数量
print(len(faces),"faces detected!")

# 绘制检测到人脸的图像
finalimg = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
plt.figure(figsize=(12,12))
plt.imshow(finalimg) 
plt.axis("off")
plt.show()
```
**detectMultiScale() 参数：**

>  - **image**： CV_8U 类型的矩阵，其中包含检测到对象的图像。
>  - **scaleFactor**：指定在每个图像比例下图像尺寸减小多少的参数。
>  - **minNeighbors**：参数指定每个候选矩形应该保留多少邻居。

可能需要调整一下这些值来获取最佳结果。

![在这里插入图片描述](https://img-blog.csdnimg.cn/dd5e455cf04d43ed953ef5004d97fdaa.png)


就像这样，你可以实现计算机视觉最独特的应用程序之一。可以在下面的GitHub找到整个人脸检测实现的详细代码模板。

[https://github.com/wanghao221/Face-Detection](https://github.com/wanghao221/Face-Detection)

注意：本教程仅适用于图像文件中的人脸检测，而不适用于实时摄像机源或视频。

是不是感觉很棒？你刚刚学习了如何实现人工智能和机器学习最有趣的应用之一。希望你喜欢我的博客。谢谢阅读！

🌊 <font color="#F08080">**面试题库：Java、Python、前端核心知识点大全和面试真题资料
🌊 电子图书：图灵程序丛书 300本、机械工业出版社6000册免费正版图书
🌊 办公用品：精品PPT模板几千套，简历模板一千多套
🌊 学习资料：2300套PHP建站源码，微信小程序入门资料**</font>

**📣 注意：**

回复【进群】领书不迷路，群内 <font color="#e66b6d">每位成员</font> 我都会送一本。回复【资源】可获取上面的资料👇🏻👇🏻👇🏻**

<img src="https://img-blog.csdnimg.cn/07d5c80f7ae44ca7b8d441cc0c19943f.png#pic_center%E2%80%9D"#pic_center =400x>
