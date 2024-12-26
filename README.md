# Basic-Understanding-of-deep-learning
课程作业的主题正好和我最近学习方向契合，在之前的过程中，感觉如果直接接触卷积网络有点抽象，所以选择从最原始开始学习，整个项目是从0一直到卷积神经网络前我的相关理解（肯定有不对的）
## 文件介绍
最主要的是深度学习理解**深度学习基础报告**这个word文档，相当于一个说明书，标明了代码属于哪一部分的知识，此外第一章不在李沐的课程范围体系内，但是我觉得有助于理解训练的本质，相当于一个引入。
**在学习之前可先看readme的下面这一部分做准备工作。**
## 准备工作
如果要学习深度学习神经网络的相关知识，首先需要具备**高等数学**、**概率论与数理统计**和**线性代数**的数学基本知识。同时需要掌握**Notebook**、**Python**或者其它主流编程语言。本总结主要涉及的部分为从零基础开始到卷积神经网络之前的框架构建，因此出于通俗易懂的角度考虑，有些点理解得较为肤浅，请阅读者在往后学习时如果出现认识矛盾的地方注意甄别。
那么由于本人使用的是**Pytorch**也就是以python的torch库为基础的语言，因此在这里需要对相关的资料和环境搭建做一些阐述。首先学习过程是以B站UP主@跟李沐学AI的动手学深度学习部分为基础，之所以这里也分享代码是因为链接给的代码在pytorch下训练图像显示会有点问题，其相关教材电子版及代码链接为：
**[《动手学深度学习》 — 动手学深度学习 2.0.0 documentation](https://zh.d2l.ai/index.html)**

对于在一些IDE如Pycharm学习的同学，在课程的前期部分可以用只带CPU的电脑进行学习，但是学到**第六章卷积神经网络LeNet(CPU勉强能跑，会很耗时）之后就需要用带有GPU显卡的电脑来跑代码**，如何安装Pytorch可以看下列链接的视频：
**[手把手教你安装PyTorch 傻瓜式操作 一次成功](https://www.bilibili.com/video/BV16H4y1c7Dx/?share_source=copy_web&vd_source=934dbcf707dd23affd7abb1463938dc1)**
在按照这个视频安装GPU的Pytorch时要注意，中间有一步是创建一个anaconda虚拟解释器环境，**一定要将python版本设置为3.8**，这是因为李沐老师团队开发的库d2l只能在3.8-3.10版本的python运行，3.8有网友说最为稳定

#  郑重声明：本项目中的word只是相当于个人学习笔记，最好是看李沐老师的课然后如果感觉不懂的地方希望我的笔记会对你有帮助。
