# cartoon
opencv3.2 照片动漫化
风景图片的动漫化
算法详细介绍http://blog.csdn.net/zhangpengzp/article/details/77149199
主要分四步：
1、边缘检测 
2、将边缘检测得到的边缘 以黑色的形式贴在原来的画上。 
3、对贴了边缘的图进行双边滤波，双边滤波可以较好的滤波的同时保留边缘。 
4、修改图像的颜色的饱和度，本文采用的是将RGB转化为HSI空间，然后调整S分量


有贴边缘的效果图：


![image](https://github.com/hurtnotbad/cartoon/blob/master/%E8%B4%B4%E8%BE%B9%E7%BC%98%E6%95%88%E6%9E%9C%E5%9B%BE.jpg)



没有贴的效果

![image](https://github.com/hurtnotbad/cartoon/blob/master/%E6%9C%AA%E8%B4%B4%E8%BE%B9%E7%BC%98%E6%95%88%E6%9E%9C%E5%9B%BE.jpg)
