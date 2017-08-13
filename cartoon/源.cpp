#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp> 
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>
#include <iostream>

#include <math.h>

using namespace cv;
using namespace std;
void find_move_people();
IplImage *srcImage, *cannyImage;
char *grayWindow, *cannyWindow;
void canny();
void changeSImage(Mat &image, IplImage &outImg, float sRadio);
IplImage* catHSImage(CvMat* HSI_H, CvMat* HSI_S, CvMat* HSI_I);
IplImage* HSI2RGBImage(CvMat* HSI_H, CvMat* HSI_S, CvMat* HSI_I);
void pasteEdge(Mat &image, Mat &outImg, IplImage cannyImage);
void on_trackbar(int threshold)
{
	// 灰度图，canny输出图，低阈值，高阈值，灰度梯度的算子内核大小
	cvCanny(srcImage, cannyImage, threshold, threshold * 3, 3);
	cvShowImage(cannyWindow, cannyImage);
}
int main(void)
{
	char *img = "data//3.jpg";
	Mat image = imread(img);

	//canny边缘检测
	srcImage = cvLoadImage(img, CV_LOAD_IMAGE_GRAYSCALE);
	cannyImage = cvCreateImage(cvGetSize(srcImage), IPL_DEPTH_8U, 1);
	int threshold = 70;
	cvCanny(srcImage, cannyImage, threshold, threshold * 3, 3);

	cvShowImage("边缘", cannyImage);
	Mat pasteEdgeMat;
	pasteEdge(image, pasteEdgeMat, *cannyImage);
	// 显示将边缘贴在原图上
	imshow("canny贴图", pasteEdgeMat);

	// 双边滤波
	Mat binateMat;
	bilateralFilter(pasteEdgeMat, binateMat, 10, 50, 50, BORDER_DEFAULT);
	//不贴边缘直接双边滤波
	//bilateralFilter(image, binateMat, 10, 50, 50, BORDER_DEFAULT);
	imshow("原图", image);
	imshow("双边滤波", binateMat);

	//增强图像的颜色饱和度
	IplImage outImg;
	changeSImage(binateMat, outImg, 5);
	cvShowImage("hsi2", &outImg);
	waitKey(0);
}

// 将边缘检测后的图 cannyImage 边以黑色的形式贴在 image上。
void pasteEdge(Mat &image, Mat &outImg, IplImage cannyImage)
{
	Mat cannyMat;
	cannyMat = cvarrToMat(&cannyImage);
	//颜色反转
	cannyMat = cannyMat < 100;
	image.copyTo(outImg, cannyMat);


}

// 将image 像素转化到 HSI 空间，并调整S 即颜色的饱和度，
void changeSImage(Mat &image, IplImage &outImg, float sRadio)
{
	int rows = image.rows;
	int cols = image.cols;
	// 三个HSI空间数据矩阵
	CvMat* HSI_H = cvCreateMat(rows, cols, CV_32FC1);
	CvMat* HSI_S = cvCreateMat(rows, cols, CV_32FC1);
	CvMat* HSI_I = cvCreateMat(rows, cols, CV_32FC1);

	// 原始图像数据指针, HSI矩阵数据指针
	uchar* data;

	// rgb分量
	int img_r, img_g, img_b;
	int min_rgb;  // rgb分量中的最小值
	// HSI分量
	float fHue, fSaturation, fIntensity;
	int channels = image.channels();
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			data = image.ptr<uchar>(i);
			data = data + j*channels;
			img_b = *data;
			data++;
			img_g = *data;
			data++;
			img_r = *data;

			// Intensity分量[0, 1]
			fIntensity = (float)((img_b + img_g + img_r) / 3) / 255;

			// 得到RGB分量中的最小值
			float fTemp = img_r < img_g ? img_r : img_g;
			min_rgb = fTemp < img_b ? fTemp : img_b;
			// Saturation分量[0, 1]
			fSaturation = 1 - (float)(3 * min_rgb) / (img_r + img_g + img_b);

			// 计算theta角
			float numerator = (img_r - img_g + img_r - img_b) / 2;
			float denominator = sqrt(
				pow((img_r - img_g), 2) + (img_r - img_b)*(img_g - img_b));

			// 计算Hue分量
			if (denominator != 0)
			{
				float theta = acos(numerator / denominator) * 180 / 3.14;

				if (img_b <= img_g)
				{
					fHue = theta;
				}
				else
				{
					fHue = 360 - theta;
				}
			}
			else
			{
				fHue = 0;
			}

			// 赋值
			cvmSet(HSI_H, i, j, fHue);
			cvmSet(HSI_S, i, j, fSaturation * sRadio);
			cvmSet(HSI_I, i, j, fIntensity);
		}
	}
	outImg = *HSI2RGBImage(HSI_H, HSI_S, HSI_I);
	//IplImage* ip2 = HSI2RGBImage(&hsi[0], &hsi[1], &hsi[2]);

	
}


IplImage* catHSImage(CvMat* HSI_H, CvMat* HSI_S, CvMat* HSI_I)
{
	IplImage* HSI_Image = cvCreateImage(cvGetSize(HSI_H), IPL_DEPTH_8U, 3);

	for (int i = 0; i < HSI_Image->height; i++)
	{
		for (int j = 0; j < HSI_Image->width; j++)
		{
			double d = cvmGet(HSI_H, i, j);
			int b = (int)(d * 255 / 360);
			d = cvmGet(HSI_S, i, j);
			int g = (int)(d * 255);
			d = cvmGet(HSI_I, i, j);
			int r = (int)(d * 255);

			cvSet2D(HSI_Image, i, j, cvScalar(b, g, r));
		}
	}

	return HSI_Image;
}


IplImage* HSI2RGBImage(CvMat* HSI_H, CvMat* HSI_S, CvMat* HSI_I)
{
	IplImage * RGB_Image = cvCreateImage(cvGetSize(HSI_H), IPL_DEPTH_8U, 3);

	int iB, iG, iR;
	for (int i = 0; i < RGB_Image->height; i++)
	{
		for (int j = 0; j < RGB_Image->width; j++)
		{
			// 该点的色度H
			double dH = cvmGet(HSI_H, i, j);
			// 该点的色饱和度S
			double dS = cvmGet(HSI_S, i, j);
			// 该点的亮度
			double dI = cvmGet(HSI_I, i, j);

			double dTempB, dTempG, dTempR;
			// RG扇区
			if (dH < 120 && dH >= 0)
			{
				// 将H转为弧度表示
				dH = dH * 3.1415926 / 180;
				dTempB = dI * (1 - dS);
				dTempR = dI * (1 + (dS * cos(dH)) / cos(3.1415926 / 3 - dH));
				dTempG = (3 * dI - (dTempR + dTempB));
			}
			// GB扇区
			else if (dH < 240 && dH >= 120)
			{
				dH -= 120;

				// 将H转为弧度表示
				dH = dH * 3.1415926 / 180;

				dTempR = dI * (1 - dS);
				dTempG = dI * (1 + dS * cos(dH) / cos(3.1415926 / 3 - dH));
				dTempB = (3 * dI - (dTempR + dTempG));
			}
			// BR扇区
			else
			{
				dH -= 240;

				// 将H转为弧度表示
				dH = dH * 3.1415926 / 180;

				dTempG = dI * (1 - dS);
				dTempB = dI * (1 + (dS * cos(dH)) / cos(3.1415926 / 3 - dH));
				dTempR = (3 * dI - (dTempG + dTempB));
			}

			iB = dTempB * 255;
			iG = dTempG * 255;
			iR = dTempR * 255;

			cvSet2D(RGB_Image, i, j, cvScalar(iB, iG, iR));
		}
	}

	return RGB_Image;
}




void find_move_people()
{
	char *video_path = "E:\\opencv\\opencv\\sources\\samples\\data\\vtest.avi";
	VideoCapture capture;
	Mat frame, image, foreGround, backGround, fgMask;
	//	Ptr<BackgroundSubtractor> pBgmodel = createBackgroundSubtractorMOG2().dynamicCast<BackgroundSubtractor>();
	Ptr<BackgroundSubtractorMOG2> pBgmodel = createBackgroundSubtractorMOG2();
	pBgmodel->setVarThreshold(50);
	capture.open(video_path);
	if (!capture.isOpened())
	{
		cout << "open videp eror!" << endl;
	}

	while (true)
	{
		//source是原始帧
		capture >> frame;
		if (frame.empty())
			break;
		//缩小为原来四分之一，加快处理速度
		resize(frame, image, Size(frame.cols / 2, frame.rows / 2), INTER_LINEAR);

		if (foreGround.empty())
			foreGround.create(image.size(), image.type());
		//得到前景图像，是黑白灰 3种灰度值的图
		pBgmodel->apply(image, fgMask);

		// 下面是根据前景图的操作，和原图像融合得到有纹理的前景图
		GaussianBlur(fgMask, fgMask, Size(5, 5), 0);
		threshold(fgMask, fgMask, 10, 255, THRESH_BINARY);
		// 将foreGraound 所有像素置为0
		foreGround = Scalar::all(0);
		//fgMask对应点像素值为255则 foreGround像素为image里的像素，为0则直接为0
		image.copyTo(foreGround, fgMask);

		pBgmodel->getBackgroundImage(backGround);

		imshow("frame", frame);
		imshow("backGround", backGround);
		imshow("foreGround", foreGround);
		//imshow("fgMask", fgMask);

		char key = waitKey(100);
		if (key == 27)//27 对应得assic 码是27
			break;
	}

}

void canny()
{
	char *f = "E:\\opencv\\opencv\\sources\\samples\\wp8\\OpenCVXaml\\OpenCVXaml\\Assets\\Lena.png";
	grayWindow = "原图";
	cannyWindow = "边缘检测";
	srcImage = cvLoadImage(f, CV_LOAD_IMAGE_GRAYSCALE);
	cannyImage = cvCreateImage(cvGetSize(srcImage), IPL_DEPTH_8U, 1);
	cvNamedWindow(grayWindow, CV_WINDOW_AUTOSIZE);
	cvNamedWindow(cannyWindow, CV_WINDOW_AUTOSIZE);

	const char *threshold = "Threshold";
	int nThresholdEdge = 1;
	//on_trackbar 为回调函数，既当滑动条动，nThresholdEdge 改变时调用的函数
	cvCreateTrackbar(threshold, cannyWindow, &nThresholdEdge, 100, on_trackbar);
	cvShowImage(grayWindow, srcImage);
	on_trackbar(1);
}