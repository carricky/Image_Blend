/****************************************************************************
** Image editing tool, based on "Poisson Image Editing" and "Mean-Value Coordinates".
**
** For more details see "Poisson Image Editing" 
** by Patrick P´erez Michel Gangnet Andrew Blake 2003 
** "Coordinates for Instant Image Cloning" 
** by Farbman, Hoffer, Cohen-Or, Lipman and Lischinski 2009
** Developed by Siyuan Gao in Zhejiang University
****************************************************************************/

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "poissonEditing.hpp"
#include "meanValueCloneHeuristic.hpp"
#include "meanValueCloneNaive.hpp"

using namespace std;
using namespace cv;

IplImage		*img0, *img1, *img2, *subimg, *result;
CvPoint			point;
int				drag = 0;
int				destx, desty;
int*			src_face = new int[6];
int*			dst_face = new int[6];
int				src_num, dst_num;
double			widScale = 1, heiScale = 1;
int				mode = 0, face_mode = 0;
char			src[50];
char			dest[50];

void mouseHandler(int event, int x, int y, int flags, void* param) 
{

	if ((flags & CV_EVENT_FLAG_CTRLKEY) && (event == CV_EVENT_LBUTTONDOWN))
	{
		if (src_num == 0) img1 = cvCloneImage(img0);
		if (src_num < 3)
		{
			src_face[src_num * 2] = x;
			src_face[src_num * 2 + 1] = y;
		}
		src_num++;
		cvCircle(img1, cv::Point(x, y), 1, Scalar(0, 255, 0), 5, 8, 0);
		cvShowImage("Source", img1);
	}
	else
	{
		if (event == CV_EVENT_LBUTTONDOWN && !drag)
		{
			point = cvPoint(x, y);
			drag = 1;
		}

		if (event == CV_EVENT_MOUSEMOVE && drag)
		{
			img1 = cvCloneImage(img0);

			cvRectangle(img1, point, cvPoint(x, y), CV_RGB(0, 255, 0), 1, 8, 0);

			cvShowImage("Source", img1);
		}

		if (event == CV_EVENT_LBUTTONUP && drag)
		{
			img1 = cvCloneImage(img0);

			cvSetImageROI(img1, cvRect(point.x, point.y, x - point.x, y - point.y));

			subimg = cvCreateImage(cvGetSize(img1), img1->depth, img1->nChannels);

			cvCopy(img1, subimg, NULL);
			Mat subimg_mat(subimg);

			//vector<Point_Del> inner;
			//getInnerpoints(subimg_mat, inner);
			if (mode == 2)
			{
				clock_t tic = clock();
				preCompute_naive(subimg_mat, coordinate);
				clock_t toc = clock();
				printf("Naive MVC Pre-compute finished!\n");
				cout << "preComputeTime:" << (double)(toc - tic) / CLOCKS_PER_SEC << endl;;
				
				clock_t tic2 = clock();

				getBoundary(subimg_mat, boundary);
				preCompute(subimg_mat);
				printf("Heuristic MVC Pre-compute finished!\n");
				clock_t toc2 = clock();
				cout << "preComputeTime:" << (double)(toc2 - tic2) / CLOCKS_PER_SEC << endl;;
			}
			if (mode == 3)
			{

				clock_t tic = clock();
				getBoundary(subimg_mat, boundary);
				preCompute(subimg_mat);
				printf("preCompute finished!\n");
				clock_t toc = clock();
				cout << "preComputeTime:" << (double)(toc - tic) / CLOCKS_PER_SEC;
			}
			cvNamedWindow("ROI", 1);
			cvShowImage("ROI", subimg);
			cvWaitKey(0);
			cvDestroyWindow("ROI");
			cvResetImageROI(img1);
			cvShowImage("Source", img1);
			drag = 0;
		}

		if (event == CV_EVENT_RBUTTONUP)
		{
			cvShowImage("Source", img0);
			src_face = new int[6];
			src_num = 0;
			drag = 0;
		}
	}

}

void mouseHandler_ill(int event, int x, int y, int flags, void* param)
{
	if (event == CV_EVENT_LBUTTONDOWN && !drag)
	{
		point = cvPoint(x, y);
		destx = x;
		desty = y;
		drag = 1;
	}

	if (event == CV_EVENT_MOUSEMOVE && drag)
	{
		img1 = cvCloneImage(img0);

		cvRectangle(img1, point, cvPoint(x, y), CV_RGB(0, 255, 0), 1, 8, 0);

		cvShowImage("Source", img1);
	}

	if (event == CV_EVENT_LBUTTONUP && drag)
	{
		img1 = cvCloneImage(img0);

		cvSetImageROI(img1, cvRect(point.x, point.y, x - point.x, y - point.y));

		subimg = cvCreateImage(cvGetSize(img1), img1->depth, img1->nChannels);

		cvCopy(img1, subimg, NULL);
		Mat subimg_mat(subimg);
		cvNamedWindow("ROI", 1);
		cvShowImage("ROI", subimg);
		cvWaitKey(0);
		cvDestroyWindow("ROI");
		cvResetImageROI(img1);
		cvShowImage("Source", img1);
		drag = 0;
	}

	if (event == CV_EVENT_RBUTTONUP)
	{
		Mat result_mat;
		Mat dst_mat = Mat(img0);
		Mat src_mat = Mat(subimg);
		dst_mat.convertTo(dst_mat, CV_32FC3);
		src_mat.convertTo(src_mat, CV_32FC3);
		if (mode == 4)
		{
			double beta;
			//cout << "Please input alpha(0.2 recommended)" << endl;
			//cin >> alpha;
			cout << "Please input beta(between 0~1 recommended, the bigger beta is, the stronger the effect will be)" << endl;
			cin >> beta;
			result_mat = local_Ill(dst_mat, src_mat, destx, desty, 0.2, beta);
			IplImage result_pro = result_mat;
			result = &result_pro;
			cvShowImage("result", result);
			cvSaveImage("Output.jpg", &result_pro);
		}
		else if (mode == 5)
		{
			Mat dst = Mat(img0);
			dst.convertTo(dst, CV_32FC3);
			Mat result_mat = texture_flat(dst, destx, desty, src_mat.cols, src_mat. rows);
			IplImage result_pro = result_mat;
			result = &result_pro;
			cv::imshow("result", result_mat);

			cvSaveImage("Output.jpg", &result_pro);
		}
		src_face = new int[6];
		src_num = 0;
		drag = 0;
	}
}

void mouseHandler1_face(int event, int x, int y, int flags, void* param) {
	IplImage *im1;

	im1 = cvCloneImage(img2);

	if ((flags & CV_EVENT_FLAG_CTRLKEY) && (event == CV_EVENT_LBUTTONDOWN))
	{
		if (dst_num == 0) img1 = cvCloneImage(img2);
		if (dst_num < 3)
		{
			dst_face[dst_num * 2] = x;
			dst_face[dst_num * 2 + 1] = y;
		}
		dst_num++;
		cvCircle(img1, cv::Point(x, y), 1, Scalar(0, 0, 255), 5, 8, 0);
		cvShowImage("Destination", img1);
	}
	else
	{

		if (event == CV_EVENT_LBUTTONDOWN)
		{
			double srcWid = dist(Point(src_face[0], src_face[1]), Point(src_face[2], src_face[3]));
			double dstWid = dist(Point(dst_face[0], dst_face[1]), Point(dst_face[2], dst_face[3]));
			widScale = dstWid / srcWid;

			double srcHei = dist(Point(src_face[0], src_face[1]), Point(src_face[4], src_face[5]));
			double dstHei = dist(Point(dst_face[0], dst_face[1]), Point(dst_face[4], dst_face[5]));
			double srcHei2 = dist(Point(src_face[2], src_face[3]), Point(src_face[4], src_face[5]));
			double dstHei2 = dist(Point(dst_face[2], dst_face[3]), Point(dst_face[4], dst_face[5]));
			heiScale = dstHei / srcHei;
			double heiScale2 = dstHei2 / srcHei2;
			heiScale = (heiScale + heiScale2) / 2;

			point = cvPoint(x, y);

			cvRectangle(im1, cvPoint(x, y), cvPoint(x + (int)(subimg->width*widScale), y + (int)(subimg->height*heiScale)), CV_RGB(255, 0, 0), 1, 8, 0);

			destx = x;
			desty = y;

			cvShowImage("Destination", im1);
		}
		if (event == CV_EVENT_RBUTTONUP)
		{
			
			if (destx + subimg->width*widScale > img2->width || desty + subimg->height*heiScale > img2->height)
			{
				cout << "Index out of range" << endl;
				exit(0);
			}


			Mat I(img2);
			Mat src(subimg);
			I.convertTo(I, CV_32FC3);
			src.convertTo(src, CV_32FC3);
			resize(src, src, Size((int)(src.cols*widScale), (int)(src.rows*heiScale)));
			////////// blend ///////////////
			Mat result_mat;
			if (mode == 0)
				result_mat = poisson_blend(I, src, destx, desty);
			else if (mode == 1)
				result_mat = mixed_poisson_blend(I, src, destx, desty);
			else if (mode == 2)
				result_mat = MVC_ini_naive(I, src, destx, desty, coordinate);
			else if (mode == 3)
				result_mat = MVC_ini_hierarchic(I, src, destx, desty);
			//Mat result_mat = MVC_ini(I, src, destx, desty);
			IplImage result_pro = result_mat;
			result = &result_pro;

			////////// save blended result ////////////////////

			cvSaveImage("Output.jpg", result);
			cvSaveImage("cutpaste.jpg", im1);

			cvNamedWindow("Image cloned", 1);
			cvShowImage("Image cloned", result);
			cvWaitKey(0);
			cvDestroyWindow("Image cloned");
		}
	}
}

void mouseHandler1(int event, int x, int y, int flags, void* param)
{
	IplImage *im1;

	im1 = cvCloneImage(img2);

	if (event == CV_EVENT_LBUTTONDOWN)
	{
		point = cvPoint(x, y);
		cvRectangle(im1, cvPoint(x, y), cvPoint(x + subimg->width, y + subimg->height), CV_RGB(255, 0, 0), 1, 8, 0);
		destx = x;
		desty = y;
		cvShowImage("Destination", im1);
	}
	if (event == CV_EVENT_RBUTTONUP)
	{

		if (destx + subimg->width > img2->width || desty + subimg->height > img2->height)
		{
			cout << "Index out of range" << endl;
			exit(0);
		}

		Mat I(img2);
		Mat src(subimg);
		I.convertTo(I, CV_32FC3);
		src.convertTo(src, CV_32FC3);
		////////// blend ///////////////
		Mat result_mat;
		if (mode == 0)
			result_mat = poisson_blend(I, src, destx, desty);
		else if (mode == 1)
			result_mat = mixed_poisson_blend(I, src, destx, desty);
		else if (mode == 2)
			result_mat = MVC_ini_naive(I, src, destx, desty, coordinate);
		else if (mode == 3)
			result_mat = MVC_ini_hierarchic(I, src, destx, desty);
		cout << "solved index " << src.cols*src.rows << endl;
		//Mat result_mat = MVC_ini(I, src, destx, desty);
		IplImage result_pro = result_mat;
		result = &result_pro;

		////////// save blended result ////////////////////

		cvSaveImage("Output.jpg", result);
		cvSaveImage("cutpaste.jpg", im1);

		cvNamedWindow("Image cloned", 1);
		cvShowImage("Image cloned", result);
		cvWaitKey(0);
		cvDestroyWindow("Image cloned");
	}
	
}

void onChangeTrackBar(int poi, void* usrdata)
{
	*(int *)usrdata = poi;
	destroyWindow("message");
	char mode_str1[100];
	char mode_str2[100] ;
	strcpy_s(mode_str1, "You have chosen the ");
	if (poi == 0)
	{
		cvNamedWindow("Destination", 1);
		cvShowImage("Destination", img2);
		cvSetMouseCallback("Destination", face_mode == 0 ? mouseHandler1 : mouseHandler1_face, NULL);

		strcpy_s(mode_str2, "Poisson Blend mode ");
	}
	if (poi == 1) 
	{
		cvNamedWindow("Destination", 1);
		cvShowImage("Destination", img2);
		cvSetMouseCallback("Destination", face_mode == 0 ? mouseHandler1 : mouseHandler1_face, NULL);

		strcpy_s(mode_str2, "Mixed Poisson Blend mode "); 
	}
	if (poi == 2) 
	{
		cvNamedWindow("Destination", 1);
		cvShowImage("Destination", img2);
		cvSetMouseCallback("Destination", face_mode == 0 ? mouseHandler1 : mouseHandler1_face, NULL); 

		strcpy_s(mode_str2, "Naive Mean Value Coordinate Blend mode ");
	}
	if (poi == 3)
	{
		cvNamedWindow("Destination", 1);
		cvShowImage("Destination", img2);
		cvSetMouseCallback("Destination", face_mode == 0 ? mouseHandler1 : mouseHandler1_face, NULL);

		strcpy_s(mode_str2, "Heuristic Mean Value Coordinate Blend mode ");
	}
	if (poi == 4)
	{
		destroyWindow("Destination");
		cvSetMouseCallback("Source", NULL, NULL);
		cvSetMouseCallback("Source", mouseHandler_ill, NULL);
		strcpy_s(mode_str2, "Local Illumination Change mode ");
	}

	if (poi == 5)
	{
		destroyWindow("Destination");
		cvSetMouseCallback("Source", NULL, NULL);
		cvSetMouseCallback("Source", mouseHandler_ill, NULL);
		strcpy_s(mode_str2, "Texture Flattening mode ");
	}
	strcat_s(mode_str1, mode_str2);

	if ((face_mode == 1)&&(poi!=4)&&(poi!=5)) strcat_s(mode_str1, "with face recognition");
	else if ((poi != 4) && (poi != 5)) strcat_s(mode_str1, "without face recognition");
	Mat dis = Mat::ones(50,800,CV_8UC1)*255;
	putText(dis,  mode_str1, Point(25, 25), FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0));
	imshow("message", dis);
	moveWindow("message", 100, 0);
}

void onChangeTrackBar2(int poi, void* usrdata)
{
	*(int *)usrdata = poi;
	destroyWindow("message");
	char mode_str1[100];
	char mode_str2[100];
	strcpy_s(mode_str1, "You have chosen the ");
	if (mode == 0) strcpy_s(mode_str2, "Poisson Blend mode ");
	if (mode == 1) strcpy_s(mode_str2, "Mixed Poisson Blend mode ");
	if (mode == 2) strcpy_s(mode_str2, "Naive Mean Value Coordinate Blend mode ");
	if (mode == 3) strcpy_s(mode_str2, "Heuristic Mean Value Coordinate Blend mode ");
	if (mode == 4) strcpy_s(mode_str2, "Local Illumination Change mode ");
	if (mode == 5) strcpy_s(mode_str2, "Texture Flattening mode ");
	strcat_s(mode_str1, mode_str2);

	if ((poi == 1)&&(mode!=4)&&(mode!=5)) strcat_s(mode_str1, "with face recognition");
	else  if ((mode != 4) && (mode != 5)) strcat_s(mode_str1, "without face recognition");

	Mat dis = Mat::ones(50, 800, CV_8UC1) * 255;
	putText(dis, mode_str1, Point(25, 25), FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0));
	imshow("message", dis);
	moveWindow("message", 100, 0);
	
	if ((mode != 4) && (mode != 5))
	{
		cvSetMouseCallback("Destination", NULL, NULL);
		cvSetMouseCallback("Destination", face_mode == 0 ? mouseHandler1 : mouseHandler1_face, NULL);
	}
}

void checkfile(char *file)
{
	while (1)
	{
		printf("Enter %s Image: ", file);
		if (!strcmp(file, "Source"))
		{
			cin >> src;
			break;
		}
		else if (!strcmp(file, "Destination"))
		{
			cin >> dest;
			break;
		}
	}
}

int main(int argc, char** argv) {

	src_num = 0;
	dst_num = 0;

	cout << " Poisson Image Editing" << endl;
	cout << "-----------------------" << endl;
	cout << "Options: " << endl;
	cout << endl;
	cout << "1) Poisson Blending " << endl;
	cout << "2) Mixed Poisson Blending " << endl;
	cout << "3) Naive MVC Clone" << endl;
	cout << "4) Heuristic MVC Clone" << endl;
	cout << "5) Local Illumination Change " << endl;
	cout << "6) Texture Flattening " << endl;

	cout << endl;

	char s[] = "Source";
	char d[] = "Destination";
	checkfile(s);
	checkfile(d);

	img0 = cvLoadImage(src);

	img2 = cvLoadImage(dest);

	//////////// source image ///////////////////

	cvNamedWindow("Source", 1);
	cvSetMouseCallback("Source", mouseHandler, NULL);
	cvShowImage("Source", img0);
	createTrackbar("mode", "Source", 0, 5, onChangeTrackBar, &mode);
	createTrackbar("face mode?", "Source", 0, 1, onChangeTrackBar2, &face_mode);
	
	
	/////////// destination image ///////////////

	cvNamedWindow("Destination", 1);
	
	cvShowImage("Destination", img2);
	cvSetMouseCallback("Destination", face_mode == 0 ? mouseHandler1 : mouseHandler1_face, NULL);

	cvWaitKey(0);

	cvDestroyWindow("Source");
	cvDestroyWindow("Destination");

	cvReleaseImage(&img0);
	cvReleaseImage(&img1);
	cvReleaseImage(&img2);

	return 0;
}