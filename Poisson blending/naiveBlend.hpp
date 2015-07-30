#include <ctime>
#include <cmath>
#include <math.h>
#include <iostream>
#include <vector>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
using namespace cv;
using namespace std;

#define pi 3.1416
#define ATF at<float>
#define AT3F at<Vec3f>
#define ATB at<uchar>
#define AT3B at<Vec3b>

void interpolant(Mat &I, Mat &mask, Mat &diff_mat, int step, int posx, int posy)
{
	double diff_temp1, diff_temp2 = 0;
	for (int i = step; i < (mask.rows - step); i++)
	{
		for (int j = step; j < (mask.cols - step); j++)
		{
			if (j == step)
				diff_mat.ATF(i, j) = 1.0 / 3 * diff_mat.ATF(i - 1, j) + 1.0 / 3 * diff_mat.ATF(i - 1, j - 1) + 1.0 / 3 * diff_mat.ATF(i, j - 1);
			else if (i == mask.rows - step - 1)
				diff_mat.ATF(i, j) = 1.0 / 3 * diff_mat.ATF(i + 1, j) + 1.0 / 3 * diff_mat.ATF(i + 1, j - 1) + 1.0 / 3 * diff_mat.ATF(i, j - 1);
			else if (j == mask.cols - step - 1)
				diff_mat.ATF(i, j) = 1.0 / 3 * diff_mat.ATF(i - 1, j + 1) + 1.0 / 3 * diff_mat.ATF(i - 1, j) + 1.0 / 3 * diff_mat.ATF(i, j + 1);
			else if (i == step)
				diff_mat.ATF(i, j) = 1.0 / 3 * diff_mat.ATF(i - 1, j) + 1.0 / 3 * diff_mat.ATF(i - 1, j - 1) + 1.0 / 3 * diff_mat.ATF(i, j - 1);
			diff_temp1 = diff_mat.ATF(i, j);
		}
	}
}

Mat naiveBlend(Mat &I, Mat &mask, int posx, int posy)
{

	Mat diff_mat(mask.size(), CV_32FC1);
	double diff_temp1, diff_temp2 = 30;
	for (int i = 0; i < mask.rows; ++i)
	for (int j = 0; j < mask.cols; ++j)
	{
		if ((i == 0) || (i == mask.rows - 1) || (j == 0) || (j == mask.cols - 1))
		{
			diff_temp1 = I.ATF(i + posy, j + posx) - mask.ATF(i, j);
			diff_mat.ATF(i, j) = diff_temp1;
			//diff_temp2 = I.ATF(i + posy, j + posx) - mask.ATF(i, j);
		}
	}

	for (int i = 1; i < mask.rows / 2 + 1; i++)
		interpolant(I, mask, diff_mat, i, posx, posy);

	for (int i = 1; i < (mask.rows - 1); i++)
	{
		for (int j = 1; j < (mask.cols - 1); j++)
		{
			diff_mat.ATF(i, j) = 1.0 / 8 * diff_mat.ATF(i - 1, j) + 1.0 / 8 * diff_mat.ATF(i, j - 1) + 1.0 / 8 * diff_mat.ATF(i + 1, j) + 1.0 / 8 * diff_mat.ATF(i, j + 1)
				+ 1.0 / 8 * diff_mat.ATF(i - 1, j - 1) + 1.0 / 8 * diff_mat.ATF(i - 1, j + 1) + 1.0 / 8 * diff_mat.ATF(i + 1, j - 1) + 1.0 / 8 * diff_mat.ATF(i + 1, j + 1);
			//diff_mat.ATF(i, j) = 1.0 / 5 * diff_mat.ATF(i - 1, j) + 1.0 / 5 * diff_mat.ATF(i - 1, j - 1) + 1.0 / 5 * diff_mat.ATF(i + 1, j) + 1.0 / 5 * diff_mat.ATF(i + 1, j + 1) + 1.0 / 5 * diff_mat.ATF(i, j);
			I.ATF(i + posy, j + posx) = mask.ATF(i, j) + diff_mat.ATF(i, j);
		}
	}


	I.convertTo(I, CV_8UC1);
	return I;
}

Mat naive_ini(Mat &I, Mat &mask, int posx, int posy)
{

	vector<Mat> bgr_I;
	vector<Mat> bgr_mask;
	split(I, bgr_I);
	split(mask, bgr_mask);
	Mat resb(I.size(), CV_8UC1);
	Mat resg(I.size(), CV_8UC1);
	Mat resr(I.size(), CV_8UC1);
	clock_t tic = clock();

	/*preCompute(bgr_I[0], bgr_mask[0], posx, posy, coordinate);*/
	resb = naiveBlend(bgr_I[0], bgr_mask[0], posx, posy);
	printf("Finish b channel!\n");
	resg = naiveBlend(bgr_I[1], bgr_mask[1], posx, posy);
	printf("Finish g channel!\n");
	resr = naiveBlend(bgr_I[2], bgr_mask[2], posx, posy);
	printf("Finish r channel!\n");
	clock_t toc = clock();

	printf("Execution time: %f seconds\n", (double)(toc - tic) / CLOCKS_PER_SEC);
	Mat result(I.size(), CV_8UC3);
	for (int i = 0; i<I.rows; i++)
	for (int j = 0; j<I.cols; j++)
	{
		result.AT3B(i, j)[0] = resb.ATB(i, j);
		result.AT3B(i, j)[1] = resg.ATB(i, j);
		result.AT3B(i, j)[2] = resr.ATB(i, j);
	}
	return result;
}