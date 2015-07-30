#include <iostream>
#include <vector>
#include <assert.h>
#include <ctime>
#include <algorithm>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "math.hpp"

#define ATF		at<float>
#define AT3F	at<Vec3f>
#define ATB		at<uchar>
#define AT3B	at<Vec3b>

using namespace std;
using namespace cv;

void preCompute_naive(Mat &mask, Vector<Vector<Vector<double>>> &coordinate)
{
	int b_length = 2 * mask.rows - 2 + 2 * mask.cols - 2;
	//compute the MV coordinate

	coordinate.resize(mask.rows);
	for (int i = 0; i < mask.rows; ++i)
	{
		coordinate[i].resize(mask.cols);
		for (int j = 0; j < mask.cols; ++j)
		{
			coordinate[i][j].resize(b_length);
		}
	}

	double *w_temp = new double[b_length];
	double *alpha = new double[b_length];
	double distance_1, distance_2, distance_0, w_sum;
	int b_x, b_y, b_x_2, b_y_2;
	for (int i = 1; i < mask.rows - 1; ++i)
	{
		for (int j = 1; j < mask.cols - 1; ++j)
		{
			w_sum = 0;
			for (int k = 0; k < b_length; ++k)
			{

				deCoordinate(k, mask.rows, mask.cols, b_x, b_y);
				deCoordinate(k + 1, mask.rows, mask.cols, b_x_2, b_y_2);
				distance_1 = sqrt((i - b_y)*(i - b_y) + (j - b_x)*(j - b_x));
				distance_2 = sqrt((i - b_y_2)*(i - b_y_2) + (j - b_x_2)*(j - b_x_2));
				if (k <= mask.rows - 2)
					alpha[k] = abs(asin(j / distance_2) - asin(j / distance_1));
				else if (k <= mask.rows - 1 + mask.cols - 1 - 1)
					alpha[k] = abs(asin((mask.rows - 1 - i) / distance_2) - asin((mask.rows - 1 - i) / distance_1));
				else if (k <= 2 * mask.rows - 2 + mask.cols - 1 - 1)
					alpha[k] = abs(asin((mask.cols - 1 - j) / distance_2) - asin((mask.cols - 1 - j) / distance_1));
				else if (k <= 2 * mask.rows - 2 + 2 * mask.cols - 2 - 1)
					alpha[k] = abs(asin(i / distance_2) - asin(i / distance_1));

				if (k != 0)
				{
					w_temp[k] = (tan(alpha[k - 1] / 2) + tan(alpha[k] / 2)) / distance_1;
					w_sum += w_temp[k];
				}
				else
					distance_0 = distance_1;
			}
			w_temp[0] = (tan(alpha[b_length - 1] / 2) + tan(alpha[0] / 2)) / distance_0;
			w_sum += w_temp[0];
			for (int k = 0; k < b_length; ++k)
				coordinate[i][j][k] = w_temp[k] / w_sum;
		}

	}
}

Mat MVC_clone_naive(Mat &I, Mat &mask, int posx, int posy, Vector<Vector<Vector<double>>> &coordinate)
{

	int b_length = 2 * mask.rows - 2 + 2 * mask.cols - 2;

	//compute the differences along the boundary
	double *diff = new double[b_length];
	for (int i = 0, ii = posy; i < mask.rows; i++, ii++)
	for (int j = 0, jj = posx; j < mask.cols; j++, jj++)
	{
		int idx = -1;
		if (j == 0)
			idx = i;
		else if (i == mask.rows - 1) idx = mask.rows - 1 + j;
		else if (j == mask.cols - 1) idx = mask.rows - 1 + mask.cols - 1 + (mask.rows - 1 - i);
		else if (i == 0) idx = b_length - 1 - (j - 1);

		if (idx != -1)
			diff[idx] = I.ATF(ii, jj) - mask.ATF(i, j);
	}
	//evaluate the MV interpolant at x

	Mat diff_mat(mask.size(), CV_8UC1);
#pragma omp parallel for
	for (int i = 1; i < mask.rows - 1; ++i)
	for (int j = 1; j < mask.cols - 1; ++j)
	{
		double sum = 0;
		for (int k = 0; k < b_length; ++k)
		{

			sum += coordinate[i][j][k] * diff[k];
		}

		diff_mat.ATB(i, j) = (uchar)(sum);
		I.ATF(i + posy, j + posx) = (float)(mask.ATF(i, j) + sum);

	}

	//imshow("diff", diff_mat);
	I.convertTo(I, CV_8UC1);
	return I;
}

Mat MVC_ini_naive(Mat &I, Mat &mask, int posx, int posy, Vector<Vector<Vector<double>>> coordinate)
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
	resb = MVC_clone_naive(bgr_I[0], bgr_mask[0], posx, posy, coordinate);

	resg = MVC_clone_naive(bgr_I[1], bgr_mask[1], posx, posy, coordinate);

	resr = MVC_clone_naive(bgr_I[2], bgr_mask[2], posx, posy, coordinate);

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
