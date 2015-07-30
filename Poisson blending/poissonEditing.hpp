#include <ctime>
#include <cmath>
#include <math.h>
#include <iostream>
#include <vector>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "math.hpp"

using namespace cv;
using namespace std;

#define pi 3.1416
#define ATF at<float>
#define AT3F at<Vec3f>
#define ATB at<uchar>
#define AT3B at<Vec3b>

void poisson_solver(Mat &img, Mat &gxx, Mat &gyy, Mat &result){
	
	int w = img.cols;
	int h = img.rows;
	int channel = img.channels();

	unsigned long int idx;

	Mat lap = gxx + gyy;

	Mat bound(img);
	bound(Rect(1, 1, w - 2, h - 2)) = 0;
	double *dir_boundary = new double[h*w];

//#pragma omp parallel for
	for (int i = 1; i < h - 1; i++)
		for (int j = 1; j < w - 1; j++)
		{	
			unsigned long int idx = i*w + j;
			if ( i == 1 || i == h - 2 || j == 1 || j == w - 2 )				
				dir_boundary[idx] = (int)bound.ATF(i, (j + 1)) + (int)bound.ATF(i, (j - 1)) + (int)bound.ATF(i - 1, j) + (int)bound.ATF(i + 1, j);
			else dir_boundary[idx] = 0;
		
		}


	Mat diff(h, w, CV_32FC1);

//#pragma omp parallel for
	for (int i = 0; i<h; i++)
	{
		for (int j = 0; j<w; j++)
		{
			unsigned long int idx = i*w + j;
			diff.ATF(i, j) = (float)(lap.ATF(i, j) - dir_boundary[idx]);
		}
	}
	
	double *btemp = new double[(h - 2)*(w - 2)];
//#pragma omp parallel for
	for (int i = 0; i < h - 2; i++)
	{
		for (int j = 0; j < w - 2; j++)
		{
			unsigned long int idx = i*(w - 2) + j;
			btemp[idx] = diff.ATF(i + 1, j + 1);

		}
	}

	double *bfinal = new double[(h - 2)*(w - 2)];
	double *bfinal_t = new double[(h - 2)*(w - 2)];
	double *denom = new double[(h - 2)*(w - 2)];
	double *fres = new double[(h - 2)*(w - 2)];
	double *fres_t = new double[(h - 2)*(w - 2)];

	
	dst(btemp, bfinal, h - 2, w - 2);

	transpose(bfinal, bfinal_t, h - 2, w - 2);

	dst(bfinal_t, bfinal, w - 2, h - 2);

	transpose(bfinal, bfinal_t, w - 2, h - 2);
	
	int cx = 1;
	int cy = 1;

	for (int i = 0; i < w - 2; i++, cy++)
	{
		for (int j = 0, cx = 1; j < h - 2; j++, cx++)
		{
			idx = j*(w - 2) + i;
			denom[idx] = (float)2 * cos(pi*cy / ((double)(w - 1))) - 2 + 2 * cos(pi*cx / ((double)(h - 1))) - 2;

		}
	}

	for (idx = 0; (int)idx < (w - 2)*(h - 2); idx++)
	{
		bfinal_t[idx] = bfinal_t[idx] / denom[idx];
	}


	idst(bfinal_t, fres, h - 2, w - 2);

	transpose(fres, fres_t, h - 2, w - 2);

	idst(fres_t, fres, w - 2, h - 2);

	transpose(fres, fres_t, w - 2, h - 2);


	img.convertTo(result, CV_8UC1);

//#pragma omp parallel for
	for (int i = 0; i < h - 2; i++)
	{
		for (int j = 0; j < w - 2; j++)
		{
			unsigned long int idx = i*(w - 2) + j;
			if (fres_t[idx] < 0.0)
				result.ATB(i+1, j+1) = 0;
			else if (fres_t[idx] > 255.0)
				result.ATB(i+1, j+1) = 255;
			else
				result.ATB(i+1, j+1) = (int)fres_t[idx];
		}
	}

}

void poisson_solver_jacobi(Mat &img, Mat &gxx, Mat &gyy, Mat &result) {
	int w = img.cols;
	int h = img.rows;
	int channel = img.channels();

	unsigned long int idx;

	Mat lap = gxx + gyy;

	Mat bound(img);
	bound(Rect(1, 1, w - 2, h - 2)) = 0;
	double *dir_boundary = new double[h*w];

	for (int i = 1; i < h - 1; i++)
		for (int j = 1; j < w - 1; j++)
		{
			idx = i*w + j;
			if (i == 1 || i == h - 2 || j == 1 || j == w - 2)
				dir_boundary[idx] = (int)bound.ATF(i, (j + 1)) + (int)bound.ATF(i, (j - 1)) + (int)bound.ATF(i - 1, j) + (int)bound.ATF(i + 1, j);
			else dir_boundary[idx] = 0;

		}


	Mat diff(h, w, CV_32FC1);
	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			idx = i*w + j;
			diff.ATF(i, j) = (float)(-lap.ATF(i, j) + dir_boundary[idx]);
		}
	}

	double *gtest = new double[(h - 2)*(w - 2)];
	for (int i = 0; i < h - 2; i++)
	{
		for (int j = 0; j < w - 2; j++)
		{
			idx = i*(w - 2) + j;
			gtest[idx] = diff.ATF(i + 1, j + 1);

		}
	}
	//Iteration begins
	Mat U = Mat::zeros(img.size(), CV_32FC3);
	int k = 0;
	while (k <= 3000){
		for (int i = 1; i < h - 1; i++)
		{
			for (int j = 1; j < w - 1; j++)
			{
				U.ATF(i, j) = (float)((U.ATF(i + 1, j) + U.ATF(i, j + 1) + U.ATF(i - 1, j) + U.ATF(i, j - 1) + diff.ATF(i, j)) / 4.0);
			}
		}
		k++;
	}

	img.convertTo(result, CV_8UC1);
	for (int i = 1; i < h - 1; i++)
	{
		for (int j = 1; j < w - 1; j++)
		{
			if (U.ATF(i,j) < 0.0)
				result.ATB(i , j ) = 0;
			else if (U.ATF(i, j) > 255.0)
				result.ATB(i, j) = 255;
			else
				result.ATB(i , j ) = (int)U.ATF(i, j);
		}
	}
	



}

Mat poisson_blend(Mat &I_pre, Mat &mask, int posx_pre, int posy_pre)
{
	clock_t tic = clock();

	Mat I = I_pre(Rect(posx_pre , posy_pre , mask.cols , mask.rows ));

	int posx = 1;
	int posy = 1;

	if (I.channels() < 3)
	{
		cout << "Enter RGB image!" << endl;
		exit(0);
	}
	int w_small = mask.cols;
	int h_small = mask.rows;
	int channel = I.channels();

	//gradient of the src
	Mat gx(mask.rows, mask.cols, CV_32FC3);
	Mat gy(mask.rows, mask.cols, CV_32FC3);

	getGradientX(mask, gx);
	getGradientY(mask, gy);

	Mat gxx, gyy;
	lapx(gx, gxx);
	lapy(gy, gyy);

	vector<Mat> bgr_x, bgr_y, bgr_I;
	split(gxx, bgr_x);
	split(gyy, bgr_y);
	split(I, bgr_I);

	Mat resb(I.size(), CV_8UC1);
	Mat resg(I.size(), CV_8UC1);
	Mat resr(I.size(), CV_8UC1);

	poisson_solver(bgr_I[0], bgr_x[0], bgr_y[0], resb);
	poisson_solver(bgr_I[1], bgr_x[1], bgr_y[1], resg);
	poisson_solver(bgr_I[2], bgr_x[2], bgr_y[2], resr);

	clock_t toc = clock();
	printf("Execution time: %f seconds\n", (double)(toc - tic) / CLOCKS_PER_SEC);

	Mat result(I.size(), CV_8UC3);
	for (int i = 0; i<h_small; i++)
	for (int j = 0; j<w_small; j++)
	{
		result.AT3B(i, j)[0] = resb.ATB(i, j);
		result.AT3B(i, j)[1] = resg.ATB(i, j);
		result.AT3B(i, j)[2] = resr.ATB(i, j);
	}

	imshow("result", result);
	for (int i = posy_pre; i<posy_pre + h_small; i++)
	for (int j = posx_pre; j<posx_pre + w_small; j++)
	for (int c = 0; c < I.channels(); c++)
		I_pre.AT3F(i, j)[c] = result.AT3B(i - posy_pre , j - posx_pre )[c];

	Mat I_dis;
	I_pre.convertTo(I_dis, CV_8UC3);
	//imshow("I_dis", I_dis);


	return I_dis;

}

Mat mixed_poisson_blend(Mat &I_pre, Mat &mask, int posx_pre, int posy_pre) 
{
	clock_t tic = clock();


	Mat I = I_pre(Rect(posx_pre , posy_pre , mask.cols , mask.rows ));

	if (I.channels() < 3){
		cout << "Enter RGB image!" << endl;
		exit(0);
	}
	int channel = I.channels();

	Mat gdx(mask.rows, mask.cols, CV_32FC3);
	Mat gdy(mask.rows, mask.cols, CV_32FC3);
	Mat gsx(mask.rows, mask.cols, CV_32FC3);
	Mat gsy(mask.rows, mask.cols, CV_32FC3);

	int w_small = mask.cols;
	int h_small = mask.rows;

	getGradientX(I, gdx);
	getGradientY(I, gdy);
	getGradientX(mask, gsx);
	getGradientY(mask, gsy);

	Mat mix_gx = Mat::zeros(mask.size(), CV_32FC3);
	Mat mix_gy = Mat::zeros(mask.size(), CV_32FC3);

	//compute the gradient of the mask area
	for (int i = 0; i<h_small; i++)
	for (int j = 0; j<w_small; j++)
	for (int c = 0; c<channel; ++c)
	{
		if (abs(gsx.AT3F(i, j)[c]) > abs(gdx.AT3F(i, j)[c]))
		{
		mix_gx.AT3F(i, j)[c] = gsx.AT3F(i, j)[c];

		}
		else
		{
		mix_gx.AT3F(i, j)[c] = gdx.AT3F(i, j)[c];

		}
		if (abs(gsy.AT3F(i, j)[c]) > abs(gdy.AT3F(i, j)[c]))
		{
		mix_gy.AT3F(i, j)[c] = gsy.AT3F(i, j)[c];

		}
		else
		{
		mix_gy.AT3F(i, j)[c] = gdy.AT3F(i, j)[c];

		}

	}


	Mat gx(I.size(), CV_32FC3);
	Mat gy(I.size(), CV_32FC3);

	mix_gx.copyTo(gx);
	mix_gy.copyTo(gy);

	Mat gxx;
	lapx(gx, gxx);
	Mat gyy;
	lapy(gy, gyy);

	vector<Mat> bgr_x;
	vector<Mat> bgr_y;
	vector<Mat> bgr_I;
	split(gxx, bgr_x);
	split(gyy, bgr_y);
	split(I, bgr_I);

	Mat resb(I.size(), CV_8UC1);
	Mat resg(I.size(), CV_8UC1);
	Mat resr(I.size(), CV_8UC1);

	poisson_solver(bgr_I[0], bgr_x[0], bgr_y[0], resb);
	poisson_solver(bgr_I[1], bgr_x[1], bgr_y[1], resg);
	poisson_solver(bgr_I[2], bgr_x[2], bgr_y[2], resr);
	
	clock_t toc = clock();

	printf("Execution time: %f seconds\n", (double)(toc - tic) / CLOCKS_PER_SEC);

	Mat result(I.size(), CV_8UC3);
	for (int i = 0; i<h_small; i++)
	for (int j = 0; j<w_small; j++)
	{
		result.AT3B(i, j)[0] = resb.ATB(i, j);
		result.AT3B(i, j)[1] = resg.ATB(i, j);
		result.AT3B(i, j)[2] = resr.ATB(i, j);
	}

	imshow("result", result);
	for (int i = posy_pre; i<posy_pre + h_small; i++)
	for (int j = posx_pre; j<posx_pre + w_small; j++)
	for (int c = 0; c < I.channels(); c++)
		I_pre.AT3F(i, j)[c] = result.AT3B(i - posy_pre , j - posx_pre )[c];
	Mat I_dis;
	I_pre.convertTo(I_dis, CV_8UC3);
	//imshow("I_dis", I_dis);


	return I_dis;

}

Mat local_Ill(Mat &I_pre, Mat &mask, int posx_pre, int posy_pre, double alpha, double beta)
{
	clock_t tic = clock();

	Mat I = I_pre(Rect(posx_pre - 1, posy_pre - 1, mask.cols + 2, mask.rows + 2));

	int posx = 1;
	int posy = 1;

	if (I.channels() < 3)
	{
		cout << "Enter RGB image!" << endl;
		exit(0);
	}
	//gradient of the dst
	Mat gdx(I.rows, I.cols, CV_32FC3);
	Mat gdy(I.rows, I.cols, CV_32FC3);

	//gradient of the src
	Mat gsx(mask.rows, mask.cols, CV_32FC3);
	Mat gsy(mask.rows, mask.cols, CV_32FC3);

	int w_big = I.cols;
	int h_big = I.rows;
	int channel = I.channels();

	int w_small = mask.cols;
	int h_small = mask.rows;

	/*gdx = getGradientXp(I);
	gdy = getGradientYp(I);*/
	getGradientX(I, gdx);
	getGradientY(I, gdy);
	getGradientX(mask, gsx);
	getGradientY(mask, gsy);

	Mat mix_gx = Mat::zeros(mask.size(), CV_32FC3);
	Mat mix_gy = Mat::zeros(mask.size(), CV_32FC3);

	double sum = 0;
	for (int i = 0; i < h_small; i++)
	for (int j = 0; j < w_small; j++)
	for (int c = 0; c<channel; ++c)
	{

		sum += sqrt(pow(gsx.AT3F(i, j)[c], 2) + pow(gsy.AT3F(i, j)[c], 2));
	}
	double avg = sum / (h_small*w_small*channel);
	//compute the gradient of the mask area
	for (int ii = 0; ii<h_small; ii++)
	for (int jj = 0; jj<w_small; jj++)
	for (int c = 0; c<channel; ++c)
	{
		//double temp = pow(0.2*avg, 0.2)*pow(abs(gsx.AT3F(ii, jj)[c]), -0.2)*gsx.AT3F(ii, jj)[c];
		/*gsx.AT3F(ii, jj)[c] = 0.5*gsx.AT3F(ii, jj)[c];
		gsy.AT3F(ii, jj)[c] *= 0.5;
		cout << gsy.AT3F(ii, jj)[c] << endl;*/

		//cout << pow(0.2, 0.2)*pow(sqrt(pow(gsx.AT3F(ii, jj)[c], 2) + pow(gsy.AT3F(ii, jj)[c], 2)), -0.2)<<endl;
		if ((gsx.AT3F(ii, jj)[c] != 0) || ((gsy.AT3F(ii, jj)[c] != 0)))
		{
			gsx.AT3F(ii, jj)[c] *= (float)(pow(alpha*avg, beta)*pow(sqrt(pow(gsx.AT3F(ii, jj)[c], 2) + pow(gsy.AT3F(ii, jj)[c], 2)), -beta));
			gsy.AT3F(ii, jj)[c] *= (float)(pow(alpha*avg, beta)*pow(sqrt(pow(gsx.AT3F(ii, jj)[c], 2) + pow(gsy.AT3F(ii, jj)[c], 2)), -beta));
		}
	}
	//Mat mix_gx_show;
	//mix_gx.convertTo(mix_gx_show, CV_8UC3);
	//imshow("mix_gx", mix_gx_show);
	Mat gx(I.size(), CV_32FC3);
	Mat gy(I.size(), CV_32FC3);
	gdx.copyTo(gx);
	gdy.copyTo(gy);

	//Mat mix_gx_show;
	//gx.convertTo(mix_gx_show, CV_8UC3);
	//imshow("gx", mix_gx_show);

	//mix_gx.copyTo(gx(Rect(posx, posy, w_small, h_small)));
	//mix_gy.copyTo(gy(Rect(posx, posy, w_small, h_small)));
	gsx.copyTo(gx(Rect(posx, posy, w_small, h_small)));
	gsy.copyTo(gy(Rect(posx, posy, w_small, h_small)));

	//Mat mix_gx_show2;
	//gx.convertTo(mix_gx_show2, CV_8UC3);
	//imshow("gx_a", mix_gx_show2);

	Mat gxx;
	lapx(gx, gxx);
	Mat gyy;
	lapy(gy, gyy);

	vector<Mat> bgr_x;
	vector<Mat> bgr_y;
	vector<Mat> bgr_I;
	split(gxx, bgr_x);
	split(gyy, bgr_y);
	split(I, bgr_I);

	Mat resb(I.size(), CV_8UC1);
	Mat resg(I.size(), CV_8UC1);
	Mat resr(I.size(), CV_8UC1);

	//Mat mix_gx_show2;
	//bgr_y[0].convertTo(mix_gx_show2, CV_8UC3);
	//imshow("bgr_x[0]", mix_gx_show2);

	poisson_solver(bgr_I[0], bgr_x[0], bgr_y[0], resb);
	poisson_solver(bgr_I[1], bgr_x[1], bgr_y[1], resg);
	poisson_solver(bgr_I[2], bgr_x[2], bgr_y[2], resr);

	//cout << resb;
	//Mat mix_gx_show;
	//resb.convertTo(mix_gx_show, CV_8UC3);
	//imshow("resb", mix_gx_show);
	
	clock_t toc = clock();

	printf("Execution time: %f seconds\n", (double)(toc - tic) / CLOCKS_PER_SEC);

	Mat result(I.size(), CV_8UC3);
	for (int i = 0; i<h_big; i++)
	for (int j = 0; j<w_big; j++)
	{
		result.AT3B(i, j)[0] = resb.ATB(i, j);
		result.AT3B(i, j)[1] = resg.ATB(i, j);
		result.AT3B(i, j)[2] = resr.ATB(i, j);
	}

	imshow("result", result);
	for (int i = posy_pre; i<posy_pre + h_small; i++)
	for (int j = posx_pre; j<posx_pre + w_small; j++)
	for (int c = 0; c < I.channels(); c++)
		I_pre.AT3F(i, j)[c] = result.AT3B(i - posy_pre + 1, j - posx_pre + 1)[c];
	Mat I_dis;
	I_pre.convertTo(I_dis, CV_8UC3);
	return I_dis;

}

Mat texture_flat(Mat &I, int posx, int posy, int width, int height)
{
	clock_t tic = clock();
	Mat I_flat = I(Rect(posx, posy, width, height));


	if (I_flat.channels() < 3)
	{
		cout << "Enter RGB image!" << endl;
		exit(0);
	}
	//gradient of the dst
	Mat gdx(I_flat.rows, I_flat.cols, CV_32FC3);
	Mat gdy(I_flat.rows, I_flat.cols, CV_32FC3);

	int w_big = I_flat.cols;
	int h_big = I_flat.rows;
	int channel = I_flat.channels();



	/*gdx = getGradientXp(I);
	gdy = getGradientYp(I);*/
	getGradientX(I_flat, gdx);
	getGradientY(I_flat, gdy);

	Mat gray, edge;
	cvtColor(I_flat, gray, COLOR_BGR2GRAY);
	blur(gray, edge, Size(3, 3));
	edge.convertTo(edge, CV_8UC1);
	Canny(edge, edge, 30, 45, 3);

	double sum = 0;
	for (int i = 0; i < h_big; i++)
	for (int j = 0; j < w_big; j++)
	{
		if (edge.ATB(i, j) != 255)
		{
			gdx.AT3F(i, j)[0] = 0.0;
			gdy.AT3F(i, j)[0] = 0.0;
		}
		if (edge.ATB(i, j) != 255)
		{
			gdx.AT3F(i, j)[1] = 0.0;
			gdy.AT3F(i, j)[1] = 0.0;
		}
		if (edge.ATB(i, j) != 255)
		{
			gdx.AT3F(i, j)[2] = 0.0;
			gdy.AT3F(i, j)[2] = 0.0;
		}
	}

	Mat gx(I_flat.size(), CV_32FC3);
	Mat gy(I_flat.size(), CV_32FC3);
	gdx.copyTo(gx);
	gdy.copyTo(gy);


	Mat gxx;
	lapx(gx, gxx);
	Mat gyy;
	lapy(gy, gyy);

	vector<Mat> bgr_x;
	vector<Mat> bgr_y;
	vector<Mat> bgr_I;
	split(gxx, bgr_x);
	split(gyy, bgr_y);
	split(I_flat, bgr_I);

	Mat resb(I_flat.size(), CV_8UC1);
	Mat resg(I_flat.size(), CV_8UC1);
	Mat resr(I_flat.size(), CV_8UC1);

	poisson_solver(bgr_I[0], bgr_x[0], bgr_y[0], resb);
	poisson_solver(bgr_I[1], bgr_x[1], bgr_y[1], resg);
	poisson_solver(bgr_I[2], bgr_x[2], bgr_y[2], resr);

	clock_t toc = clock();

	printf("Execution time: %f seconds\n", (double)(toc - tic) / CLOCKS_PER_SEC);
	
	Mat result(I.size(), CV_8UC3);
	I.convertTo(result, CV_8UC3);
	
	for (int i = posy, ii=0; i<posy + h_big; ++i, ++ii)
	for (int j = posx, jj=0; j<posx+ w_big; ++j, ++jj)
	{
		result.AT3B(i, j)[0] = resb.ATB(ii, jj);
		result.AT3B(i, j)[1] = resg.ATB(ii, jj);
		result.AT3B(i, j)[2] = resr.ATB(ii, jj);
	}

	imshow("result", result);

	Mat I_dis;
	result.convertTo(I_dis, CV_8UC3);
	//imshow("I_dis", I_dis);


	return I_dis;

}





