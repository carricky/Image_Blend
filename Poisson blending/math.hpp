#ifndef _MATH_HPP
#define _MATH_HPP

#include "opencv2/core/core.hpp"
#define sqr(x)          ((x)*(x))
#define pi 3.1416
#define middle(x,a,b)    (  ((a)<(b))				\
	? ((x)<(a)) ? (a) : (((x)>(b)) ? (b) : (x))	\
	: ((x)<(b)) ? (b) : (((x)>(a)) ? (a) : (x))	\
	)

using namespace cv;
#define ATF at<float>
#define AT3F at<Vec3f>
#define ATB at<uchar>
#define AT3B at<Vec3b>

/////////////////////Vector and Point Operation/////////////////////
double dist(Point pt0, Point pt1)
{
	return sqrt(double(sqr(pt0.x - pt1.x) + sqr(pt0.y - pt1.y)));
}

double size(Point pt)
{
	return dist(pt, Point(0, 0));
}

double angle(Point origin, Point a, Point b)
{
	// Create two vectors
	Point vecToA(a.x - origin.x, a.y - origin.y);
	Point vecToB(b.x - origin.x, b.y - origin.y);

	// Find angle between them
	double dot = vecToA.x * vecToB.x + vecToA.y * vecToB.y;
	double sizeA = size(vecToA);
	double sizeB = size(vecToB);

	if (sizeA == 0 || sizeB == 0)
		return 0;

	return acos(middle(dot / (sizeA * sizeB), -1, 1));
}

static bool abs_compare(double a, double b)
{
	return (std::abs(a) < std::abs(b));
}

void deCoordinate(int idx, int rows, int cols, int &x, int &y)
{
	if (idx == 2 * rows - 2 + 2 * cols - 2) idx = 0;
	if (idx <= rows - 1)
	{
		x = 0;
		y = idx;
	}
	else if (idx <= rows - 1 + cols - 1)
	{
		x = idx - (rows - 1);
		y = rows - 1;
	}
	else if (idx <= 2 * rows - 2 + cols - 1)
	{
		x = cols - 1;
		y = (2 * rows - 2 + cols - 1) - idx;
	}
	else if (idx < 2 * rows - 2 + 2 * cols - 2)
	{
		x = cols - 1 - (idx - (2 * rows - 2 + cols - 1));
		y = 0;
	}
}


/////////////////////Matrix Operation/////////////////////

// calculate horizontal gradient, gx(i, j) = img(i+1, j) - img(i, j)
void getGradientX(Mat &img, Mat &gx)
{
	int height = img.rows;
	int width = img.cols;
	Mat cat = repeat(img, 1, 2);
	/*cat.col(width) = 0;*/
	Mat roimat = cat(Rect(1, 0, width, height));
	gx = roimat - img;
	gx.col(width - 1) = 0;
}

// calculate vertical gradient, gy(i, j) = img(i, j+1) - img(i, j)
void getGradientY(Mat &img, Mat &gy)
{
	int height = img.rows;
	int width = img.cols;
	Mat cat = repeat(img, 2, 1);
	/*cat.row(height) = 0;*/
	Mat roimat = cat(Rect(0, 1, width, height));
	gy = roimat - img;
	gy.row(height - 1) = 0;
}

// calculate horizontal gradient, gxx(i+1, j) = gx(i+1, j) - gx(i, j)
void lapx(Mat &gx, Mat &gxx)
{
	int height = gx.rows;
	int width = gx.cols;
	Mat cat = repeat(gx, 1, 2);
	/*cat.col(width - 1) = 0;*/
	Mat roi = cat(Rect(width - 1, 0, width, height));
	gxx = gx - roi;
	gxx.col(0) = 0;
}

// calculate vertical gradient, gyy(i, j+1) = gy(i, j+1) - gy(i, j)
void lapy(Mat &gy, Mat &gyy)
{
	int height = gy.rows;
	int width = gy.cols;
	Mat cat = repeat(gy, 2, 1);
	/*cat.row(height - 1) = 0;*/
	Mat roi = cat(Rect(0, height - 1, width, height));
	gyy = gy - roi;
	gyy.row(0) = 0;
}

/////////////////////Poisson Equation Solver by FFT/////////////////////
void dst(double *btest, double *bfinal, int h, int w)
{

	unsigned long int idx;

	Mat temp = Mat(2 * h + 2, 1, CV_32F);
	Mat res = Mat(h, 1, CV_32F);

	int p = 0;

	for (int i = 0; i<w; i++)
	{
		temp.ATF(0, 0) = 0.0;

		for (int j = 0, r = 1; j<h; j++, r++)
		{
			idx = j*w + i;
			temp.ATF(r, 0) = (float)btest[idx];
		}

		temp.ATF(h + 1, 0) = 0.0;

		for (int j = h - 1, r = h + 2; j >= 0; j--, r++)
		{
			idx = j*w + i;
			temp.ATF(r, 0) = (float)(-1 * btest[idx]);
		}

		Mat planes[] = { Mat_<float>(temp), Mat::zeros(temp.size(), CV_32F) };

		Mat complex1;
		merge(planes, 2, complex1);

		dft(complex1, complex1, 0, 0);

		Mat planes1[] = { Mat::zeros(complex1.size(), CV_32F), Mat::zeros(complex1.size(), CV_32F) };

		split(complex1, planes1);

		std::complex<double> two_i = std::sqrt(std::complex<double>(-1));

		double fac = -2 * imag(two_i);

		for (int c = 1, z = 0; c<h + 1; c++, z++)
		{
			res.ATF(z, 0) = (float)(planes1[1].ATF(c, 0) / fac);
		}

		for (int q = 0, z = 0; q<h; q++, z++)
		{
			idx = q*w + p;
			bfinal[idx] = res.ATF(z, 0);
		}
		p++;
	}

}

void idst(double *btest, double *bfinal, int h, int w)
{
	int nn = h + 1;
	dst(btest, bfinal, h, w);
	//#pragma omp parallel for
	for (int i = 0; i<h; i++)
	for (int j = 0; j<w; j++)
	{
		unsigned long int idx = i*w + j;
		bfinal[idx] = (double)(2 * bfinal[idx]) / nn;
	}

}

void transpose(double *mat, double *mat_t, int h, int w)
{

	Mat tmp = Mat(h, w, CV_32FC1);
	int p = 0;

	//#pragma omp parallel for
	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{

			unsigned long int idx = i*(w)+j;
			tmp.ATF(i, j) = (float)(mat[idx]);
		}
	}
	Mat tmp_t = tmp.t();

	unsigned long int idx;
	for (int i = 0; i < tmp_t.size().height; i++)
	for (int j = 0; j<tmp_t.size().width; j++)
	{
		idx = i*tmp_t.size().width + j;
		mat_t[idx] = tmp_t.ATF(i, j);
	}

}

#endif