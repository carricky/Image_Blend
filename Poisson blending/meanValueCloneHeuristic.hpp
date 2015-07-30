#include <iostream>
#include <vector>
#include <assert.h>
#include <ctime>
#include <algorithm>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "math.hpp"

#define ATF at<float>
#define AT3F at<Vec3f>
#define ATB at<uchar>
#define AT3B at<Vec3b>

typedef pair<std::vector<double>, std::vector<int> > Mvc;
Vector<Vector<Vector<double>>> coordinate;

using namespace std;
using namespace cv;


Point									start;
unsigned int							boundaryLevels;
vector<Mvc>					        	meanValues;
vector<Point>				 		    boundary;
vector<Point>*						    interior;
vector<pair<double, double> >			thresholds;

double boundaryWeight(Point pt, int localIndex, vector<unsigned int>& refBoundary)
{
	Point current = (boundary)[refBoundary[localIndex]];
	Point previous;
	Point next;

	if (localIndex == 0)
		previous = (boundary)[refBoundary[refBoundary.size() - 1]];
	else
		previous = (boundary)[refBoundary[localIndex - 1]];

	if (localIndex == refBoundary.size() - 1)
		next = (boundary)[refBoundary[0]];
	else
		next = (boundary)[refBoundary[localIndex + 1]];

	double distance = dist(pt, current);

	if (distance == 0)
		return 1;
	double weight = ((tan(angle(pt, previous, current) / 2) + tan(angle(pt, current, next) / 2)) / distance);
	return weight;
}

bool refinedEnough(Point pt, Point vertex, Point next, Point prev, int level)
{
	return (dist(pt, vertex)			> thresholds[level].first
		&& angle(pt, prev, vertex)	< thresholds[level].second
		&& angle(pt, next, vertex)	< thresholds[level].second);
}

void computeBoundaryLevels()
{
	boundaryLevels = 0;
	int boundarySize = boundary.size();
	while (boundarySize > 16)
	{
		++boundaryLevels;
		boundarySize = (int)(ceil(double(boundarySize) / 2.0));
	}

}

void computeThresholdLevels()
{
	thresholds.clear();
	for (unsigned int i = 0; i < boundaryLevels + 1; ++i)
		thresholds.push_back(pair<double, double>(double(boundary.size()) / double(16 * pow(2.5, int(i))),
		.75 * pow(.8, int(i))));
}

int wrap(int index)
{
	if (index >= int(boundary.size()))
		return index - boundary.size();
	else if (index < 0)
		return index + boundary.size();
	else
		return index;
}

void computeRefinedBoundary(Point pt, int index, int step, int level, vector<unsigned int>& refBoundary)
{
	if (step == 1)
		refBoundary.push_back(wrap(index));

	else if ((refinedEnough(pt, (boundary)[wrap(index)], (boundary)[wrap(index + step)], (boundary)[wrap(index - step)], level)) && 
		(find(refBoundary.begin(), refBoundary.end(), wrap(index)) == refBoundary.end()))
		refBoundary.push_back(wrap(index));
	else
	{
		step /= 2;
		computeRefinedBoundary(pt, index - step, step, level + 1, refBoundary);
		computeRefinedBoundary(pt, index, step, level + 1, refBoundary);
		//computeRefinedBoundary(pt, index + step, step, level + 1, refBoundary);
	}
}

vector<unsigned int> refinedBoundary(Point pt)
{
	vector<unsigned int> refBoundary;
	int step = (int)(pow(double(2), double(boundaryLevels)));
	
	for (unsigned int i = 0; i < boundary.size(); i += step)
		computeRefinedBoundary(pt, i, step, 0, refBoundary);

	return refBoundary;
}

Mvc meanValueCoordinates(Point pt)
{
	Mvc values;
	double total = 0;
	vector<unsigned int> refBoundary = refinedBoundary(pt);
	//cout << pt << ","<<refBoundary.size()<<":";
	int idx = 0;
	for (vector<unsigned int>::iterator i = refBoundary.begin(); i != refBoundary.end(); ++i)
	{
		assert(*i < boundary.size());
		double weight = boundaryWeight(pt, idx, refBoundary);
		total += weight;
		values.first.push_back(weight);
		values.second.push_back(*i);
		//cout << *i << ",";
		idx++;
	}
	//cout << "end" << endl;
	assert(values.first.size() == values.second.size());

	if (total == 0)
		total = 1;

	for (unsigned int i = 0; i < values.first.size(); ++i)
		values.first[i] = values.first[i] / total;

	return values;
}

void getBoundary(Mat mask, vector<Point> &boundary)
{
	int length = 2 * mask.cols + 2 * mask.rows - 4;
	boundary.reserve(length);
//#pragma omp parallel for
	for (int i = 0; i < length; ++i)
	{
		if (i <= mask.rows - 1)
			boundary.push_back(Point(0, i));
		else if (i <= mask.rows - 1 + mask.cols - 1)
			boundary.push_back(Point(i - (mask.rows - 1), mask.rows - 1));
		else if (i <= mask.rows - 1 + mask.cols - 1 + mask.rows - 1)
			boundary.push_back(Point(mask.cols - 1, mask.rows - 1 - (i - (mask.rows - 1 + mask.cols - 1))));
		else boundary.push_back(Point(length - i, 0));
		//cout << boundary.back() << ";";

	}
}

void preCompute(Mat &mask)
{
	computeBoundaryLevels();
	computeThresholdLevels();
	meanValues.reserve((mask.rows - 2)*(mask.cols - 2));

	for (int i = 1; i < mask.rows - 1; ++i)
	for (int j = 1; j < mask.cols - 1; ++j)
		meanValues.push_back(meanValueCoordinates(Point(j, i)));

}

Mat MVC_clone_hierarchic(cv::Mat &I, cv::Mat &mask, int posx, int posy)
{
	

	//compute the differences along the boundary
	double *diff = new double[boundary.size()];
#pragma omp parallel for 
	for (int i = 0; i < (int)boundary.size(); ++i)
	{
		diff[i] = I.ATF(boundary[i].y + posy, boundary[i].x + posx) - mask.ATF(boundary[i].y, boundary[i].x);
	}

	Mat diff_mat(mask.size(), CV_8UC1);
	int idx = 0;
	//clock_t tic = clock();
#pragma omp parallel for 
	for (int i = 1; i < mask.rows - 1; ++i)
	for (int j = 1; j < mask.cols - 1; ++j)
	{
		float sum = 0;
		int idx = (i - 1)*(mask.cols - 2) + j - 1;
		for (int k = 0; k < (int)meanValues[idx].first.size(); k ++ )
		{
			sum += (float)(meanValues[idx].first[k] * diff[meanValues[idx].second[k]]);
		}
		
		I.ATF(i + posy, j + posx) = mask.ATF(i, j) + sum;

		//idx++;
	}

	//clock_t toc = clock();

	//printf("Execution time: %f seconds\n", (double)(toc - tic) / CLOCKS_PER_SEC);

	I.convertTo(I, CV_8UC3);
	return I;
	
	
}

Mat MVC_ini_hierarchic(cv::Mat &I, cv::Mat &mask, int posx, int posy)
{

	if (I.channels() < 3)
	{
		cout << "Input the image with 3 channels!" << endl;
		cvWaitKey();
	}
	vector<cv::Mat> bgr_I;
	vector<cv::Mat> bgr_mask;
	split(I, bgr_I);
	split(mask, bgr_mask);
	cv::Mat resb(I.size(), CV_8UC1);
	cv::Mat resg(I.size(), CV_8UC1);
	cv::Mat resr(I.size(), CV_8UC1);
	clock_t tic = clock();

	resb = MVC_clone_hierarchic(bgr_I[0], bgr_mask[0], posx, posy);
	//printf("Finish b channel!\n");
	resg = MVC_clone_hierarchic(bgr_I[1], bgr_mask[1], posx, posy);
	//printf("Finish g channel!\n");
	resr = MVC_clone_hierarchic(bgr_I[2], bgr_mask[2], posx, posy);
	//printf("Finish r channel!\n");
	clock_t toc = clock();

	printf("Execution time: %f seconds\n", (double)(toc - tic) / CLOCKS_PER_SEC);
	cv::Mat result(I.size(), CV_8UC3);
	for (int i = 0; i<I.rows; i++)
	for (int j = 0; j<I.cols; j++)
	{
		result.AT3B(i, j)[0] = resb.ATB(i, j);
		result.AT3B(i, j)[1] = resg.ATB(i, j);
		result.AT3B(i, j)[2] = resr.ATB(i, j);
	}

	return result;
}





