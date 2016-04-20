#include <cv.h>
#include <cxcore.h>
#include <highgui.h>
#include <iostream>
#include <opencv2\contrib\contrib.hpp>
using namespace std;

void main()
{	
	cv::Directory dir;

	string folderpath = "D:/";
	string extension = "*.bmp";//"*"
	bool addpath = false;//true;

	vector<string> filenames = dir.GetListFiles(folderpath, extension, addpath);

	cout<<"file names: "<<endl;
	for (int a = 0; a < filenames.size(); a++)
	{
	
		IplImage *src;
		char filepath[50];
		sprintf(filepath,"D:/%s",filenames[a].c_str());
		cout<<filepath<<endl;
		src=cvLoadImage(filepath,-1);
		
		IplImage *pImg8u=cvCreateImage(cvGetSize(src),8,1);
		cvCvtColor(src,pImg8u,CV_RGB2GRAY);           //turn to grayscale image

		IplImage * im_median_filter = cvCreateImage(cvSize(pImg8u->width,pImg8u->height),8, 1); 
		cvSmooth(pImg8u, im_median_filter, CV_MEDIAN);//median filtering
		
		int nWidth=20;
		int nHeight=40;
		
		IplImage *pImgResize=cvCreateImage(cvSize(nWidth,nHeight),8,1);
		cvResize(src,pImgResize,CV_INTER_LINEAR); //normalization
		sprintf(filepath, "E:/%d.bmp", a);
		cout<<filepath<<endl;
		cvSaveImage(filepath, pImgResize);

	}
}