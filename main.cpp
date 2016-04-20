#include "cv.h"
#include "highgui.h"       
#include <ml.h>    
#include <iostream>    
#include <fstream>    
#include <string>    
#include <vector>    
using namespace cv;    
using namespace std;    
    
    
int main(int argc, char** argv)      
{      
    vector<string> img_path;//enter file name
    vector<int> img_catg;    
    int nLine = 0;    
    string buf;    
    ifstream svm_data( "train.txt" );//This txt is a list of train set, including the file path and its label
    unsigned long n;    
    
    while( svm_data )//read the input train set
    {    
        if( getline( svm_data, buf ) )    
        {    
            nLine ++;    
            if( nLine % 2 == 0 )//this is the label
            {    
                 img_catg.push_back( atoi( buf.c_str() ) );//convert string to integer
            }    
            else    
            {    
                img_path.push_back( buf );//this is file path
            }    
        }    
    }    
    svm_data.close();//close the file
    
    CvMat *data_mat, *res_mat;    
    int nImgNum = nLine / 2; //the number of train set samples
    data_mat = cvCreateMat( nImgNum, 2916, CV_32FC1 ); //sample matrix, the second parameter shows the dimension of the matrix
    cvSetZero( data_mat );      
    res_mat = cvCreateMat( nImgNum, 1, CV_32FC1 );    //label matrix, used to store the label of each sample
    cvSetZero( res_mat );    
    
    IplImage* src;    
    
    //HOG feature
    for( string::size_type i = 0; i != img_path.size(); i++ )    
    {    
        src=cvLoadImage(img_path[i].c_str(),1);    
        if( src == NULL )    
        {    
            cout<<" can not load the image: "<<img_path[i].c_str()<<endl;    
            continue;    
        }    
    
        cout<<" processing "<<img_path[i].c_str()<<endl;    
                   
		HOGDescriptor *hog=new HOGDescriptor(cvSize(20,40),cvSize(4,8),cvSize(2,4),cvSize(2,4),9);       

        vector<float>descriptors;//the result array
		hog->compute(src, descriptors,Size(1,1), Size(0,0)); //start to compute HOG feature
        cout<<"HOG dims: "<<descriptors.size()<<endl;    
        n=0;    
        for(vector<float>::iterator iter=descriptors.begin();iter!=descriptors.end();iter++)    
        {    
            cvmSet(data_mat,i,n,*iter);//store the HOG feature
            n++;    
        }    
        cvmSet( res_mat, i, 0, img_catg[i] );    
        cout<<" end processing "<<img_path[i].c_str()<<" "<<img_catg[i]<<endl;    
    }    
        
                 
    CvSVM svm = CvSVM();//new a SVM
    CvSVMParams param;//parameters
    CvTermCriteria criteria;      
    criteria = cvTermCriteria( CV_TERMCRIT_EPS, 1000, FLT_EPSILON );      
    param = CvSVMParams( CvSVM::C_SVC, CvSVM::RBF, 10.0,pow(2,-13) , 1.0,pow(2,2), 0.5, 1.0, NULL, criteria );      
/*     
    type of SVM: CvSVM::C_SVC
    type of Kernel: CvSVM::RBF
    degree: 10.0 (not used)
    gamma: pow(2,-13)
    coef0: 1.0 (not used)
    C: pow(2,2)
    nu: 0.5(not used)
    p: 0.1(not used)
*/         

    //***********SVM learning***********
    svm.train( data_mat, res_mat, NULL, NULL, param );//training

	/***********auto training************
	svm.train_auto(data_mat,res_mat,
		(const CvMat*)0,(const CvMat*)0,param,5,
		CvParamGrid(pow(2,2),pow(2,8),2),
		CvParamGrid(pow(2,-13),pow(2,-7),2),
		CvParamGrid(1.0,1.0,0.0),
		CvParamGrid(1.0,1.0,0.0),
		CvParamGrid(1.0,1.0,0.0),
		CvParamGrid(1.0,1.0,0.0));
	*/
    svm.save( "test.xml" );  
	
	//CvSVMParams params = svm.get_params();
	//cout<<"params: C="<<params.C<<"; gamma="<<params.gamma<<endl;

    
    //test
    IplImage *test;    
    vector<string> img_tst_path;    
    ifstream img_tst( "testpredict.txt" );// a list of test set,  including file path only
    while( img_tst )    
    {    
        if( getline( img_tst, buf ) )    
        {    
            img_tst_path.push_back( buf );    
        }    
    }    
    img_tst.close();    
    
    
     
    char line[512];    
    ofstream predict_txt( "test.txt" );//store the result in txt
    for( string::size_type j = 0; j != img_tst_path.size(); j++ )//load all test samples one by one
    {    
        test = cvLoadImage( img_tst_path[j].c_str(), 1);    
        if( test == NULL )    
        {    
             cout<<" can not load the image: "<<img_tst_path[j].c_str()<<endl;    
               continue;    
         }    
            
       
		HOGDescriptor *hog=new HOGDescriptor(cvSize(20,40),cvSize(4,8),cvSize(2,4),cvSize(2,4),9);       
        vector<float>descriptors;//result array
		hog->compute(test, descriptors,Size(1,1), Size(0,0)); //start to compute HOG feature
        cout<<"HOG dims: "<<descriptors.size()<<endl;    
        CvMat* SVMtrainMat=cvCreateMat(1,descriptors.size(),CV_32FC1);    
        n=0;    
        for(vector<float>::iterator iter=descriptors.begin();iter!=descriptors.end();iter++)    
            {    
                cvmSet(SVMtrainMat,0,n,*iter);    
                n++;    
            }    
    
        int ret = svm.predict(SVMtrainMat);//get the final predict result
        std::sprintf( line, "%s %d\r\n", img_tst_path[j].c_str(), ret );    
        predict_txt<<line;    
    }    
    predict_txt.close();    
    

cvReleaseMat( &data_mat );    
cvReleaseMat( &res_mat );    
    
return 0;    
}