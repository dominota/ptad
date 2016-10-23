/*
C++ Felzenszwalb HOG extractor
 
This repository is meant to provide an easy-to-use implementation of the Felzenszwalb HOG features extractor.
This approach followed the one presented by Felzenszwalb, Pedro F., et al. "Object detection with discriminatively trained part-based models." Pattern Analysis and Machine Intelligence, IEEE Transactions on 32.9 (2010): 1627-1645. 
The OpenCV library have only the original HOG, proposed by Dalal and Triggs. However, the Latent SVM OpenCV implementation have its own FHOG extractor. This code allows you to use it without having do deal with Latent SVM objects.

To run the code you need OpenCV library.

Author: Joao Faro
Contacts: joaopfaro@gmail.com
*/

#include "FHOG.hpp"

using namespace std;
using namespace cv;

template<typename T>
cv::Size_<T> sizeFloor(cv::Size_<T> size)
{
    return cv::Size_<T>(floor(size.width), floor(size.height));
}

HogFeature::HogFeature(){
	_cell_size = 4;
	_scale = 1;
}

HogFeature::HogFeature(uint cell_size,cv::Size & size_scale){
    cv::Size2d targetSize = size_scale ;
    double scaleFactor = sqrt(MAX_TRACKING_AREA / targetSize.area());
    _tmpl_sz = sizeFloor(targetSize * scaleFactor);

     _cell_size = cell_size;

     _tmpl_sz.width = ( ( (int)(_tmpl_sz.width / (2 * _cell_size)) ) * 2 * _cell_size ) +_cell_size*2;
     _tmpl_sz.height = ( ( (int)(_tmpl_sz.height / (2 * _cell_size)) ) * 2 * _cell_size ) +_cell_size*2;

    featureNums = static_cast<int>(floor(_tmpl_sz.width / _cell_size -2) * floor(_tmpl_sz.height / _cell_size -2) * (NUM_SECTOR * 3 + 4));



}
void HogFeature::init(uint cell_size, cv::Size & size_scale)
{
     cv::Size2d targetSize = size_scale ;
     double scaleFactor = sqrt(MAX_TRACKING_AREA / targetSize.area());
     _tmpl_sz = sizeFloor(targetSize * scaleFactor);

     _cell_size = cell_size;

     _tmpl_sz.width = ( ( (int)(_tmpl_sz.width / (2 * _cell_size)) ) * 2 * _cell_size ) +_cell_size*2;
     _tmpl_sz.height = ( ( (int)(_tmpl_sz.height / (2 * _cell_size)) ) * 2 * _cell_size )+_cell_size*2 ;

     featureNums = static_cast<int>(floor(_tmpl_sz.width / _cell_size -2) * floor(_tmpl_sz.height / _cell_size -2) * (NUM_SECTOR * 3 + 4));
}

void HogFeature::set_tmpl_sz(uint cell_size,cv::Size & size_scale)
{
     _cell_size = cell_size;
     //变化size_scale到固定大小；
     size_scale.width = ( ( (int)(size_scale.width / (2 * _cell_size)) ) * 2 * _cell_size ) +_cell_size*2;
     size_scale.height = ( ( (int)(size_scale.height / (2 * _cell_size)) ) * 2 * _cell_size )+_cell_size*2 ;

     _tmpl_sz = size_scale;
     featureNums = static_cast<int>(floor(_tmpl_sz.width / _cell_size -2) * floor(_tmpl_sz.height / _cell_size -2) * (NUM_SECTOR * 3 + 4));
}


HogFeature::~HogFeature(){
	freeFeatureMapObject(&_map);	
}

HogFeature* HogFeature::clone() const{
    return new HogFeature(*this);
}

void HogFeature::getFeature(const cv::Mat & image,cv::Mat & feature){

    cv::Mat convertImage;
    if (image.cols != _tmpl_sz.width || image.rows != _tmpl_sz.height) {
        resize(image, convertImage, _tmpl_sz);
    }
    convertImage.convertTo(convertImage, CV_32F);
    convertImage *= 0.003921568627451; //1/255
    //cv::Mat featurePaddingMat;
    // Add extra cell filled with zeros around the image
    //if(image.channels() == 3)
     //   featurePaddingMat = cv::Mat( _tmpl_sz.height+_cell_size*2, _tmpl_sz.width+_cell_size*2, CV_32FC3, cvScalar(0,0,0) ); //FHOG feature color
    //if(image.channels() == 1)
    static cv::Mat featurePaddingMat = cv::Mat( _tmpl_sz.height+_cell_size*2, _tmpl_sz.width+_cell_size*2, CV_32FC1, cvScalar(0,0,0) ); //FHOG feature color
    //image.copyTo(featurePaddingMat.rowRange(_cell_size, _cell_size+_tmpl_sz.height).colRange(_cell_size, _cell_size+_tmpl_sz.width));
    convertImage.copyTo(featurePaddingMat); //_tmpl_sz
    // HOG features
    IplImage zz = featurePaddingMat;
    getFeatureMaps(&zz, _cell_size, &_map);
    normalizeAndTruncate(_map, 0.2f);
    PCAFeatureMaps(_map);
    feature = Mat(Size(_map->numFeatures*_map->sizeX*_map->sizeY,1), CV_32FC1, _map->map).clone();  // Procedure do deal with cv::Mat multichannel bug
    CV_Assert(featureNums == feature.cols);
    freeFeatureMapObject(&_map);
}
