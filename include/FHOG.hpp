/*
C++ Felzenszwalb HOG extractor

This repository is meant to provide an easy-to-use implementation of the Felzenszwalb HOG features extractor.
This approach followed the one presented by Felzenszwalb, Pedro F., et al. "Object detection with discriminatively trained part-based models." Pattern Analysis and Machine Intelligence, IEEE Transactions on 32.9 (2010): 1627-1645. 
The OpenCV library have only the original HOG, proposed by Dalal and Triggs. However, the Latent SVM OpenCV implementation have its own FHOG extractor. This code allows you to use it without having do deal with Latent SVM objects.

To run the code you need OpenCV library.

Author: Joao Faro
Contacts: joaopfaro@gmail.com
*/

#ifndef FHOG_H
#define FHOG_H
#include "fhogtools.h"
#include <opencv2/opencv.hpp> 


//int getFeatureMaps(const IplImage*, const int , CvLSVMFeatureMap **);
//int normalizeAndTruncate(CvLSVMFeatureMap *, const float );
//int PCAFeatureMaps(CvLSVMFeatureMap *);
//int freeFeatureMapObject (CvLSVMFeatureMap **);

class HogFeature {
    const double MAX_TRACKING_AREA = 128;
public:
    HogFeature();
    HogFeature(uint cell_size, cv::Size & size_scale);
    void init(uint cell_size, cv::Size & size_scale);
    void set_tmpl_sz(uint cell_size,cv::Size  & tmplsz);
    ~HogFeature();
    virtual HogFeature* clone() const;
    void getFeature(const cv::Mat & image,cv::Mat & feature);

public:
    uint _cell_size;
    uint _scale;
    cv::Size _tmpl_sz;
    int featureNums;
    cv::Mat _featuresMap;
    cv::Mat _featurePaddingMat;
    CvLSVMFeatureMapCaskade *_map;
};

#endif
