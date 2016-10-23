#ifndef HAARFEATURE_H
#define HAARFEATURE_H

#include <stdio.h>
#include <opencv2/opencv.hpp>

#include "onlineBoosting.hpp"
#include "onlineMIL.hpp"
#include "feature.hpp"
#include "trackerFeature.hpp"
#include "trackerFeatureSet.hpp"
#include <opencv2/tracking/feature.hpp>
#include <opencv2/tracking.hpp>

using namespace cv;
class HaarFeature
{
public:
    enum
    {
       RatioClassifierAlgorithm = 0,
       BoostingMILClassifierAlgorithm = 1,
       AdaBoostinglassifierAlgorithm = 2,
       SVMClassifierAlgorithm = 3
    };

public:

    HaarFeature();
    void initHaarFeature(int learningMethod = BoostingMILClassifierAlgorithm);

    bool prepareHaar(const cv::Size& scale); // Haar Tracking Object detection for image pyramids
    bool prepareHaarScale(const std::vector<cv::Size>& scales); // Haar Tracking Object detection for orignal image with scale information

    void getFeatureHaar( std::vector<cv::Mat>& image,  cv::Mat & Features);
    void getFeatureHaarScale( std::vector<cv::Mat>& image,  cv::Mat & Features, std::vector<int> scale);
    void getFeatureHaar( std::vector<cv::Mat>& image,  std::vector<cv::Mat> & Features);
    void getFeatureHaarScale( std::vector<cv::Mat>& image,  std::vector<cv::Mat> & Features, std::vector<int> scale);


    void  ratioclassifierUpdate(cv::Mat & positiveStates, cv::Mat & negativeStates);
    void  ratioclassifierUpdate(std::vector<cv::Mat> & positiveStates, std::vector<cv::Mat> & negativeStates);
    void  radioClassifier(cv::Mat & positiveStates,std::vector<float> & prob);
    void  radioClassifier(std::vector<cv::Mat> & positiveStates,std::vector<float> & prob);

    void  MilClassifierUpdate(cv::Mat & positiveStates, cv::Mat & negativeStates);
    void  MilClassifierUpdate(std::vector<cv::Mat> & positiveStates, std::vector<cv::Mat> & negativeStates);
    void  Milclassify(cv::Mat & States, std::vector<float> & prob);
    void  Milclassify(std::vector<cv::Mat> & States, std::vector<float> & prob);


    void  BoostingclassifierUpdate(cv::Mat& posfeatures,cv::Mat & negfeatures);
    void  iterationInitBoosting(cv::Mat& posfeatures,cv::Mat & negfeatures);
    void  BoostingClassify(cv::Mat& posfeatures, std::vector<float> & prob);

    void  BoostingclassifierUpdate(std::vector<cv::Mat>& posfeatures,std::vector<cv::Mat> & negfeatures);
    void  iterationInitBoosting(std::vector<cv::Mat>& posfeatures,std::vector<cv::Mat> & negfeatures);
    void  BoostingUpdateInterface(std::vector<cv::Mat>& posfeatures,std::vector<cv::Mat> & negfeatures);
    void  BoostingClassify(std::vector<cv::Mat>& posfeatures, std::vector<float> & prob);

    void  classifier(std::vector<cv::Mat> & States,std::vector<float> & prob);
    void  classifierUpdate(std::vector<cv::Mat> & positiveStates, std::vector<cv::Mat> & negativeStates);


public:

    float learnRate;
    int featureNum;
    int learningAlgorithm;
    // ratioClassifier paras
    bool ratioTrained;
    std::vector<float> muPositive;
    std::vector<float> sigmaPositive;
    std::vector<float> muNegative;
    std::vector<float> sigmaNegative;
    std::vector<cv::Size> scaleSize;

    //MIL paras
    bool Miltrained;
    onlineMIL::ClfMilBoost MilBoost;


    // Adaboostin paras
    int numClassifiers;  //!<the number of classifiers to use in a OnlineBoosting algorithm
    int iterationInit;  //!<the initial iterations
    int weakclassifiersfactors;
    int featureSetNumFeatures;  //!< # features
    int numBaseClassifier;
    bool boostingTrained;
    Size initPatchSize;//no used for our algorithms
    Rect sampleROI; //no used for our algorithms
    std::vector<int> replacedClassifier;
    std::vector<int> swappedClassifier;


    Ptr<features::TrackerFeature> trackerFeature;
    Ptr<features::TrackerFeatureSet> featureSet;
    Ptr<onlineboosting::StrongClassifierDirectSelection> boostingClassifier;
};

#endif // HAARFEATURE_H
