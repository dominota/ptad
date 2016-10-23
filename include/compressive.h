#ifndef COMPRESSIVE_H
#define COMPRESSIVE_H

#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <opencv2/tracking/feature.hpp>
#include <opencv2/tracking.hpp>
using namespace cv;
class compressive
{
public:
    enum
    {
       RatioClassifier = 0,
       BoostingMILClassifier = 1,
       AdaBoostinglassifier = 2,
       SVMClassifier = 3
    };

public:
  void prepareCompressive(const cv::Size& scales); // Compressive Tracking Object detection  for pyramids
  void getFeatureCompressive(const cv::Mat& image, std::vector<float>& ct_feature);  //or pyramids

  void prepareCompressiveScale(const std::vector<cv::Size>& scales); // Compressive Tracking Object detection
  void getFeatureCompressiveScale(const cv::Mat& image,const int& scale_idx, std::vector<float>& ct_feature);

  void prepareData( std::vector< std::vector<float> > & posfeatures,std::vector< std::vector<float> >& negfeatures,
                     cv::Mat & positiveStates, cv::Mat & negativeStates);

  void radioClassifier(std::vector<float>& features, float& radio);
  void radioClassifier(std::vector< std::vector<float> >& ffeatures, std::vector<float>& radio);
  void radioclassifierUpdate(std::vector< std::vector<float> >& posfeatures,std::vector< std::vector<float> > & negfeatures);

  void MilClassifierUpdate(std::vector< std::vector<float> >& posfeatures,std::vector< std::vector<float> > & negfeatures);
  void Milclassify(std::vector< std::vector<float> >& features, std::vector<float> & prob);

  struct Feature
      {
          int x1, y1, x2, y2;
          Feature() : x1(0), y1(0), x2(0), y2(0) {}
          Feature(int _x1, int _y1, int _x2, int _y2)
          : x1((int)_x1), y1((int)_y1), x2((int)_x2), y2((int)_y2)
          {}
      };

public:
  compressive();
  void initCompressive();
  int featureMinNumRect;
  int featureMaxNumRect;
  int featureNum;

  std::vector<float> muPositive;
  std::vector<float> sigmaPositive;

  std::vector<float> muNegative;
  std::vector<float> sigmaNegative;
  float learnRate;
  std::vector< std::vector< std::vector<Feature> > >Compressivefeatures;// features for extracting;
  std::vector< std::vector< std::vector<float> > >  CompressivefeaturesWeight;

  cv::RNG CTrng;
  bool ratioTrained;
  bool milTrained;
  cv::ClfMilBoost MilBoost;
};

#endif // COMPRESSIVE_H
