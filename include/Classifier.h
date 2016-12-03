/*
 * FerNNClassifier.h
 *
 *  Created on: Jun 14, 2011
 *      Author: alantrrs
 */

#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <opencv2/tracking/feature.hpp>
#include <opencv2/tracking.hpp>

#include <compressive.h>
#include <haarfeature.h>
#include <hogfeature.h>
using namespace cv;

//#define  using_pyramid
#define  FernFeature
#define  CompressiveFeature // no using_pyramid
#define  HaarLikeFeature  // no using_pyramid
#define  HOGFeature  //HOG must use image pyramids

class Classifier{
private:
  float thr_fern;
  int structSize;
  int nstructs;
  float valid;
  float ncc_thesame;
  float thr_nn;
  int acum;
public:
  //Parameters
  float thr_nn_valid;

  void read(const cv::FileNode& file);

  void prepare(const cv::Size & scales);  //using pyramids
  void getFeatures(const cv::Mat& image,std::vector<int>& fern);  //using pyramids

  void prepareScale(const std::vector<cv::Size>& scales);   //not pyramids
  void getFeaturesScale(const cv::Mat& image,const int& scale_idx,std::vector<int>& fern); //not pyramids


  void update(const std::vector<int>& fern, int C, int N);
  float measure_forest(std::vector<int> fern);
  void trainF(const std::vector<std::pair<std::vector<int>,int> >& ferns,int resample);
  void trainNN(const std::vector<cv::Mat>& nn_examples,int nums_possamples =1);
  float NNConf(const cv::Mat& example,std::vector<int>& isin,float& rsconf,float& csconf);
  void evaluateNCCTh(const std::vector<cv::Mat>& nExT);
  void evaluateFernTh(const std::vector<std::pair<std::vector<int>,int> >& nXT);
  void show();
  //Ferns Members
  int getNumStructs(){return nstructs;}
  float getFernTh(){return thr_fern;}
  float getNNTh(){return thr_nn;}

  void clear();
  struct Feature
      {
          int x1, y1, x2, y2;
          Feature() : x1(0), y1(0), x2(0), y2(0) {}
          Feature(int _x1, int _y1, int _x2, int _y2)
          : x1((int)_x1), y1((int)_y1), x2((int)_x2), y2((int)_y2)
          {}
          bool operator ()(const cv::Mat& patch) const
          { return patch.at<uchar>(y1,x1) > patch.at<uchar>(y2, x2); }
      };


  std::vector<std::vector<Feature> > features; //Ferns features (one std::vector for each scale)

  std::vector< std::vector<int> > nCounter; //negative counter
  std::vector< std::vector<int> > pCounter; //positive counter
  std::vector< std::vector<float> > posteriors; //Ferns posteriors
  float thrN;  //Negative threshold
  float thrP;  //Positive thershold
  //NN Members
  std::vector<cv::Mat> pEx;  //otherwarp templates
  std::vector<cv::Mat> incrpEx; //warp templates with best box
  std::vector<cv::Mat> opEx; //  template from orginal image
  std::vector<cv::Mat> nEx;  //NN negative examples

  compressive compDetector;
  HaarFeature haarfeature;
  std::vector< cv::Mat> haarSamples;

  FHOGFeature fhogfeature;
};
