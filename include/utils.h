#include <opencv2/opencv.hpp>
#pragma once

//Bounding Boxes
struct BoundingBox : public cv::Rect2d {
  BoundingBox(){}
  BoundingBox(cv::Rect2d r): cv::Rect2d(r){}
public:
  float overlap;        //Overlap with current Bounding Box
  int sidx;             //scale index
};

void drawBox(cv::Mat& image, cv::Rect box, cv::Scalar color = cvScalarAll(255), int thick=1);

void drawPoints(cv::Mat& image, std::vector<cv::Point2f> points,cv::Scalar color=cv::Scalar::all(255));

cv::Mat createMask(const cv::Mat& image, CvRect box);

float median(std::vector<float> v);

std::vector<int> index_shuffle(int begin,int end);

