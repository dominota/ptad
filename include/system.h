#ifndef SYSTEM_H
#define SYSTEM_H

#include <opencv2/opencv.hpp>

#include <detection.h>
#include <utils.h>
#include <tracking.hpp>
#include <TrackerLK.hpp>

#include <boost/thread.hpp>
#include <boost/thread/shared_mutex.hpp>
#include <boost/thread/condition_variable.hpp>
#include <boost/thread/locks.hpp>

//#define TRACKER_LKAlgorithm
#define KCFTrackerAlgorithm
//#define BoostingTrackerAlgorithm
//#define MILTrackerAlgorithm
//#define CompressiveTrackerAlgorithm
#include "kcf_tracker.hpp"
#include "dsst_tracker.hpp"
#include "CompressiveTracker.h"

class System
{
    enum
    {
       NotChangingTrackingSate = 0,
       ReInitialization = 1,
       ReCorrecting = 2
    };
public:
    System();
    ~System();
    void Run(const Mat& image,Rect2d& boundingBox,bool & tracked);
    void detect();
    void initImpl(const Mat& image,  const Rect2d& boundingBox);
    double overlap(const Rect& r1, const Rect& r2);
    void getPattern(const Mat& img, Mat& pattern,Scalar& mean,Scalar& stdev);
    void filterVar(const cv::Size box,const cv::Mat & sum,const cv::Mat & sqsum,cv::Mat & result);
public:
    Ptr<TrackerProxy>        trackerProxy_;
    std::vector<cv::Mat>     timages_;
    std::vector<bool>        tsuccess_;
    std::vector<BoundingBox> tboundBox_;
    std::vector<cv::Mat> trackingTemplate_;
    Detection    detection_;
    cv::Mat      detectImg_;
    BoundingBox  detectBox_;
    double var;
    int  adjustingTypes;


    boost::thread thread_detected;
    bool detect_running;
    boost::thread thread_constraint_search;
    boost::mutex undetecedTrackMutex;
    boost::mutex untrackDetectedMutex;
    boost::condition_variable  newTrackFrame;

    boost::condition_variable  newDetectFrame;
};

#endif // SYSTEM_H
