#ifndef TRACKING_HPP
#define TRACKING_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#define TRACKER_LK
using namespace cv;
class TrackerProxy
{
public:
    virtual bool init(const Mat& image, const Rect2d& boundingBox) = 0;
    virtual bool update(const Mat& image, Rect2d& boundingBox) = 0;
#ifdef TRACKER_LK
    virtual bool reinit(const Mat& image, Rect2d& boundingBox) =0;
#endif
    virtual ~TrackerProxy(){}
};

template<class T, class Tparams>
class TrackerProxyImpl : public TrackerProxy
{
public:
    TrackerProxyImpl(Tparams params = Tparams()) :params_(params){}
    bool init(const Mat& image, const Rect2d& boundingBox)
    {
        trackerPtr = T::createTracker();
        return trackerPtr->init(image, boundingBox);
    }
    bool update(const Mat& image, Rect2d& boundingBox)
    {
        return trackerPtr->update(image, boundingBox);
    }

#ifdef TRACKER_LK
    bool reinit(const Mat& image, Rect2d& boundingBox)
    {
        return trackerPtr->reinit(image, boundingBox);
    }
#endif
private:
    Ptr<T> trackerPtr;
    Tparams params_;
    Rect2d boundingBox_;
};


#endif // TRACKING_HPP
