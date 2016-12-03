#include "system.h"
using namespace std;
using namespace cv;

cf_tracking::CfTracker* tracker_;
CompressiveTracker *compressivetracker;
System::System()
{
#if defined(TRACKER_LKAlgorithm)
    trackerProxy_ = Ptr<TrackerProxyImpl<TrackerLK, TrackerLK::Params> >
            (new TrackerProxyImpl<TrackerLK, TrackerLK::Params>());
#elif defined(KCFTrackerAlgorithm)
    cf_tracking::KcfParameters paras;
    tracker_ = new cf_tracking::KcfTracker(paras);
    // trackerProxy_ = Ptr<TrackerProxyImpl<TrackerKCF, TrackerKCF::Params> >
    //  (new TrackerProxyImpl<TrackerKCF, TrackerKCF::Params>());*/
#elif defined(BoostingTrackerAlgorithm)
    trackerProxy_ = Ptr<TrackerProxyImpl<TrackerBoosting, TrackerBoosting::Params> >
            (new TrackerProxyImpl<TrackerBoosting, TrackerBoosting::Params>());
#elif defined(MILTrackerAlgorithm)
    trackerProxy_ = Ptr<TrackerProxyImpl<TrackerMIL, TrackerMIL::Params> >
            (new TrackerProxyImpl<TrackerMIL, TrackerMIL::Params>());
#elif defined(CompressiveTrackerAlgorithm)
    compressivetracker = new CompressiveTracker();
#endif
    timages_.clear();
    detect_running = true;
    adjustingTypes  = NotChangingTrackingSate;
    thread_detected = boost::thread(&System::detect, this);
}
System::~System(){
    detect_running = false;
    thread_detected.join();
}
double System::overlap(const Rect& r1, const Rect& r2)
{
    double a1 = r1.area(), a2 = r2.area(), a0 = (r1&r2).area();
    return a0 / (a1 + a2 - a0);
}
void System::Run(const Mat& image,Rect2d& boundingBox,bool & tracked){
    // static int trackingerror = 0;
    static int frames =0 ;
    static int detections =1 ;
    frames++;
    cv::Mat image_gray;
#if defined(TRACKER_LKAlgorithm) //gray image initialization
    if(image.channels() > 1)
        cvtColor(image, image_gray, CV_BGR2GRAY);
    else
        image_gray = image.clone();
    bool trackedTmp = trackerProxy_->update(image_gray,boundingBox);
    if(tracked) tracked = trackedTmp;
#elif defined(KCFTrackerAlgorithm) ||  defined(BoostingTrackerAlgorithm)  ||  defined(MILTrackerAlgorithm)
    if(image.channels() >= 1)
        cvtColor(image, image_gray, CV_BGR2GRAY);
    else
    {
        image_gray = image.clone();
        cvtColor(image, image, CV_GRAY2BGR);
    }
    bool trackedTmp = tracker_->update(image,boundingBox);
    if(tracked)
        tracked = trackedTmp;
#elif defined(CompressiveTrackerAlgorithm)
    if(image.channels() == 1) // gray image
    {
        image_gray = image.clone();
        cvtColor(image, image, CV_GRAY2BGR);
    }
    else
        cvtColor(image, image_gray, CV_BGR2GRAY);
    cv::Rect box(boundingBox.x,boundingBox.y,boundingBox.width,boundingBox.height);
    bool trackedTmp;
    compressivetracker->processFrame(image_gray,box,trackedTmp);
    boundingBox = box;
    if(tracked)
        tracked = trackedTmp;
#endif
    Rect bb;
    bb.x = max(int(boundingBox.x),0);
    bb.y = max(int(boundingBox.y),0);
    bb.width  = min(min(double(image.cols-bb.x),boundingBox.width),min(boundingBox.width,boundingBox.br().x));
    bb.height = min(min(double(image.rows-bb.y),boundingBox.height),min(boundingBox.height,boundingBox.br().y));
    if( bb.width <=0  ||  bb.height <=0)
        tracked = false;
    else
    {
        Scalar stdev, mean;
        cv::Mat templateTracked;
        getPattern(image_gray(bb),templateTracked,mean,stdev);
#if defined(TRACKER_LKAlgorithm) //gray image initialization
        std::cout<<"tracking sucessful: "<<tracked<<std::endl;
#elif defined(KCFTrackerAlgorithm) //||defined(CompressiveTrackerAlgorithm)
        float maxNCC=-FLT_MAX;
        for(int k =0 ; k < trackingTemplate_.size(); k++)
        {
            cv::Mat ncc;
            cv::matchTemplate(templateTracked,trackingTemplate_[k],ncc,CV_TM_CCORR_NORMED);
            float nccP=((float*)ncc.data)[0];
            if(nccP > maxNCC)
                maxNCC = nccP;
        }
        float threshold = 0.65;
        if(maxNCC > threshold)
        {
            if(tracked)
                trackingTemplate_.push_back(templateTracked);
        }
        else
            tracked = false; // tracked = false very important Daviad //
        if(trackingTemplate_.size() > 5)
            trackingTemplate_.erase(trackingTemplate_.begin());
#endif
    }

    boost::unique_lock<boost::mutex> trackedLock(untrackDetectedMutex);
    timages_.push_back(image_gray.clone()); //gray image
    tboundBox_.push_back(BoundingBox(boundingBox));
    tsuccess_.push_back(tracked);
    newTrackFrame.notify_one();
    trackedLock.unlock();
    /*
    static BoundingBox box_detect;
    static  int adjusting = NotChangingTrackingSate ;
    static bool re_initialization  = false;
    static cv::Mat dimage;
    // simulation our PTAD framework with fixed KFs
    int KFs = 1;
    if((frames + KFs -1) % KFs == 0 ) //detect in 5 frame before
    {
        bool tsuccess = tracked;
        BoundingBox tboundingBox = BoundingBox(bb);
        detection_.detectProcess(image_gray,tboundingBox,tsuccess); //processed
        detection_.learn();
        box_detect = detection_.getDetectBox();
        adjusting = detection_.getReinitialization();
        re_initialization = false;
        dimage = image_gray.clone();
    }
    if(frames % KFs== 0) //correct and re-initialize  after 5 frames*/
    {
        //std::cout<<"adjusting0:  "<<adjustingTypes<<std::endl;
        cv::Mat dimage;
        BoundingBox box_detect;
        int adjusting = NotChangingTrackingSate;
        bool re_initialization = false;
        boost::unique_lock<boost::timed_mutex> detectedLock(undetecedTrackMutex,boost::defer_lock);
        auto now= boost::chrono::steady_clock::now();
        if(detectedLock.try_lock_until(now+ boost::chrono::milliseconds(15)))
        {
            box_detect = detectBox_;
            adjusting = adjustingTypes;//检测到了好的结果
            dimage = detectImg_.clone();
            adjustingTypes = NotChangingTrackingSate; //运行到这儿才表示初始化完成了，不管有没有成功
            detectImg_.release();
            if(!dimage.empty())  detections++;
            detectedLock.unlock();//*/
        }
        if(!dimage.empty() && (box_detect.x>0 && box_detect.y>0 && (box_detect.x+box_detect.width) <image_gray.cols
                               &&  (box_detect.y+box_detect.height) <image_gray.rows ) && adjusting!=NotChangingTrackingSate)
        {
            cv::Mat result,temple_gray;
            temple_gray  = dimage(box_detect);
            cv::Rect foundBox;
            cv::Point2f p_position;
            p_position.x = (float)box_detect.x + (float)box_detect.width / 2.0;
            p_position.y = (float)box_detect.y + (float)box_detect.height / 2.0;
            float scale = 3; //  for Moto //vrey imortant only for Moto 10 others for 3
            float left = MAX(round(p_position.x - (float)box_detect.width *scale), 0);
            float top =  MAX(round(p_position.y  - (float)box_detect.height*scale ), 0);
            float right = MIN(round(p_position.x  + (float)box_detect.width*scale ), image_gray.cols - 1);
            float bottom = MIN(round(p_position.y + (float)box_detect.height*scale), image_gray.rows - 1);
            cv::Rect roi((int) left, (int) top, (int) (right - left), (int) (bottom - top));
            cv::matchTemplate(image_gray(roi),temple_gray, result, CV_TM_CCORR_NORMED);
            cv::Mat iisum,iisqsum;
            cv::integral(image_gray(roi),iisum,iisqsum);
            filterVar(temple_gray.size(),iisum,iisqsum,result);
            double minVal; double maxVal; Point minLoc; Point maxLoc;
            cv::Point matchLoc;
            cv::minMaxLoc(result, &minVal, &maxVal,&minLoc, &maxLoc, Mat() );
            matchLoc = maxLoc;
            foundBox.x =  left + matchLoc.x ;//matchLoc.x;
            foundBox.y =  top  + matchLoc.y; //matchLoc.y;
            foundBox.width  = temple_gray.cols;
            foundBox.height = temple_gray.rows;
            double overlaps = overlap(foundBox,bb);

            if(!tracked)
            {

                boundingBox = foundBox;
                Scalar  mean, stdev;
                cv::Mat foundPatch;
                getPattern(image_gray(foundBox),foundPatch,mean,stdev); //???
                trackingTemplate_.push_back(foundPatch);
                if(trackingTemplate_.size() > 5)
                    trackingTemplate_.erase(trackingTemplate_.begin());
                re_initialization = true;

            }
            else
            {
                cv::Mat tracked_gray;
                tracked_gray =  image_gray(bb);
                if(overlaps >= 0.7 || (adjusting == ReCorrecting && overlaps >0.7)) //re_correct 表示跟踪成功了
                {
                    int weighted = 10;
                    boundingBox.x = cvRound((float)(weighted*bb.x+foundBox.x)/(float)(weighted+1));   // weighted average trackers trajectory with the close detections
                    boundingBox.y = cvRound((float)(weighted*bb.y+foundBox.y)/(float)(weighted+1));
                    boundingBox.width = cvRound((float)(weighted*bb.width+foundBox.width)/(float)(weighted+1));
                    boundingBox.height =  cvRound((float)(weighted*bb.height+foundBox.height)/(float)(weighted+1));

                    Scalar  mean, stdev;
                    cv::Mat foundPatch;
                    getPattern(image_gray(boundingBox),foundPatch,mean,stdev); //???
                    trackingTemplate_.push_back(foundPatch);
                    if(trackingTemplate_.size() > 5)
                        trackingTemplate_.erase(trackingTemplate_.begin());
                    tracked = true;

#if defined(TRACKER_LKAlgorithm) //gray image initialization
                    re_initialization =true;
#elif defined(KCFTrackerAlgorithm) //color image initialization
                    re_initialization =false;
#elif defined(BoostingTrackerAlgorithm)  ||  defined(MILTrackerAlgorithm)
#elif  defined(CompressiveTrackerAlgorithm)
                    re_initialization =true;
#endif
                }
                else
                {
                    Scalar  mean, stdev;
                    cv::Mat trackedpatch,templepatch,foundPatch,ncc;
                    getPattern(tracked_gray,trackedpatch,mean,stdev);
                    getPattern(temple_gray,templepatch,mean,stdev);
                    cv::matchTemplate(trackedpatch,templepatch,ncc,CV_TM_CCORR_NORMED);
                    float nccP=((float*)ncc.data)[0];//(((float*)ncc.data)[0]+1)*0.5;
                    getPattern(image_gray(foundBox),foundPatch,mean,stdev);
                    cv::matchTemplate(foundPatch,templepatch,ncc,CV_TM_CCORR_NORMED);
                    float nccP2=((float*)ncc.data)[0];//(((float*)ncc.data)[0]+1)*0.5;
                    float similarity = 0.65;
                    if( nccP2 > nccP && nccP2 >similarity  || (adjusting == ReInitialization && nccP2 >nccP && nccP2 >similarity)) //可能有变化
                    {
                        {
                            re_initialization = true;
                            boundingBox = foundBox;
                            trackingTemplate_.push_back(foundPatch);
                        }
                        if(trackingTemplate_.size() > 5)
                            trackingTemplate_.erase(trackingTemplate_.begin());
                    }
                }
            }
            if(re_initialization)
            {
                tracked = true;
#if defined(TRACKER_LKAlgorithm) //gray image initialization
                trackerProxy_->init(image_gray,boundingBox);
#elif defined(KCFTrackerAlgorithm) //color image initialization
                tracker_->reinit(image,boundingBox);
#elif defined(BoostingTrackerAlgorithm)  ||  defined(MILTrackerAlgorithm)
                trackerProxy_->init(image,boundingBox);
#elif defined(CompressiveTrackerAlgorithm)
                cv::Rect init_box = boundingBox;
                if(adjusting ==ReCorrecting)
                    boundingBox = bb;
                else
                    compressivetracker->init(image_gray,init_box);
#endif
            }
            // printf("Detection rate: %d/%d\n",detections,frames);
        }
        else
            boundingBox = bb;
    }
}
void System::filterVar(const cv::Size box,const cv::Mat & sum, const cv::Mat & sqsum,cv::Mat & result)
{
    cv::Size imageSize(sum.cols-1,sum.rows-1);
    CV_Assert(box.width <= imageSize.width || box.height <= imageSize.height);
    CV_Assert(result.rows ==imageSize.height - box.height + 1 || result.cols == imageSize.width - box.width+ 1);
    float * ptr_data = (float*)result.data;
    for(int curRow = 0 ; curRow < result.rows; curRow++)
    {
        for(int curCol = 0; curCol < result.cols; curCol++)
        {
            double brs = sum.at<int>(curRow+box.height,curCol+box.width);
            double bls = sum.at<int>(curRow+box.height,curCol);
            double trs = sum.at<int>(curRow,curCol+box.width);
            double tls = sum.at<int>(curRow,curCol);
            double brsq = sqsum.at<double>(curRow+box.height,curCol+box.width);
            double blsq = sqsum.at<double>(curRow+box.height,curCol);
            double trsq = sqsum.at<double>(curRow,curCol+box.width);
            double tlsq = sqsum.at<double>(curRow,curCol);
            double mean = (brs+tls-trs-bls)/((double)box.area());
            double sqmean = (brsq+tlsq-trsq-blsq)/((double)box.area());
            if(sqmean-mean*mean < var)
                ptr_data[curRow*result.cols+curCol] = 0;
        }
    }
    // std::cout<<result;
}

void System::getPattern(const Mat& img, Mat& pattern,Scalar& mean,Scalar& stdev){
    int patch_size = 15;
    //Output: resized Zero-Mean patch
    resize(img,pattern,Size(patch_size,patch_size));
    meanStdDev(pattern,mean,stdev);
    pattern.convertTo(pattern,CV_32F);
    pattern = pattern-mean.val[0];
}
void System::detect()
{
    cv::Mat image;
    bool tsuccess=false;
    BoundingBox tboundingBox;
    while(detect_running)
    {
        boost::unique_lock<boost::mutex> trackedLock(untrackDetectedMutex);
        newTrackFrame.wait(trackedLock);
        int nums = timages_.size();
        if(nums > 0)
        {
            image    =  timages_[nums-1];
            tsuccess =  tsuccess_[nums-1];
            tboundingBox = tboundBox_[nums-1];
            timages_.clear();
            tboundBox_.clear();
            tsuccess_.clear();
        }
        trackedLock.unlock();
        if(nums>0)
        {
            detection_.detectProcess(image,tboundingBox,tsuccess); //processed
            boost::unique_lock<boost::timed_mutex> detectedLock(undetecedTrackMutex,boost::defer_lock);
            auto now= boost::chrono::steady_clock::now();
            if(detectedLock.try_lock_until(now+ boost::chrono::milliseconds(15)))
            {
                if(detection_.detected)
                {
                    detectBox_ = detection_.getDetectBox();
                    detectImg_ = image.clone();
                    adjustingTypes = detection_.getReinitialization();
                }
                detectedLock.unlock();
            }
            detection_.learn();
        }
    }
}

void System::initImpl(const Mat& image, const Rect2d& boundingBox)
{
    cv::Mat image_gray;
#if defined(TRACKER_LKAlgorithm) //gray image initialization
    if(image.channels() != 1)
        cvtColor(image, image_gray, CV_BGR2GRAY);
    else
        image_gray = image;
    trackerProxy_->init(image_gray,boundingBox);
#elif defined(KCFTrackerAlgorithm) //color image initialization
    if(image.channels() == 1)
    {
        image_gray = image;
        cvtColor(image, image, CV_GRAY2BGR);
    }
    else
        cvtColor(image, image_gray, CV_BGR2GRAY);  //gray image
    cv::Rect2d init_box = boundingBox;
    tracker_->reinit(image,init_box);
#elif defined(BoostingTrackerAlgorithm) || defined(MILTrackerAlgorithm)
    if(image.channels() == 1) // 灰度图像
    {
        image_gray = image;
        cvtColor(image, image, CV_GRAY2BGR);
    }
    else
        cvtColor(image, image_gray, CV_BGR2GRAY);
    cv::Rect2d init_box = boundingBox;
    trackerProxy_->init(image,boundingBox);
#elif defined(CompressiveTrackerAlgorithm)
    if(image.channels() == 1) // 灰度图像
    {
        image_gray = image;
        cvtColor(image, image, CV_GRAY2BGR);
    }
    else
        cvtColor(image, image_gray, CV_BGR2GRAY);
    cv::Rect init_box = boundingBox;
    compressivetracker->init(image_gray,init_box);
#endif
    Scalar mean, stdev;
    cv::Mat templateTracked;
    getPattern(image_gray(boundingBox),templateTracked,mean,stdev);
    trackingTemplate_.push_back(templateTracked);
    var = pow(stdev.val[0],2)*0.5;
    detection_.init(image_gray,boundingBox); // gray image initialization
}
