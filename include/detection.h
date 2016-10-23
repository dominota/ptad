#include <opencv2/opencv.hpp>
#include <utils.h>
#include <Classifier.h>
#include <fstream>
#include <opencv2/tracking.hpp>
#include "opencv2/tracking/tracker.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include <unordered_map>
template <typename T>
struct DetStruct {
    std::vector<int> bb;
    std::vector<int> candidatebb;
    std::vector<std::vector<T> > patt; //features
    std::vector<T>  patt1;
    std::vector<float> conf1;
    std::vector<float> conf2;
    std::vector<std::vector<int> > isin;
    std::vector<cv::Mat> patch;
};

template <typename T>
struct TempStruct {
    std::vector<std::vector<T> > patt;
    std::vector<T>  patt1;
    std::vector<float> conf;
    std::vector<int> candidateIndex;
};

struct OComparator{
    OComparator(const std::vector<BoundingBox>& _grid):grid(_grid){}
    std::vector<BoundingBox> grid;
    bool operator()(int idx1,int idx2){
        return grid[idx1].overlap > grid[idx2].overlap;
    }
};

struct CComparator{
    CComparator(const std::vector<float>& _conf):conf(_conf){}
    std::vector<float> conf;
    bool operator()(int idx1,int idx2){
        return conf[idx1]> conf[idx2];
    }
};


class Detection{
public:
    enum
    {
       NotChangingTrackingSate = 0,
       ReInitialization = 1,
       ReCorrecting = 2
    };
private:

    int selectionFeature;
    Classifier classifier;
    ///Parameters
    int bbox_step;
    int min_win;
    int patch_size;
    //initial parameters for positive examples
    int num_closest_init;
    int num_warps_init;
    int noise_init;
    float angle_init;
    float shift_init;
    float scale_init;
    //update parameters for positive examples
    int num_closest_update;
    int num_warps_update;
    int noise_update;
    float angle_update;
    float shift_update;
    float scale_update;
    //parameters for negative examples
    float bad_overlap;
    float bad_patches;
    int increase_ncc_samples;
    int pyramidlevels;
    bool directionalFilter;
    ///Variables
    //Integral Images
    float var;
    float col_var;
    float row_var;
    cv::Mat pEx;  //positive NN example
    std::vector<cv::Mat> nEx;  //negative NN examples
    std::vector<cv::Mat> nExT; //data to Test
    //Last frame data
    bool  lastdetect;
    BoundingBox lastbox;
    bool lastvalid;
    bool tmplastvalid;
    float lastconf;
    std::vector<Mat> nn_examples;
    int nn_posnums;

    int levelOfTracker;
    int processNumsofLevels;
    bool is_debug ;
#if defined(FernFeature)  //select one features
    TempStruct<int> tmp;
    DetStruct<int>  dt;
    std::vector<std::pair<std::vector<int>,int> > pX;  //positive ferns <features,labels=1>
    std::vector<std::pair<std::vector<int>,int> > nX; // negative ferns <features,labels=0>
    std::vector<std::pair<std::vector<int>,int> > nXT; //data to Test
#elif defined(CompressiveFeature)
    TempStruct<float> tmp;
    DetStruct<float>  dt;
    std::vector<std::vector<float> > pX; //compressive ct_nx
    std::vector<std::vector<float> > nX; //compressive ct_nx
    std::vector<std::vector<float> > fX; //test（known unlebeled）
#elif defined(HaarLikeFeature)
    TempStruct<cv::Mat> tmp;
    DetStruct<cv::Mat>  dt;
    std::vector<cv::Mat> pX;
    std::vector<cv::Mat> nX;
    std::vector<cv::Mat> fX;
#elif defined(HOGFeature)  //using pyramids
    TempStruct<cv::Mat> tmp;
    DetStruct<cv::Mat>  dt;
    std::vector<cv::Mat> pX;
    std::vector<cv::Mat> nX;
#endif
#ifdef using_pyramid   //using image pyramids
    std::vector<cv::Mat> timages;
    std::vector<cv::Mat> filterImages;
    std::vector<cv::Mat> iisum;
    std::vector<cv::Mat> iisqsum;
    std::vector<BoundingBox> gridpyramids;  //sliding windows
    std::vector<cv::Size>    scalespyramids;
    std::unordered_map<float,int> scaleindex;
    BoundingBox best_pyramidbox; // maximum overlapping bbox
    std::vector<int> scalesgridnums;
#else
    cv::Mat timages;
    cv::Mat filterImages;
    cv::Mat iisum;
    cv::Mat iisqsum;
#endif
    cv::Mat frameImage;
    std::vector<BoundingBox> dbb;
    std::vector<float> dconf;
    std::vector<BoundingBox> candidatedbb;
    std::vector<float> candidatedconf;
    std::vector<BoundingBox> grid;    // //sliding windows
    std::vector<int> BoundingBoxScales;    // //sliding windows
    std::vector<float> scales;
    std::vector<cv::Size> scalesizes;
    std::vector<int> good_boxes; //indexes of bboxes with overlap > 0.6
    std::vector<int> bad_boxes; //indexes of bboxes with overlap < 0.2
    BoundingBox bbhull; // hull of good_boxes
    BoundingBox best_box; // maximum overlapping bbox

    int directionFilterValid = 0;
public:
    //Constructors
    void clear();
    Detection();
    ~Detection();
    Detection(const cv::FileNode& file);
    void read(const cv::FileNode& file);
#ifdef using_pyramid
    void constructImagePyramids(const cv::Mat& frame);
#endif
    void init(const cv::Mat& frame,const cv::Rect &box);
    void generatePositiveData(int num_warps);
    void generateNegativeData();

    void detect();     // random fern detect
    void clusterConf(const std::vector<BoundingBox>& dbb,const std::vector<float>& dconf,std::vector<BoundingBox>& cbb,std::vector<float>& cconf);
    void evaluate();
    void learn();
    void buildGrid(const cv::Mat& img, const cv::Rect& box);
    float bbOverlap(const BoundingBox& box1,const BoundingBox& box2);
    void getOverlappingBoxes(const cv::Rect& box1,int num_closest);
    void getBBHull();
    void getPattern(const cv::Mat& img, cv::Mat& pattern,cv::Scalar& mean,cv::Scalar& stdev);
    void bbPoints(std::vector<cv::Point2f>& points, const BoundingBox& bb);
    void bbPredict(const std::vector<cv::Point2f>& points1,const std::vector<cv::Point2f>& points2,
                   const BoundingBox& bb1,BoundingBox& bb2);
    double getVar(const BoundingBox& box,const cv::Mat& sum,const cv::Mat& sqsum);
    double getColVar(const BoundingBox& box,const cv::Mat& sum,const cv::Mat& sqsum);
    double getRowVar(const BoundingBox& box,const cv::Mat& sum,const cv::Mat& sqsum);
    bool filterVar(const BoundingBox& box,const cv::Mat& sum,const cv::Mat& sqsum,double factor =1);
    bool bbComp(const BoundingBox& bb1,const BoundingBox& bb2);
    int  clusterBB(const std::vector<BoundingBox>& dbb,std::vector<int>& indexes);
    void resample(const Mat& img, const Rect2d& r2, Mat_<uchar>& samples);
    void resample(const Mat& img, const RotatedRect& r2, Mat_<uchar>& samples);
    void detectProcess(cv::Mat & image, BoundingBox & tboundingBox, bool & tracked);
    int middleClassifier(BoundingBox bb ,bool & is_pass);
    void determinateTrackingState(cv::Mat & image, BoundingBox & tboundingBox, bool & tracked,bool & dtracked, bool & tvalid,float & tconf);
    BoundingBox  getDetectBox();
    bool getReinitialization();
    bool detected;
    int adjustingTypes;
public:


};

