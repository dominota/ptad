#include <TrackerLK.hpp>

using namespace std;

namespace cv{


TrackerAdditional::~TrackerAdditional(){
}

bool TrackerAdditional::init( const Mat& image, const Rect2d& boundingBox )
{

  if( isInit )
  {
    return false;
  }

  if( image.empty() )
    return false;

  bool initTracker = initImpl( image, Rect2i(boundingBox) );

  if( initTracker )
  {
    isInit = true;
  }

  return initTracker;
}

bool TrackerAdditional::update( const Mat& image, Rect2d& boundingBox )
{

  if( !isInit )
  {
    return false;
  }

  if( image.empty() )
    return false;

  return updateImpl( image, boundingBox );
}

bool TrackerAdditional::reinit(const Mat& image, Rect2d& boundingBox)
{
    isInit = false;
    if( image.empty() )
      return false;
    bool initTracker = reinitImpl( image, boundingBox) ;
    if( initTracker )
      isInit = true;
    return initTracker;
}
class TrackerLKImpl: public TrackerLK
{
private:
    std::vector<cv::Point2f> pointsFB;

    std::vector<uchar> status;
    std::vector<uchar> FB_status;
    std::vector<float> similarity;
    std::vector<float> FB_error;
    float simmed;
    float fbmed;


    void normCrossCorrelation(const cv::Mat& img1,const cv::Mat& img2, std::vector<cv::Point2f>& points1, std::vector<cv::Point2f>& points2);

    bool filterPts(std::vector<cv::Point2f>& points1,std::vector<cv::Point2f>& points2);
public:
    TrackerLKImpl(const TrackerLK::Params &parameters = TrackerLK::Params());
    bool trackf2f(const cv::Mat& img1, const cv::Mat& img2,
                  std::vector<cv::Point2f> &points1, std::vector<cv::Point2f> &points2);
    float getFB(){return fbmed;}
    float median(std::vector<float> v);
    void  bbPoints(std::vector<cv::Point2f>& points, const Rect& bb);
    void  bbPredict(const std::vector<cv::Point2f>& points1,const std::vector<cv::Point2f>& points2,const Rect& bb1,Rect& bb2);

    void read( const FileNode& fn );
    void write( FileStorage& fs ) const;

public:
    cv::Mat  lastImg;
    cv::Rect lastbox;

protected:

    bool reinitImpl( const Mat& image, Rect2d& boundingBox );
    bool initImpl( const Mat& image, const Rect2d& boundingBox );
    bool updateImpl( const Mat& image, Rect2d& boundingBox );
    TrackerLK::Params params;
};


/*
* Constructor
*/
Ptr<TrackerLK> TrackerLK::createTracker(const TrackerLK::Params &parameters){
    return Ptr<TrackerLKImpl>(new TrackerLKImpl(parameters));
}
/*
* Params
*/
TrackerLK::Params::Params()
{
    term_criteria = TermCriteria( TermCriteria::COUNT+TermCriteria::EPS, 20, 0.03);
    window_size = Size(4,4);
    level = 5;
    lambda = 0.5;
}

void TrackerLK::Params::read( const cv::FileNode& fn ){
}

void TrackerLK::Params::write( cv::FileStorage& fs ) const{
}


void TrackerLKImpl::read( const cv::FileNode& fn )
{
    params.read( fn );
}

void TrackerLKImpl::write( cv::FileStorage& fs ) const
{
    params.write( fs );
}

TrackerLKImpl::TrackerLKImpl(const TrackerLK::Params &parameters): params( parameters )
{
    isInit = false;
}
bool TrackerLKImpl::initImpl( const Mat& image, const Rect2d& boundingBox )
{
    if(image.channels() ==3)
        cvtColor(image, lastImg, CV_RGB2GRAY);
    else
        lastImg = image;
    lastbox = boundingBox;
    return true;
}
bool TrackerLKImpl::reinitImpl( const Mat& image, Rect2d& boundingBox ){
    lastbox = boundingBox;
}

bool TrackerLKImpl::updateImpl( const Mat& image, Rect2d& boundingBox )
{
    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;
    cv::Rect tbb;
    cv::Mat image_gray = image;
    bool tracked  = false;
    bbPoints(points1,lastbox);
    if (points1.size()<1){
        tracked=false;
        return false;
    }
    vector<Point2f> points = points1;
    //Frame-to-frame tracking with forward-backward error cheking
    tracked = trackf2f(lastImg,image_gray,points,points2);
    if (tracked){
        bbPredict(points,points2,lastbox,tbb);
        if (getFB()>10 || tbb.x>image_gray.cols ||  tbb.y>image_gray.rows || tbb.br().x < 1 || tbb.br().y <1){
            tracked = false;
        }
    }
    if(tracked)
    {
       boundingBox = tbb;
       lastbox =tbb;
    }
    swap(lastImg,image_gray);
    return tracked;
}

bool TrackerLKImpl::trackf2f(const Mat& img1, const Mat& img2,vector<Point2f> &points1, vector<cv::Point2f> &points2){
    //TODO!:implement c function cvCalcOpticalFlowPyrLK() or Faster tracking function
    //Forward-Backward tracking
    status.clear();similarity.clear();FB_status.clear();FB_error.clear();
    calcOpticalFlowPyrLK( img1,img2, points1, points2, status,similarity, params.window_size, params.level, params.term_criteria, params.lambda, 0);
    calcOpticalFlowPyrLK( img2,img1, points2, pointsFB, FB_status,FB_error, params.window_size, params.level, params.term_criteria, params.lambda, 0);
    //Compute the real FB-error
    for( int i= 0; i<points1.size(); ++i ){
        FB_error[i] = norm(pointsFB[i]-points1[i]);
    }
    //Filter out points with FB_error[i] > median(FB_error) && points with sim_error[i] > median(sim_error)
    normCrossCorrelation(img1,img2,points1,points2);
    return filterPts(points1,points2);
}

void TrackerLKImpl::normCrossCorrelation(const Mat& img1,const Mat& img2, vector<Point2f>& points1, vector<Point2f>& points2) {
    Mat rec0(10,10,CV_8U);
    Mat rec1(10,10,CV_8U);
    Mat res(1,1,CV_32F);

    for (int i = 0; i < points1.size(); i++) {
        if (status[i] == 1) {
            getRectSubPix( img1, Size(10,10), points1[i],rec0 );
            getRectSubPix( img2, Size(10,10), points2[i],rec1);
            matchTemplate( rec0,rec1, res, CV_TM_CCOEFF_NORMED);
            similarity[i] = ((float *)(res.data))[0];

        } else {
            similarity[i] = 0.0;
        }
    }
    rec0.release();
    rec1.release();
    res.release();
}

float TrackerLKImpl::median(std::vector<float> v)
{
    int n = floor(v.size() / 2);
    nth_element(v.begin(), v.begin()+n, v.end());
    return v[n];
}

bool TrackerLKImpl::filterPts(vector<Point2f>& points1,vector<Point2f>& points2){
    //Get Error Medians
    simmed = median(similarity);
    size_t i, k;
    for( i=k = 0; i<points2.size(); ++i ){
        if( !status[i])
            continue;
        if(similarity[i]> simmed){
            points1[k] = points1[i];
            points2[k] = points2[i];
            FB_error[k] = FB_error[i];
            k++;
        }
    }
    if (k==0)
        return false;
    points1.resize(k);
    points2.resize(k);
    FB_error.resize(k);

    fbmed = median(FB_error);
    for( i=k = 0; i<points2.size(); ++i ){
        if( !status[i])
            continue;
        if(FB_error[i] <= fbmed){
            points1[k] = points1[i];
            points2[k] = points2[i];
            k++;
        }
    }
    points1.resize(k);
    points2.resize(k);
    if (k>0)
        return true;
    else
        return false;
}

void TrackerLKImpl::bbPoints(vector<cv::Point2f>& points,const Rect& bb){
    int max_pts=10;
    int margin_h=0;
    int margin_v=0;
    int stepx = ceil((bb.width-2*margin_h)/max_pts);
    int stepy = ceil((bb.height-2*margin_v)/max_pts);
    if(stepx==0 || stepy==0 ) return;
    for (int y=bb.y+margin_v;y<bb.y+bb.height-margin_v;y+=stepy){
        for (int x=bb.x+margin_h;x<bb.x+bb.width-margin_h;x+=stepx){
            points.push_back(Point2f(x,y));
        }
    }
}

void TrackerLKImpl::bbPredict(const vector<cv::Point2f>& points1,const vector<cv::Point2f>& points2,
                              const Rect& bb1,Rect& bb2)    {
    int npoints = (int)points1.size();
    vector<float> xoff(npoints);
    vector<float> yoff(npoints);
    // printf("tracked points : %d\n",npoints);
    for (int i=0;i<npoints;i++){
        xoff[i]=points2[i].x-points1[i].x;
        yoff[i]=points2[i].y-points1[i].y;
    }
    float dx = median(xoff);
    float dy = median(yoff);
    float s;
    if (npoints>1){
        vector<float> d;
        d.reserve(npoints*(npoints-1)/2);
        for (int i=0;i<npoints;i++){
            for (int j=i+1;j<npoints;j++){
                d.push_back(norm(points2[i]-points2[j])/norm(points1[i]-points1[j]));
            }
        }
        s = median(d);
    }
    else {
        s = 1.0;
    }
    float s1 = 0.5*(s-1)*bb1.width;
    float s2 = 0.5*(s-1)*bb1.height;
    //printf("s= %f s1= %f s2= %f \n",s,s1,s2);
    bb2.x = round( bb1.x + dx -s1);
    bb2.y = round( bb1.y + dy -s2);
    bb2.width = round(bb1.width*s);
    bb2.height = round(bb1.height*s);
}


}

