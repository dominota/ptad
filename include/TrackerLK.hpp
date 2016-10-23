#ifndef TRACKERLK_HPP
#define TRACKERLK_HPP

#endif // TRACKERLK_HPP
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>

#define BOILERPLATE_CODE(name,classname) \
    static Ptr<classname> createTracker(const classname::Params &parameters=classname::Params());\
    virtual ~classname(){};

namespace cv
{

class CV_EXPORTS TrackerAdditional : public virtual Algorithm
{
 public:

  virtual ~TrackerAdditional();

  /** @brief Initialize the tracker with a know bounding box that surrounding the target
    @param image The initial frame
    @param boundingBox The initial boundig box

    @return True if initialization went succesfully, false otherwise
     */
  CV_WRAP bool init( const Mat& image, const Rect2d& boundingBox );

  CV_WRAP bool reinit( const Mat& image, Rect2d& boundingBox );
  /** @brief Update the tracker, find the new most likely bounding box for the target
    @param image The current frame
    @param boundingBox The boundig box that represent the new target location, if true was returned, not
    modified otherwise

    @return True means that target was located and false means that tracker cannot locate target in
    current frame. Note, that latter *does not* imply that tracker has failed, maybe target is indeed
    missing from the frame (say, out of sight)
     */
  CV_WRAP bool update( const Mat& image, CV_OUT Rect2d& boundingBox );

  virtual void read( const FileNode& fn )=0;
  virtual void write( FileStorage& fs ) const=0;


 protected:

  virtual bool initImpl( const Mat& image, const Rect2d& boundingBox ) = 0;
  virtual bool updateImpl( const Mat& image, Rect2d& boundingBox ) = 0;
  virtual bool reinitImpl( const Mat& image,  Rect2d& boundingBox ) = 0;

  bool isInit;

};



class CV_EXPORTS TrackerLK : public TrackerAdditional
{
public:
 struct CV_EXPORTS Params
 {
   Params();
   cv::Size window_size;
   int level;
   float lambda;
   cv::TermCriteria term_criteria;
   /**
    * \brief Read parameters from file
    */
   void read( const FileNode& fn );

   /**
    * \brief Write parameters in a file
    */
   void write( FileStorage& fs ) const;
 };

public:
    BOILERPLATE_CODE("LK",TrackerLK);
};

}
