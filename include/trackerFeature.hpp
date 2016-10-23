/*M///////////////////////////////////////////////////////////////////////////////////////
 //
 //  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
 //
 //  By downloading, copying, installing or using the software you agree to this license.
 //  If you do not agree to this license, do not download, install,
 //  copy or use the software.
 //
 //
 //                           License Agreement
 //                For Open Source Computer Vision Library
 //
 // Copyright (C) 2013, OpenCV Foundation, all rights reserved.
 // Third party copyrights are property of their respective owners.
 //
 // Redistribution and use in source and binary forms, with or without modification,
 // are permitted provided that the following conditions are met:
 //
 //   * Redistribution's of source code must retain the above copyright notice,
 //     this list of conditions and the following disclaimer.
 //
 //   * Redistribution's in binary form must reproduce the above copyright notice,
 //     this list of conditions and the following disclaimer in the documentation
 //     and/or other materials provided with the distribution.
 //
 //   * The name of the copyright holders may not be used to endorse or promote products
 //     derived from this software without specific prior written permission.
 //
 // This software is provided by the copyright holders and contributors "as is" and
 // any express or implied warranties, including, but not limited to, the implied
 // warranties of merchantability and fitness for a particular purpose are disclaimed.
 // In no event shall the Intel Corporation or contributors be liable for any direct,
 // indirect, incidental, special, exemplary, or consequential damages
 // (including, but not limited to, procurement of substitute goods or services;
 // loss of use, data, or profits; or business interruption) however caused
 // and on any theory of liability, whether in contract, strict liability,
 // or tort (including negligence or otherwise) arising in any way out of
 // the use of this software, even if advised of the possibility of such damage.
 //
 //M*/
#ifndef __TRACKERFEATURE_HPP__
#define __TRACKERFEATURE_HPP__

#include "feature.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

using namespace cv;
namespace features
{


/** @brief Abstract base class for TrackerFeature that represents the feature.
 */
class CV_EXPORTS TrackerFeature
{
 public:
  virtual ~TrackerFeature();

  /** @brief Compute the features in the images collection
    @param images The images
    @param response The output response
     */
  void compute(  std::vector<Mat>& images, Mat& response, std::vector<int> scaleSize  = std::vector<int>());

  void compute(  std::vector<Mat>& images, std::vector<Mat> & response, std::vector<int> scaleSize  = std::vector<int>());
  /** @brief Create TrackerFeature by tracker feature type
    @param trackerFeatureType The TrackerFeature name

    The modes available now:

    -   "HAAR" -- Haar Feature-based

    The modes that will be available soon:

    -   "HOG" -- Histogram of Oriented Gradients features
    -   "LBP" -- Local Binary Pattern features
    -   "FEATURE2D" -- All types of Feature2D
     */
  static Ptr<TrackerFeature> create( const String& trackerFeatureType );

  /** @brief Identify most effective features
    @param response Collection of response for the specific TrackerFeature
    @param npoints Max number of features

    @note This method modifies the response parameter
     */
  virtual void selection( Mat& response, int npoints ) = 0;

  /** @brief Get the name of the specific TrackerFeature
     */
  String getClassName() const;

 protected:

  virtual bool computeImpl(  std::vector<Mat>& images, Mat& response , std::vector<int> scaleSize = std::vector<int>()) = 0;

  virtual bool computeImpl(  std::vector<Mat>& images, std::vector<Mat>& response , std::vector<int> scaleSize = std::vector<int>()) = 0;
  String className;
};

/************************************ Specific TrackerFeature Classes ************************************/

/**
 * \brief TrackerFeature based on Feature2D
 */
class CV_EXPORTS TrackerFeatureFeature2d : public TrackerFeature
{
 public:

  /**
   * \brief Constructor
   * \param detectorType string of FeatureDetector
   * \param descriptorType string of DescriptorExtractor
   */
  TrackerFeatureFeature2d( String detectorType, String descriptorType );

  ~TrackerFeatureFeature2d();

  void selection( Mat& response, int npoints );

 protected:

  bool computeImpl(  std::vector<Mat>& images, Mat& response, std::vector<int> scaleSize = std::vector<int>());
  bool computeImpl(  std::vector<Mat>& images, std::vector<Mat>& response , std::vector<int> scaleSize = std::vector<int>());
 private:

  std::vector<KeyPoint> keypoints;
};

/**
 * \brief TrackerFeature based on HOG
 */
class CV_EXPORTS TrackerFeatureHOG : public TrackerFeature
{
 public:

  TrackerFeatureHOG();

  ~TrackerFeatureHOG();

  void selection( Mat& response, int npoints );

 protected:

  bool computeImpl(  std::vector<Mat>& images, Mat& response, std::vector<int> scaleSize = std::vector<int>());
  bool computeImpl(  std::vector<Mat>& images, std::vector<Mat>& response , std::vector<int> scaleSize = std::vector<int>());
};

/** @brief TrackerFeature based on HAAR features, used by TrackerMIL and many others algorithms
@note HAAR features implementation is copied from apps/traincascade and modified according to MIL
 */
class CV_EXPORTS TrackerFeatureHAAR : public TrackerFeature
{
 public:
  struct CV_EXPORTS Params
  {
    Params();
    int numFeatures;  //!< # of rects
    Size rectSize;    //!< rect size
    bool isIntegral;  //!< true if input images are integral, false otherwise
  };

  /** @brief Constructor
    @param parameters TrackerFeatureHAAR parameters TrackerFeatureHAAR::Params
     */
  TrackerFeatureHAAR( const TrackerFeatureHAAR::Params &parameters = TrackerFeatureHAAR::Params(), std::vector<cv::Size> scaleSize = std::vector<cv::Size>());

  ~TrackerFeatureHAAR();

  /** @brief Compute the features only for the selected indices in the images collection
    @param selFeatures indices of selected features
    @param images The images
    @param response Collection of response for the specific TrackerFeature
     */
  bool extractSelected( const std::vector<int> selFeatures, const std::vector<Mat>& images, Mat& response );

//  bool extractSelected( const std::vector<int> selFeatures, const std::vector<Mat>& images, std::vector<Mat>& response );
  /** @brief Identify most effective features
    @param response Collection of response for the specific TrackerFeature
    @param npoints Max number of features

    @note This method modifies the response parameter
     */
  void selection( Mat& response, int npoints );

  /** @brief Swap the feature in position source with the feature in position target
  @param source The source position
  @param target The target position
 */
  bool swapFeature( int source, int target );

  /** @brief   Swap the feature in position id with the feature input
  @param id The position
  @param feature The feature
 */
  bool swapFeature( int id, CvHaarEvaluator::FeatureHaar& feature );

  /** @brief Get the feature in position id
    @param id The position
     */
  CvHaarEvaluator::FeatureHaar& getFeatureAt( int id );

 protected:
  bool computeImpl(  std::vector<Mat>& images, Mat& response, std::vector<int> scaleSize = std::vector<int>());
  bool computeImpl(  std::vector<Mat>& images, std::vector<Mat>& response , std::vector<int> scaleSize = std::vector<int>());
 private:

  Params params;
  Ptr<CvHaarEvaluator> featureEvaluator;
};

/**
 * \brief TrackerFeature based on LBP
 */
class CV_EXPORTS TrackerFeatureLBP : public TrackerFeature
{
 public:

  TrackerFeatureLBP();

  ~TrackerFeatureLBP();

  void selection( Mat& response, int npoints );

 protected:

  bool computeImpl(  std::vector<Mat>& images, Mat& response, std::vector<int> scaleSize = std::vector<int>());
  bool computeImpl(  std::vector<Mat>& images, std::vector<Mat>& response , std::vector<int> scaleSize = std::vector<int>());
};


} /* namespace cv */
#endif
