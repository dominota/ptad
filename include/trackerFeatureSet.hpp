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
#ifndef __TRACKERFEATURESET_HPP__
#define __TRACKERFEATURESET_HPP__
#include "trackerFeature.hpp"
using namespace cv;
namespace features
{

/** @brief Class that manages the extraction and selection of features

@cite AAM Feature Extraction and Feature Set Refinement (Feature Processing and Feature Selection).
See table I and section III C @cite AMVOT Appearance modelling -\> Visual representation (Table II,
section 3.1 - 3.2)

TrackerFeatureSet is an aggregation of TrackerFeature

@sa
   TrackerFeature

 */
class CV_EXPORTS TrackerFeatureSet
{
 public:

  TrackerFeatureSet();

  ~TrackerFeatureSet();

  /** @brief Extract features from the images collection
    @param images The input images
     */
  void extraction(  std::vector<Mat>& images,std::vector<int> scaleSize = std::vector<int>() );

  /** @brief Extract features from the images collection
    @param images The input images
     */
  void extractionCollection(  std::vector<Mat>& images,std::vector<int> scaleSize = std::vector<int>() );
  /** @brief Identify most effective features for all feature types (optional)
     */

  void extraction(  std::vector<Mat>& images,cv::Mat & Features,std::vector<int> scaleSize = std::vector<int>() );

  /** @brief Extract features from the images collection
    @param images The input images
     */
  void extractionCollection(  std::vector<Mat>& images, std::vector<cv::Mat> & Features, std::vector<int> scaleSize = std::vector<int>());

  void selection();

  /** @brief Remove outliers for all feature types (optional)
     */
  void removeOutliers();

  /** @brief Add TrackerFeature in the collection. Return true if TrackerFeature is added, false otherwise
    @param trackerFeatureType The TrackerFeature name

    The modes available now:

    -   "HAAR" -- Haar Feature-based

    The modes that will be available soon:

    -   "HOG" -- Histogram of Oriented Gradients features
    -   "LBP" -- Local Binary Pattern features
    -   "FEATURE2D" -- All types of Feature2D

    Example TrackerFeatureSet::addTrackerFeature : :
    @code
        //sample usage:

        Ptr<TrackerFeature> trackerFeature = new TrackerFeatureHAAR( HAARparameters );
        featureSet->addTrackerFeature( trackerFeature );

        //or add CSC sampler with default parameters
        //featureSet->addTrackerFeature( "HAAR" );
    @endcode
    @note If you use the second method, you must initialize the TrackerFeature
     */
  bool addTrackerFeature( String trackerFeatureType );

  /** @overload
    @param feature The TrackerFeature class
    */
  bool addTrackerFeature( Ptr<TrackerFeature>& feature );

  /** @brief Get the TrackerFeature collection (TrackerFeature name, TrackerFeature pointer)
     */
  const std::vector<std::pair<String, Ptr<TrackerFeature> > >& getTrackerFeature() const;

  /** @brief Get the responses

    @note Be sure to call extraction before getResponses Example TrackerFeatureSet::getResponses : :
     */
  const std::vector<Mat>& getResponses() const;

  const std::vector<std::vector<Mat> >& getResponseCollections() const;


 private:
  void clearResponses();
  void clearResponseCollections();
  bool blockAddTrackerFeature;

  std::vector<std::pair<String, Ptr<TrackerFeature> > > features;  //list of features
  std::vector<Mat> responses;				            //list of response after compute
  std::vector<std::vector<Mat> > responseCollections;				//list of response after compute
};


} /* namespace cv */
#endif
