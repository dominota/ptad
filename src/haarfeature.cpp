#include "haarfeature.h"
using namespace std;
using namespace cv;
HaarFeature::HaarFeature()
{
    learningAlgorithm = BoostingMILClassifierAlgorithm;
    featureNum = 250;
    learnRate = 0.85; //0.85f;
    Miltrained = false;
    ratioTrained = false;
    boostingTrained = false;
    initHaarFeature();
}
void HaarFeature::initHaarFeature(int learningMethod)
{
    learningAlgorithm = learningMethod;
    numClassifiers = 50;
    iterationInit = 50;
    weakclassifiersfactors = 10;
    featureSetNumFeatures = ( numClassifiers * weakclassifiersfactors ) + iterationInit;
    numBaseClassifier = numClassifiers;
    sampleROI = cv::Rect(0,0,640,480);
    featureSet = Ptr<features::TrackerFeatureSet>( new features::TrackerFeatureSet() );

    boostingTrained = false;
    ratioTrained = false;

    if(learningAlgorithm == AdaBoostinglassifierAlgorithm)  //Adaboosting feature must be considered alone;
        featureNum = featureSetNumFeatures;
    if(learningAlgorithm == RatioClassifierAlgorithm)
    {
        featureNum = 250;
        muPositive = vector<float>(featureNum, 0.0f);
        muNegative = vector<float>(featureNum, 0.0f);
        sigmaPositive = vector<float>(featureNum, 1.0f);
        sigmaNegative = vector<float>(featureNum, 1.0f);
    }
}

void  HaarFeature::radioClassifier(cv::Mat & positiveStates,std::vector<float> & prob) // row feature
{
    float pPos, pNeg,sumPositive,sumNegative;
    prob.resize(positiveStates.rows);
    for(int  j = 0 ; j < positiveStates.rows ; j++)
    {

        pPos = pNeg = sumNegative = sumPositive = 0;
        cv::Mat features = positiveStates.row(j);
        for (int i=0; i<featureNum; i++)
        {
            float tmp = features.at<float>(i);
            pPos = exp((tmp-muPositive[i])*(tmp-muPositive[i]) / -(2.0f*sigmaPositive[i]*sigmaPositive[i]+1e-30) )
                    / (sigmaPositive[i]+1e-30);
            pNeg = exp( (tmp-muNegative[i])*(tmp-muNegative[i]) / -(2.0f*sigmaNegative[i]*sigmaNegative[i]+1e-30))
                    / (sigmaNegative[i]+1e-30);
            sumPositive += log(pPos+1e-30) ;
            sumNegative += - log(pNeg+1e-30);
        }
        prob[j] = sumPositive + sumNegative;
    }
}

void  HaarFeature::radioClassifier(std::vector<cv::Mat> & positiveStates,std::vector<float> & prob) // row feature
{
    float pPos, pNeg,sumPositive,sumNegative;
    prob.resize(positiveStates.size());
    for(int  j = 0 ; j < positiveStates.size() ; j++)
    {

        pPos = pNeg = sumNegative = sumPositive = 0;
        cv::Mat features = positiveStates[j];
        for (int i=0; i<featureNum; i++)
        {
            float tmp = features.at<float>(i);
            pPos = exp((tmp-muPositive[i])*(tmp-muPositive[i]) / -(2.0f*sigmaPositive[i]*sigmaPositive[i]+1e-30) )
                    / (sigmaPositive[i]+1e-30);
            pNeg = exp( (tmp-muNegative[i])*(tmp-muNegative[i]) / -(2.0f*sigmaNegative[i]*sigmaNegative[i]+1e-30))
                    / (sigmaNegative[i]+1e-30);
            sumPositive += log(pPos+1e-30) ;
            sumNegative += - log(pNeg+1e-30);
        }
        prob[j] = sumPositive + sumNegative;
    }
}


void  HaarFeature::ratioclassifierUpdate(cv::Mat & positiveStates, cv::Mat & negativeStates)
{
    Scalar posmuTemp,possigmaTemp,negmuTemp,negsigmaTemp;
    if(!ratioTrained)
    {
        ratioTrained = true;
        for (int i=0; i<featureNum; i++)
        {
            cv::meanStdDev(positiveStates.col(i), posmuTemp, possigmaTemp);
            cv::meanStdDev(negativeStates.col(i), negmuTemp, negsigmaTemp);
            muPositive[i] =  posmuTemp.val[0];
            muNegative[i] =  negmuTemp.val[0];
            sigmaPositive[i] = possigmaTemp.val[0];
            sigmaNegative[i] = negsigmaTemp.val[0];
        }
    }
    else
    {
        for (int i=0; i<featureNum; i++)
        {
            cv::meanStdDev(positiveStates.col(i), posmuTemp, possigmaTemp);
            cv::meanStdDev(negativeStates.col(i), negmuTemp, negsigmaTemp);
            muPositive[i] =  posmuTemp.val[0];
            muNegative[i] =  negmuTemp.val[0];
            sigmaPositive[i] = possigmaTemp.val[0];
            sigmaNegative[i] = negsigmaTemp.val[0];

            sigmaPositive[i] = (float)sqrt(learnRate*sigmaPositive[i]*sigmaPositive[i]+
                                           (1.0f-learnRate)*possigmaTemp.val[0]*possigmaTemp.val[0]
                    + learnRate*(1.0f-learnRate)*(muPositive[i]-posmuTemp.val[0])*(muPositive[i]-posmuTemp.val[0]));	// equation 6 in paper
            muPositive[i] = muPositive[i]*learnRate + (1.0f-learnRate)*posmuTemp.val[0];	// equation 6 in paper


            sigmaNegative[i] = (float)sqrt(learnRate*sigmaNegative[i]*sigmaNegative[i]+
                                           (1.0f-learnRate)*negsigmaTemp.val[0]*negsigmaTemp.val[0]
                    + learnRate*(1.0f-learnRate)*(muNegative[i]-negmuTemp.val[0])*(muNegative[i]-negmuTemp.val[0]));	// equation 6 in paper
            muNegative[i] = muNegative[i]*learnRate + (1.0f-learnRate)*negmuTemp.val[0];	// equation 6 in paper
        }
    }
}
void   HaarFeature::ratioclassifierUpdate(std::vector<cv::Mat> & positiveStates, std::vector<cv::Mat> & negativeStates)
{
    int posCounter = positiveStates.size();
    int negCounter = negativeStates.size();
    cv::Mat  positive,negative;
    positive.create( posCounter, featureNum, CV_32FC1 );
    negative.create( negCounter, featureNum, CV_32FC1 );
    float* posdata = reinterpret_cast<float*>(positive.data);
    float* negdata = reinterpret_cast<float*>(negative.data);
    for(unsigned int i = 0; i < posCounter; i++)
    {
        float* tmpdata = reinterpret_cast<float*>(positiveStates[i].data);
        std::copy(tmpdata, tmpdata + featureNum, posdata+i*featureNum);
    }
    for(unsigned int i = 0; i < negCounter; i++)
    {
        float* tmpdata = reinterpret_cast<float*>(negativeStates[i].data);
        std::copy(tmpdata, tmpdata + featureNum, negdata+i*featureNum);
    }
    ratioclassifierUpdate(positive,negative);
}




// only supprot row feature   one row reprents one feature
void  HaarFeature::MilClassifierUpdate(cv::Mat & positiveStates, cv::Mat & negativeStates)
{
    if( !Miltrained )
    {
        onlineMIL::ClfMilBoost::Params params;
        params._lRate =    learnRate;
        params._numFeat =  featureNum;
        params._numSel = 50;
        MilBoost.init(params);
        Miltrained = true;
    }
    MilBoost.update(positiveStates, negativeStates);
}

void  HaarFeature::MilClassifierUpdate(std::vector<cv::Mat> & positiveStates, std::vector<cv::Mat> & negativeStates)
{
    int posCounter = positiveStates.size();
    int negCounter = negativeStates.size();
    cv::Mat  positive,negative;
    positive.create( posCounter, featureNum, CV_32FC1 ); //row features
    negative.create( negCounter, featureNum, CV_32FC1 );


    float* posdata = reinterpret_cast<float*>(positive.data);
    float* negdata = reinterpret_cast<float*>(negative.data);

    for(unsigned int i = 0; i < posCounter; i++)
    {
        float* tmpdata = reinterpret_cast<float*>(positiveStates[i].data);
        std::copy(tmpdata, tmpdata + featureNum, posdata+i*featureNum);
    }

    for(unsigned int i = 0; i < negCounter; i++)
    {
        float* tmpdata = reinterpret_cast<float*>(negativeStates[i].data);
        std::copy(tmpdata, tmpdata + featureNum, negdata+i*featureNum);
    }
    MilClassifierUpdate(positive, negative);
}


void  HaarFeature::Milclassify(cv::Mat & States, std::vector<float> & prob)
{
    prob.clear();
    prob = MilBoost.classify(States);
}

void  HaarFeature::Milclassify(std::vector<cv::Mat> & States, std::vector<float> & prob){
    if(prob.size()!=States.size())
        prob = std::vector<float>(States.size(),0);
    MilBoost.classify(States,prob);
}

/*Description: compute Haar features
  Arguments:
  -_objectBox: [x y width height] object rectangle
  -_numFeature: total number of features.The default is 50.
  //离线生成随机投影矩阵，由featureNum个卷积核级联组成，每个卷积核由featureMinNumRect到featureMaxNumRect个矩形块来表示
*/
bool HaarFeature::prepareHaar(const cv::Size& scale)//准备Haar-feature,采用金字塔的方式提取特征
{
    scaleSize.clear();
    initPatchSize = scale; //
    features::TrackerFeatureHAAR::Params HAARparameters;
    HAARparameters.numFeatures = featureNum;
    HAARparameters.isIntegral = true;
    HAARparameters.rectSize = Size( static_cast<int>(scale.width), static_cast<int>(scale.height) );
    trackerFeature = Ptr<features::TrackerFeatureHAAR>( new features::TrackerFeatureHAAR( HAARparameters ) );
    if( !featureSet->addTrackerFeature( trackerFeature ) )
        return false;
    return true;

}

bool HaarFeature::prepareHaarScale(const std::vector<cv::Size>& scale)//准备Haar-feature,采用金字塔的方式提取特征
{
    if(scale.size() == 0)
        return false;
    scaleSize = scale;
    initPatchSize = scale[0];
    features::TrackerFeatureHAAR::Params HAARparameters;
    HAARparameters.numFeatures = featureNum;
    HAARparameters.isIntegral = true;
    HAARparameters.rectSize = Size( static_cast<int>(scale[0].width), static_cast<int>(scale[0].height) );
    trackerFeature = Ptr<features::TrackerFeatureHAAR>( new features::TrackerFeatureHAAR(HAARparameters,scale));
    if( !featureSet->addTrackerFeature( trackerFeature ) )
        return false;
    return true;

}
// Compute the features of samples
void HaarFeature::getFeatureHaar( std::vector<cv::Mat> & image,  cv::Mat& Features){
    featureSet->extraction(image);
    std::vector<cv::Mat> features =featureSet->getResponses();
    Features = features[0];
}

void HaarFeature::getFeatureHaar( std::vector<cv::Mat> & image,  std::vector<cv::Mat>& Features){
    featureSet->extractionCollection(image,Features);
    //std::cout<<Features[0]<<std::endl;
    //std::cout<<Features[1]<<std::endl;
}

void HaarFeature::getFeatureHaarScale( std::vector<cv::Mat> & image,  cv::Mat& Features, std::vector<int> scale ){
    if(image.size() != scale.size() ) return;
    featureSet->extraction(image,scale);
    std::vector<cv::Mat> features =featureSet->getResponses(); //featureSet 特征集合 Hog Haar LBP
    Features = features[0];
}

void HaarFeature::getFeatureHaarScale( std::vector<cv::Mat> & image,  std::vector<cv::Mat> & Features, std::vector<int> scale ){
    if(image.size() != scale.size() ) return;  // scale
    featureSet->extractionCollection(image,Features,scale);
}


void  HaarFeature::BoostingclassifierUpdate(std::vector<cv::Mat>& posfeatures,std::vector<cv::Mat> & negfeatures)
{
    // row and col features all ok
    if( !boostingTrained )
    {
        //this is the first time that the classifier is built
        int numWeakClassifier = numBaseClassifier * weakclassifiersfactors;

        bool useFeatureExchange = true;
        boostingClassifier = Ptr<onlineboosting::StrongClassifierDirectSelection>(
                    new onlineboosting::StrongClassifierDirectSelection( numBaseClassifier, numWeakClassifier,
                                                                         initPatchSize, sampleROI, useFeatureExchange, iterationInit ) );
        //init base classifiers
        boostingClassifier->initBaseClassifier();

        boostingTrained = true;
    }

    bool featureEx = boostingClassifier->getUseFeatureExchange();

    int posnums = posfeatures.size();
    int negnums = negfeatures.size();
    int samplesnums = posnums+negnums;

    replacedClassifier.clear();
    replacedClassifier.resize(samplesnums, -1 );
    swappedClassifier.clear();
    swappedClassifier.resize(samplesnums, -1 );


    for ( size_t i = 0; i < samplesnums / 2; i++ )
    {
        Mat res;
        int currentFg = -1;
        if( i < posnums)
        {
            res = posfeatures[i];
            currentFg = 1;
        }
        else
            res = negfeatures[i -posnums];

        // evey classifier for one dimension of a feature.
        boostingClassifier->update( res, currentFg ); //feature vector res
        if( featureEx )
        {
            replacedClassifier[i] = boostingClassifier->getReplacedClassifier();
            swappedClassifier[i]  = boostingClassifier->getSwappedClassifier();
            if( replacedClassifier[i] >= 0 && swappedClassifier[i] >= 0 )
                boostingClassifier->replaceWeakClassifier( replacedClassifier[i] );
        }
        else
        {
            replacedClassifier[i] = -1;
            swappedClassifier[i] = -1;
        }

        currentFg = -1;
        int mapPosition = (int)(i + samplesnums / 2);
        Mat res2;
        if( mapPosition < posnums)
        {
            res2  = posfeatures[i];
            currentFg = 1;
        }
        else
            res2 = negfeatures[mapPosition - posnums];
        boostingClassifier->update( res2, currentFg );
        if( featureEx )
        {
            replacedClassifier[mapPosition] = boostingClassifier->getReplacedClassifier();
            swappedClassifier[mapPosition] = boostingClassifier->getSwappedClassifier();
            if( replacedClassifier[mapPosition] >= 0 && swappedClassifier[mapPosition] >= 0 )
                boostingClassifier->replaceWeakClassifier( replacedClassifier[mapPosition] );
        }
        else
        {
            replacedClassifier[mapPosition] = -1;
            swappedClassifier[mapPosition] = -1;
        }
    }
}

void  HaarFeature::iterationInitBoosting(std::vector<cv::Mat>& posfeatures,std::vector<cv::Mat> & negfeatures)
{
    int posnums = posfeatures.size();
    int negnums = negfeatures.size();
    int samplesnums = posnums+negnums;
    for ( int i = 0; i < iterationInit; i++ )
    {
        features::TrackerFeatureHAAR::Params HAARparametersTmp;
        HAARparametersTmp.numFeatures = static_cast<int>( samplesnums );
        HAARparametersTmp.isIntegral = true;
        HAARparametersTmp.rectSize = Size( static_cast<int>(initPatchSize.width), static_cast<int>(initPatchSize.height) );
        Ptr<features::TrackerFeatureHAAR> trackerFeatureTmp = Ptr<features::TrackerFeatureHAAR>(
                    new features::TrackerFeatureHAAR( HAARparametersTmp,scaleSize ) );

        BoostingclassifierUpdate(posfeatures, negfeatures);

        for ( size_t j = 0; j < replacedClassifier.size(); j++ )
        {
            if( replacedClassifier[j] != -1 && swappedClassifier[j] != -1 )
            {
                trackerFeature.staticCast<features::TrackerFeatureHAAR>()->swapFeature( replacedClassifier[j], swappedClassifier[j] );
                trackerFeature.staticCast<features::TrackerFeatureHAAR>()->swapFeature( swappedClassifier[j], trackerFeatureTmp->getFeatureAt( (int)j ) );
            }
        }
    }
}

void  HaarFeature::BoostingClassify(std::vector<cv::Mat>& posfeatures, std::vector<float> & prob)
{
    boostingClassifier->classifySmooth( posfeatures, prob );
}
void  HaarFeature::BoostingUpdateInterface(std::vector<cv::Mat>& posfeatures,std::vector<cv::Mat> & negfeatures)
{
  if(boostingTrained)
  {
      int posnums = posfeatures.size();
      int negnums = negfeatures.size();
      int samplesnums = posnums+negnums;

      features::TrackerFeatureHAAR::Params HAARparametersTmp;
      HAARparametersTmp.numFeatures = static_cast<int>( samplesnums );
      HAARparametersTmp.isIntegral = true;
      HAARparametersTmp.rectSize = Size( static_cast<int>(initPatchSize.width), static_cast<int>(initPatchSize.height) );
      Ptr<features::TrackerFeatureHAAR> trackerFeatureTmp = Ptr<features::TrackerFeatureHAAR>(
                  new features::TrackerFeatureHAAR( HAARparametersTmp,scaleSize ) );

      BoostingclassifierUpdate(posfeatures,negfeatures);

      for ( size_t j = 0; j < replacedClassifier.size(); j++ )
      {
          if( replacedClassifier[j] != -1 && swappedClassifier[j] != -1 )
          {
              trackerFeature.staticCast<features::TrackerFeatureHAAR>()->swapFeature( replacedClassifier[j], swappedClassifier[j] );
              trackerFeature.staticCast<features::TrackerFeatureHAAR>()->swapFeature( swappedClassifier[j], trackerFeatureTmp->getFeatureAt( (int)j ) );
          }
      }
  }
  else
      iterationInitBoosting(posfeatures,negfeatures);

}

void  HaarFeature::classifier(std::vector<cv::Mat> & States,std::vector<float> & prob)
{
    switch (learningAlgorithm)
    {
    case RatioClassifierAlgorithm: radioClassifier(States,prob); break;
    case BoostingMILClassifierAlgorithm: Milclassify(States,prob); break;
    case AdaBoostinglassifierAlgorithm:  BoostingClassify(States,prob); break;
    case SVMClassifierAlgorithm: break;
    default: break;
    }
}

void  HaarFeature::classifierUpdate(std::vector<cv::Mat> & positiveStates, std::vector<cv::Mat> & negativeStates)
{
    switch (learningAlgorithm)
    {
    case RatioClassifierAlgorithm: ratioclassifierUpdate(positiveStates,negativeStates); break;
    case BoostingMILClassifierAlgorithm: MilClassifierUpdate(positiveStates,negativeStates); break;
    case AdaBoostinglassifierAlgorithm:  BoostingUpdateInterface(positiveStates,negativeStates); break;
    case SVMClassifierAlgorithm: break;
    default: break;
    }
}


void HaarFeature::BoostingClassify(cv::Mat& posfeatures, std::vector<float> & prob)
{
    boostingClassifier->classifySmooth( posfeatures, prob );
}

void HaarFeature::iterationInitBoosting(cv::Mat & posfeatures, cv::Mat & negfeatures)
{
    int posnums = posfeatures.cols;
    int negnums = negfeatures.cols;
    int samplesnums = posnums+negnums;
    for ( int i = 0; i < iterationInit; i++ )
    {
        //compute temp features
        features::TrackerFeatureHAAR::Params HAARparameters2;
        HAARparameters2.numFeatures = static_cast<int>( samplesnums );
        HAARparameters2.isIntegral = true;
        HAARparameters2.rectSize = Size( static_cast<int>(initPatchSize.width), static_cast<int>(initPatchSize.height) );
        Ptr<features::TrackerFeatureHAAR> trackerFeature2 = Ptr<features::TrackerFeatureHAAR>( new features::TrackerFeatureHAAR( HAARparameters2 ,scaleSize) );

        BoostingclassifierUpdate(posfeatures, negfeatures);

        for ( size_t j = 0; j < replacedClassifier.size(); j++ )
        {
            if( replacedClassifier[j] != -1 && swappedClassifier[j] != -1 )
            {
                trackerFeature.staticCast<features::TrackerFeatureHAAR>()->swapFeature( replacedClassifier[j], swappedClassifier[j] );
                trackerFeature.staticCast<features::TrackerFeatureHAAR>()->swapFeature( swappedClassifier[j], trackerFeature2->getFeatureAt( (int)j ) );
            }
        }
    }
}

void  HaarFeature::BoostingclassifierUpdate(cv::Mat & posfeatures, cv::Mat & negfeatures)
{
    if( !boostingTrained )
    {
        //this is the first time that the classifier is built
        int numWeakClassifier = numBaseClassifier * 10;

        bool useFeatureExchange = true;
        boostingClassifier = Ptr<onlineboosting::StrongClassifierDirectSelection>(
                    new onlineboosting::StrongClassifierDirectSelection( numBaseClassifier, numWeakClassifier,
                                                                         initPatchSize, sampleROI, useFeatureExchange, iterationInit ) );
        //init base classifiers
        boostingClassifier->initBaseClassifier();

        boostingTrained = true;
    }

    Mat positiveStates =  posfeatures;
    Mat negativeStates =  negfeatures;

    bool featureEx = boostingClassifier->getUseFeatureExchange();

    int posnums = positiveStates.cols;
    int negnums = negativeStates.cols;
    int samplesnums = posnums+negnums;

    replacedClassifier.clear();
    replacedClassifier.resize(samplesnums, -1 );
    swappedClassifier.clear();
    swappedClassifier.resize(samplesnums, -1 );


    for ( size_t i = 0; i < samplesnums / 2; i++ )
    {
        Mat res;
        int currentFg = -1;
        if( i < posnums)
        {
            res = positiveStates.col(i);
            currentFg = 1;
        }
        else
            res = negativeStates.col( i -posnums );

        boostingClassifier->update( res, currentFg );
        if( featureEx )
        {
            replacedClassifier[i] = boostingClassifier->getReplacedClassifier();
            swappedClassifier[i] = boostingClassifier->getSwappedClassifier();
            if( replacedClassifier[i] >= 0 && swappedClassifier[i] >= 0 )
                boostingClassifier->replaceWeakClassifier( replacedClassifier[i] );
        }
        else
        {
            replacedClassifier[i] = -1;
            swappedClassifier[i] = -1;
        }

        currentFg = -1;
        int mapPosition = (int)(i + samplesnums / 2);
        Mat res2;
        if( mapPosition < posnums)
        {
            res2 = positiveStates.col(mapPosition);
            currentFg = 1;
        }
        else
            res2 = negativeStates.col( mapPosition - posnums );
        boostingClassifier->update( res2, currentFg );
        if( featureEx )
        {
            replacedClassifier[mapPosition] = boostingClassifier->getReplacedClassifier();
            swappedClassifier[mapPosition] = boostingClassifier->getSwappedClassifier();
            if( replacedClassifier[mapPosition] >= 0 && swappedClassifier[mapPosition] >= 0 )
                boostingClassifier->replaceWeakClassifier( replacedClassifier[mapPosition] );
        }
        else
        {
            replacedClassifier[mapPosition] = -1;
            swappedClassifier[mapPosition] = -1;
        }
    }
}
