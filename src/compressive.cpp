#include "compressive.h"
using namespace std;
using namespace cv;
compressive::compressive()
{
    initCompressive();
}
void compressive::initCompressive()
{
    featureMinNumRect = 2;											//每个卷积核中的矩形数下限
    featureMaxNumRect = 14;//4;	// number of rectangle from 2 to 4	//每个卷积核中的矩形数上限
    featureNum = 100;	// number of all weaker classifiers, i.e,feature pool //低维特征数//4;	// radical scope of positive samples	//当目标在此范围内偏移时，都认为是正样本，更新分类器时使用
    learnRate = 0.85;//0.85f;	// Learning rate parameter//yqh: the weight to remember the last param 这个值太大会导致第一帧直接飞掉
    muPositive = vector<float>(featureNum, 0.0f);
    muNegative = vector<float>(featureNum, 0.0f);
    sigmaPositive = vector<float>(featureNum, 1.0f);
    sigmaNegative = vector<float>(featureNum, 1.0f);
    ratioTrained = false;
    milTrained   = false;

}

/*Description: compute Haar features
  Arguments:
  -_objectBox: [x y width height] object rectangle
  -_numFeature: total number of features.The default is 50.
  //离线生成随机投影矩阵，由featureNum个卷积核级联组成，每个卷积核由featureMinNumRect到featureMaxNumRect个矩形块来表示
*/
void compressive::prepareCompressiveScale(const std::vector<cv::Size>& scales)// with scale norm
{
    Compressivefeatures.resize(scales.size());
    CompressivefeaturesWeight.resize(scales.size());
    for(int j = 0 ; j <scales.size() ; j++)
    {
        Compressivefeatures[j]=   vector< vector<Feature> >(featureNum, vector<Feature>());
        CompressivefeaturesWeight[j]=  vector< vector<float> >(featureNum, vector<float>());
    }
    int numRect;
    float weightTemp,weightTemp2;
    float x1f,y1f;
    float xstart,ystart,xend,yend;
    int x1, x2, y1, y2;
    cv::Size minisize = scales[0];

    for (int i=0; i<featureNum; i++)
    {
        numRect = cvFloor(CTrng.uniform((double)featureMinNumRect, (double)featureMaxNumRect));
        for (int currRect=0; currRect<numRect; currRect++) //Rect number
        {
            bool nogood =true;
            while(nogood)
            {
                x1f = xstart = (float)CTrng.uniform(0.0,1.0);
                y1f = ystart = (float)CTrng.uniform(0.0,1.0);
                xend = (float)CTrng.uniform(0.0,1.0);
                yend = (float)CTrng.uniform(0.0,1.0);
                if(xstart > xend)
                {
                    xstart = xend;
                    xend   = x1f;
                }
                if(ystart > yend)
                {
                    ystart = yend;
                    yend   = y1f;
                }
                x1 =  xstart * minisize.width;
                y1 =  ystart * minisize.height;
                x2 =  xend * minisize.width;
                y2 =  yend * minisize.height;
                if(x1 < minisize.width -3 && y1 <minisize.height-3 && (x2-x1)>=2 && (y2-y1) >=2 )
                    nogood = false;
            }
            weightTemp = (float)pow(-1.0, cvFloor(CTrng.uniform(0.0, 2.0))) / sqrt(float(numRect));

            for (int s=0;s<scales.size();s++){//在不同的尺度上，放大了 scales[s].width
                x1 = int(xstart * scales[s].width);
                y1 = int(ystart * scales[s].height);
                x2 = int(xend * scales[s].width);
                y2 = int(yend * scales[s].height);
                Compressivefeatures[s][i].push_back(Feature(x1,y1,x2,y2));
                //Normalization  double(minisize.area())/double(scales[s].area()) == scale^2
                weightTemp2 =  weightTemp *  1.0 /double(scales[s].area()); //double(minisize.area())
                CompressivefeaturesWeight[s][i].push_back(weightTemp2);
            }
        }
    }
}
// Compute the features of samples
void compressive::getFeatureCompressiveScale(const cv::Mat& image, const int& scale_idx, std::vector<float>& ct_feature )
{
    std::vector< std::vector<Feature> > & cts = Compressivefeatures[scale_idx];
    std::vector < vector<float> >  & ct_weights = CompressivefeaturesWeight[scale_idx];
    ct_feature.resize(featureNum);
    float tempValue;
    for (int i=0; i<featureNum; i++) //featureNum
    {
        tempValue = 0.0f;
        for (size_t currRect=0; currRect<cts[i].size(); currRect++)//特征
        {
            tempValue += ct_weights[i][currRect] *
                    (image.at<int>(cts[i][currRect].y1, cts[i][currRect].x1) +
                     image.at<int>(cts[i][currRect].y2, cts[i][currRect].x2) -
                     image.at<int>(cts[i][currRect].y1, cts[i][currRect].x2) -
                     image.at<int>(cts[i][currRect].y2, cts[i][currRect].x1));
        }
        ct_feature[i]= tempValue;
    }
}

void compressive::prepareCompressive(const cv::Size& scales) // Compressive Tracking Object detection  for pyramids
{
    Compressivefeatures.resize(1);
    CompressivefeaturesWeight.resize(1);
    Compressivefeatures[0]=   vector< vector<Feature> >(featureNum, vector<Feature>());
    CompressivefeaturesWeight[0]=  vector< vector<float> >(featureNum, vector<float>());
    int numRect;
    float weightTemp,weightTemp2;
    float x1f,y1f;
    float xstart,ystart,xend,yend;
    int x1, x2, y1, y2;
    cv::Size minisize = scales;

    for (int i=0; i<featureNum; i++)
    {
        numRect = cvFloor(CTrng.uniform((double)featureMinNumRect, (double)featureMaxNumRect));
        for (int currRect=0; currRect<numRect; currRect++)
        {
            bool nogood =true;
            while(nogood)
            {
                x1f = xstart = (float)CTrng.uniform(0.0,1.0);
                y1f = ystart = (float)CTrng.uniform(0.0,1.0);
                xend = (float)CTrng.uniform(0.0,1.0);
                yend = (float)CTrng.uniform(0.0,1.0);
                if(xstart > xend)
                {
                    xstart = xend;
                    xend   = x1f;
                }
                if(ystart > yend)
                {
                    ystart = yend;
                    yend   = y1f;
                }
                x1 =  xstart * minisize.width;
                y1 =  ystart * minisize.height;
                x2 =  xend * minisize.width;
                y2 =  yend * minisize.height;
                if(x1 < minisize.width -3 && y1 <minisize.height-3 && (x2-x1)>=2 && (y2-y1) >=2 )
                    nogood = false;
            }
            weightTemp = (float)pow(-1.0, cvFloor(CTrng.uniform(0.0, 2.0))) / sqrt(float(numRect));
            Compressivefeatures[0][i].push_back(Feature(x1,y1,x2,y2));
            weightTemp2 =  weightTemp *  1.0 /double(scales.area()); //double(minisize.area())
            CompressivefeaturesWeight[0][i].push_back(weightTemp2);
        }
    }
}
void compressive::getFeatureCompressive(const cv::Mat& image, std::vector<float>& ct_feature)  //for pyramids
{

    std::vector< std::vector<Feature> > & cts   = Compressivefeatures[0];
    std::vector < vector<float> >  & ct_weights = CompressivefeaturesWeight[0];
    ct_feature.resize(featureNum);
    float tempValue;
    for (int i=0; i<featureNum; i++) //featureNum
    {
        tempValue = 0.0f;
        for (size_t currRect=0; currRect<cts[i].size(); currRect++)//特征
        {
            tempValue += ct_weights[i][currRect] *
                    (image.at<int>(cts[i][currRect].y1, cts[i][currRect].x1) +
                     image.at<int>(cts[i][currRect].y2, cts[i][currRect].x2) -
                     image.at<int>(cts[i][currRect].y1, cts[i][currRect].x2) -
                     image.at<int>(cts[i][currRect].y2, cts[i][currRect].x1));
        }
        ct_feature[i]= tempValue;
    }
}
void  compressive::radioClassifier(std::vector<float>& features, float& radio)
{
        float pPos(0);
        float pNeg(0);
        float sumPositive(0);
        float sumNegative(0);
        for (int i=0; i<featureNum; i++)
        {
            pPos = exp((features[i]-muPositive[i])*(features[i]-muPositive[i])  /
                       -(2.0f*sigmaPositive[i]*sigmaPositive[i]+1e-30) )
                    / (sigmaPositive[i]+1e-30);
            pNeg = exp( (features[i]-muNegative[i])*(features[i]-muNegative[i]) /
                        -(2.0f*sigmaNegative[i]*sigmaNegative[i]+1e-30))
                       / (sigmaNegative[i]+1e-30);
            sumPositive += log(pPos+1e-30) ;
            sumNegative += - log(pNeg+1e-30);
        }
        radio = sumPositive + sumNegative;

}

void  compressive::radioClassifier(std::vector< std::vector<float> >& ffeatures, std::vector<float>& radio)
{
        float pPos, pNeg,sumPositive,sumNegative;
        radio.resize(ffeatures.size());
        for(int k = 0 ; k < ffeatures.size(); k++)
        {
            pPos = pNeg = sumNegative = sumPositive = 0;
            std::vector<float> & features  = ffeatures[k];
            for (int i=0; i<featureNum; i++)
            {
                pPos = exp((features[i]-muPositive[i])*(features[i]-muPositive[i])  /
                           -(2.0f*sigmaPositive[i]*sigmaPositive[i]+1e-30) )
                        / (sigmaPositive[i]+1e-30);
                pNeg = exp( (features[i]-muNegative[i])*(features[i]-muNegative[i]) /
                            -(2.0f*sigmaNegative[i]*sigmaNegative[i]+1e-30))
                        / (sigmaNegative[i]+1e-30);
                sumPositive += log(pPos+1e-30) ;
                sumNegative += - log(pNeg+1e-30);
            }
            radio[k] = sumPositive + sumNegative;
        }
}
void  compressive::prepareData( std::vector< std::vector<float> >& posfeatures,
                                std::vector< std::vector<float> >& negfeatures,
                                cv::Mat & positiveStates, cv::Mat & negativeStates)
{
    CV_Assert((posfeatures.size()!=0 || negfeatures.size()!=0));
    positiveStates.create(posfeatures.size(),featureNum,  CV_32F);
    negativeStates.create(negfeatures.size(),featureNum,  CV_32F);

    unsigned int j = 0;
    int len = positiveStates.rows * positiveStates.cols;
    float* posdata = reinterpret_cast<float*>(positiveStates.data);
    for(unsigned int i = 0; i < len; i += featureNum, ++j)
        std::copy(posfeatures[j].begin(),posfeatures[j].end(),posdata + i);

    j = 0;
    len = negativeStates.rows * negativeStates.cols;
    float* negdata = reinterpret_cast<float*>(negativeStates.data);
    for(unsigned int i = 0; i < len; i += featureNum, ++j)
        std::copy(negfeatures[j].begin(),negfeatures[j].end(),negdata + i);

}
void  compressive::radioclassifierUpdate(std::vector< std::vector<float> >& posfeatures,std::vector< std::vector<float> >& negfeatures)
{
    Mat positiveStates;
    Mat negativeStates;
    prepareData(posfeatures,negfeatures,positiveStates,negativeStates);
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

void compressive::MilClassifierUpdate(std::vector< std::vector<float> >& posfeatures,
                                    std::vector< std::vector<float> >& negfeatures)
{
    Mat positiveStates;
    Mat negativeStates;
    prepareData(posfeatures,negfeatures,positiveStates,negativeStates);
    if(!milTrained)
    {
        cv::ClfMilBoost::Params params;
        params._lRate =    learnRate;
        params._numFeat =  featureNum;
        params._numSel = 50;
        MilBoost.init(params);
        milTrained = true;
    }
    MilBoost.update(positiveStates, negativeStates );
}
void compressive::Milclassify(std::vector< std::vector<float> >& features, std::vector<float> & prob)
{
    if(features.size()==0) return;
    prob.clear();
    Mat positiveStates;
    Mat negativeStates;
    std::vector< std::vector<float> > negfeatures;
    prepareData(features,negfeatures,positiveStates,negativeStates);
    prob = MilBoost.classify(positiveStates);
}
