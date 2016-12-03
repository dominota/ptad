#ifndef HOGFEATURE_H
#define HOGFEATURE_H

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "onlineBoosting.hpp"
#include "onlineMIL.hpp"
#include "FHOG.hpp"
#include "feature.hpp"
#include "trackerFeature.hpp"
#include "trackerFeatureSet.hpp"
#include <opencv2/tracking.hpp>
#include <linear.h>

#include "vector.h"
#include "lasvm.h"

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL

using namespace cv;
class FHOGFeature
{
public:
    enum
    {
        RatioClassifier = 0,
        BoostingMILClassifier = 1,
        AdaBoostinglassifier = 2,
        SVMClassifier = 3
    };

    enum
    {
        OnLineLearning = 0,
        IncrementWithPersistence = 1,
        IncrementWithoutPersistence = 2
    };
public:
    FHOGFeature();
    void initHOGFeature(int learningMethod = BoostingMILClassifier);
    void getFeatureHog(const std::vector<cv::Mat>& image,  std::vector<cv::Mat> & ct_feature);
    void getFeatureHog(const cv::Mat & image,  cv::Mat& Features);
    void getSelectFeature(const std::vector<cv::Mat> & image,  std::vector<cv::Mat>& Features);

    void ratioclassifierUpdate(cv::Mat & positiveStates, cv::Mat & negativeStates);
    void ratioclassifierUpdate(std::vector<cv::Mat> & positiveStates, std::vector<cv::Mat> & negativeStates);
    void ratioClassifier(cv::Mat & positiveStates,std::vector<float> & prob);
    void ratioClassifier(std::vector<cv::Mat> & positiveStates,std::vector<float> & prob);
    void MilClassifierUpdate(cv::Mat & positiveStates, cv::Mat & negativeStates);
    void MilClassifierUpdate(std::vector<cv::Mat> & positiveStates, std::vector<cv::Mat> & negativeStates);
    void Milclassify(cv::Mat & States, std::vector<float> & prob);
    void Milclassify(std::vector<cv::Mat> & States, std::vector<float> & prob);

    void prepareSVMdata(std::vector<cv::Mat>& posfeatures, std::vector<cv::Mat>& negfeatures);
    void SVMClassifyUpdate(std::vector<cv::Mat> & positiveStates, std::vector<cv::Mat> & negativeStates);
    void SVMclassify(std::vector<cv::Mat> & States, std::vector<float> & prob);\

    void prepareLaSVMdata(std::vector<cv::Mat>& posfeatures, std::vector<cv::Mat>& negfeatures);
    void prepareLaSVMdata(std::vector<cv::Mat>& posfeatures);
    void LaSVMClassifyUpdate(std::vector<cv::Mat> & positiveStates, std::vector<cv::Mat> & negativeStates);
    void LaSVMclassify(std::vector<cv::Mat> & States, std::vector<float> & prob);
    void saveModel();
    void finish(lasvm_t *sv);
    int count_svs();
    void make_old(int val);
    int  select(lasvm_t *sv) ;

    void classifier(std::vector<cv::Mat> & States,std::vector<float> & prob);
    void classifierUpdate(std::vector<cv::Mat> & positiveStates, std::vector<cv::Mat> & negativeStates);

public:


    float learnRate;
    int featureNum;
    int learningAlgorithm;
    HogFeature fhog_;

    // Ratio learning
    bool ratioTrained;
    std::vector<float> muPositive;
    std::vector<float> sigmaPositive;
    std::vector<float> muNegative;
    std::vector<float> sigmaNegative;


    // MIL learning
    onlineMIL::ClfMilBoost MilBoost; //MIL learning algorithms
    bool Miltrained;


    //SVM learning
    bool  SVMtrained;                       //SVM learning algorithms
    struct feature_node *x_space;
    struct parameter param;
    struct problem prob_;
    struct model* model_;
    struct model* initial_model;
    struct feature_node *predict_x_space;

       // labels
    std::vector <double> kparam;           // kernel parameters
    std::vector <double> alpha;            // alpha_i, SV weights
    double b0;                        // threshold

    int use_b0;                     // use threshold via constraint \sum a_i y_i =0
    int selection_type;        // RANDOM, GRADIENT or MARGIN selection strategies
    int optimizer; // strategy of optimization
    double C;                       // C, penalty on errors
    double C_neg;                   // C-Weighting for negative examples
    double C_pos;                   // C-Weighting for positive examples
    int epochs;                     // epochs of online learning
    int candidates;				  // number of candidates for "active" selection process
    double deltamax;			  // tolerance for performing reprocess step, 1000=1 reprocess only
    std::vector <double> select_size;      // Max number of SVs to take with selection strategy (for early stopping)


    double epsgr;                       // tolerance on gradients
    int history_;
    int accumulateSamples_;
    int current_samples_;
    int learningSamples_;
    std::vector <int> iold, inew;// sets of old (already seen) points + new (unseen) points
    int incrLearningMode_;
    long long cache_size;               // 256Mb cache size as default
    int termination_type;
    int verbosity;
    lasvm_kcache_t *kcache;
    lasvm_t *sv;
    int nums0fsvs;
    float alpha_tol;

};

#endif // HAARFEATURE_H
