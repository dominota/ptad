#include "hogfeature.h"
#include <vector>
#include <cmath>
#include <ctime>
using namespace std;
using namespace cv;

/*******lasvm parameters*************/
#define LINEAR  0
#define POLY    1
#define RBF     2
#define SIGMOID 3

#define ONLINE 0
#define ONLINE_WITH_FINISHING 1

#define RANDOM 0
#define GRADIENT 1
#define MARGIN 2

#define ITERATIONS 0
#define SVS 1
#define TIME 2

#if USE_FLOAT
# define real_t float
#else
# define real_t double
#endif
const char *kernel_type_table[] = {"linear","polynomial","rbf","sigmoid"};

int kernel_type;              // LINEAR, POLY, RBF or SIGMOID kernels
double degree;// kernel params
double kgamma; // kernel params
double coef0;// kernel params
//LASVM parameters
std::vector <lasvm_sparsevector_t*> X_Samples; // feature vectors
std::vector <double> x_square;         // norms of input vectors, used for RBF

std::vector <lasvm_sparsevector_t*> Test_Samples; // feature vectors
std::vector <double> Test_square;         // norms of input vectors, used for RBF

std::vector <lasvm_sparsevector_t*> Xsv;// feature vectors for SVs
std::vector <double> xsv_square;
std::vector <double> alpha_sv;
std::vector <int> Y_Samples;
/*******lasvm parameters*************/
class stopwatch
{
public:
    stopwatch() : start(std::clock()){} //start counting time
    ~stopwatch();
    double get_time()
    {
        clock_t total = clock()-start;;
        return double(total)/CLOCKS_PER_SEC;
    };
private:
    std::clock_t start;
};
stopwatch::~stopwatch()
{
    clock_t total = clock()-start; //get elapsed time
    cout<<"Time(secs): "<<double(total)/CLOCKS_PER_SEC<<endl;
}


FHOGFeature::FHOGFeature()
{
    learningAlgorithm = BoostingMILClassifier;
    featureNum = 250;
    learnRate = 0.85; //0.85f;
    Miltrained = false;
    ratioTrained = false;
    SVMtrained = false;

    initHOGFeature();
}
void FHOGFeature::initHOGFeature(int learningMethod)
{
    learningAlgorithm = learningMethod;
    featureNum = 250;
    learnRate = 0.85;//0.85f;
    Miltrained = false;


    param.solver_type = L2R_LR;//L2R_L2LOSS_SVC; //L2R_LR
    param.C = 1;
    param.eps = INF; // see setting below
    param.p = 0.1;
    param.nr_weight = 0;
    param.weight_label = NULL;
    param.weight = NULL;
    param.init_sol = NULL;

    if(param.eps == INF)
    {
        switch(param.solver_type)
        {
        case L2R_LR:
        case L2R_L2LOSS_SVC:
            param.eps = 0.01;
            break;
        case L2R_L2LOSS_SVR:
            param.eps = 0.001;
            break;
        case L2R_L2LOSS_SVC_DUAL:
        case L2R_L1LOSS_SVC_DUAL:
        case MCSVM_CS:
        case L2R_LR_DUAL:
            param.eps = 0.1;
            break;
        case L1R_L2LOSS_SVC:
        case L1R_LR:
            param.eps = 0.01;
            break;
        case L2R_L1LOSS_SVR_DUAL:
        case L2R_L2LOSS_SVR_DUAL:
            param.eps = 0.1;
            break;
        }
    }

    SVMtrained = false;

    if(learningAlgorithm == RatioClassifier)
    {
        muPositive = vector<float>(featureNum, 0.0f);
        muNegative = vector<float>(featureNum, 0.0f);
        sigmaPositive = vector<float>(featureNum, 1.0f);
        sigmaNegative = vector<float>(featureNum, 1.0f);
    }


    /*** LaSVM ***/
    history_ = 0;
    /* Hyperparameters */
    kernel_type=RBF;              // LINEAR, POLY, RBF or SIGMOID kernels
    degree=3;
    kgamma=-1;
    coef0=0;// kernel params
    use_b0=1;                     // use threshold via constraint \sum a_i y_i =0
    selection_type=RANDOM;        // RANDOM, GRADIENT or MARGIN selection strategies
    optimizer=ONLINE_WITH_FINISHING; // strategy of optimization
    C=1;                       // C, penalty on errors
    C_neg=1;                   // C-Weighting for negative examples
    C_pos=1;                   // C-Weighting for positive examples
    epochs=1;                     // epochs of online learning
    candidates=50;				  // number of candidates for "active" selection process
    deltamax=1000;			  // tolerance for performing reprocess step, 1000=1 reprocess only
    incrLearningMode_ = 2;
    cache_size = 2000;
    termination_type=0;
    epsgr=1e-3;
    verbosity = 1;
    nums0fsvs = 0;
    // parameters 怎么设置
    select_size.push_back(100000000);
    alpha_tol = 0;
    accumulateSamples_ = 0;
}

// Compute the features of samples
void FHOGFeature::getFeatureHog(const std::vector<cv::Mat> & image,  std::vector<cv::Mat>& Features){
      if(image.size() == 0)
      {
        Features.clear();
        return;
      }
      Features.resize(image.size());
      for(size_t i = 0 ; i < image.size() ; i++)
           fhog_.getFeature(image[i],Features[i]);
}
void FHOGFeature::getFeatureHog(const cv::Mat & image,  cv::Mat& Features){
      fhog_.getFeature(image ,Features);
}


void  FHOGFeature::ratioClassifier(cv::Mat & positiveStates,std::vector<float> & prob)
{
    float pPos, pNeg,sumPositive,sumNegative;
    prob.resize(positiveStates.cols);
    for(int  j = 0 ; j < positiveStates.cols ; j++)
    {

        pPos = pNeg = sumNegative = sumPositive = 0;
        cv::Mat features = positiveStates.col(j);
        for (int i=0; i<featureNum; i++)
        {
            float tmp = features.at<float>(i,0);
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
void  FHOGFeature::ratioClassifier(std::vector<cv::Mat> & positiveStates,std::vector<float> & prob) // row feature
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

void  FHOGFeature::ratioclassifierUpdate(cv::Mat & positiveStates, cv::Mat & negativeStates)
{
    Scalar posmuTemp,possigmaTemp,negmuTemp,negsigmaTemp;
    if(!Miltrained)
    {
        Miltrained = true;
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
void   FHOGFeature::ratioclassifierUpdate(std::vector<cv::Mat> & positiveStates, std::vector<cv::Mat> & negativeStates)
{
    int posCounter = positiveStates.size();
    int negCounter = negativeStates.size();
    cv::Mat  positive,negative;
    positive.create( posCounter, featureNum, CV_32F );
    negative.create( negCounter, featureNum, CV_32F );
    int pc = 0;
    int nc = 0;
    for(int i = 0 ; i < posCounter ; i++)
    {
        cv::Mat & feat = positiveStates[i];
        for ( int j = 0; j < feat.rows; j++ )
          positive.at<float>( pc, j ) = feat.at<float>( j, 0 );
         pc++;
    }
    for(int i = 0 ; i < negCounter ; i++)
    {
        cv::Mat & feat = negativeStates[i];
        for ( int j = 0; j < feat.rows; j++ )
          negative.at<float>( nc, j ) = feat.at<float>( j, 0 );
         nc++;
    }
    ratioclassifierUpdate(positive,negative);
}

void  FHOGFeature::MilClassifierUpdate(cv::Mat & positiveStates, cv::Mat & negativeStates)
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

void  FHOGFeature::MilClassifierUpdate(std::vector<cv::Mat> & positiveStates, std::vector<cv::Mat> & negativeStates)
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


void  FHOGFeature::Milclassify(cv::Mat & States, std::vector<float> & prob)
{
    prob.clear();
    prob = MilBoost.classify(States);
}
void  FHOGFeature::Milclassify(std::vector<cv::Mat> & States, std::vector<float> & prob)
{
    if(prob.size()!=States.size())
        prob = std::vector<float>(States.size(),0);
    MilBoost.classify(States,prob);
}



void FHOGFeature::prepareSVMdata(std::vector<cv::Mat>& posfeatures, std::vector<cv::Mat>& negfeatures)
{
    int posCounts  = posfeatures.size() ;
    int negCounts  = negfeatures.size();
    int featurecols  = posfeatures[0].cols;
    prob_.l = (posCounts + negCounts) /2 * 2;
    prob_.y = Malloc(double,prob_.l);
    prob_.x = Malloc(struct feature_node *,prob_.l);
    x_space = Malloc(struct feature_node, prob_.l *(featurecols +1) +prob_.l);
    prob_.bias= -1;
    int j = 0;

    for(int i = 0 ; i < prob_.l /2; i++)
    {
        float * p;
        int  currentFg = -1;
        if( i < posCounts)
        {
            p = (float*)posfeatures[i].data;
            currentFg = 1;
        }
        else
            p = (float*)negfeatures[i -posCounts].data;

        prob_.y[i *2] = currentFg;

        prob_.x[i *2] = &x_space[j];
        for(int k = 0 ; k < featurecols ; k++)
        {
            if(p[k]!=0)
            {
                x_space[j].index = k + 1;
                x_space[j].value  = p[k];
                ++j;
            }
        }
        if(prob_.bias >= 0)
            x_space[j++].value = prob_.bias;
        x_space[j++].index = -1;


        float * p2;
        currentFg = -1;
        int mapPosition = (int)(i +  prob_.l / 2);
        if( mapPosition < posCounts)
        {
            p2 = (float*)posfeatures[mapPosition].data;
            currentFg = 1;
        }
        else
            p2 = (float*)negfeatures[mapPosition -posCounts].data;

        prob_.y[i*2 +1] = currentFg;
        prob_.x[i*2 +1] = &x_space[j];
        for(int k = 0 ; k < featurecols ; k++)
        {
            if(p2[k]!=0)
            {
                x_space[j].index = k + 1;
                x_space[j].value  = p2[k];
                ++j;
            }
        }
        if(prob_.bias >= 0)
            x_space[j++].value = prob_.bias;
        x_space[j++].index = -1;
    }
    if(prob_.bias >= 0)
    {
        prob_.n=featurecols+1;
        for(int i=1;i<prob_.l;i++)
            (prob_.x[i]-2)->index = prob_.n;
        x_space[j-2].index = prob_.n;
    }
    else
        prob_.n=featurecols;
}

void  FHOGFeature::SVMClassifyUpdate(std::vector<cv::Mat> & positiveStates, std::vector<cv::Mat> & negativeStates)
{
    prepareSVMdata(positiveStates,negativeStates);
    if(!SVMtrained)
    {
         initial_model = train(&prob_, &param);
         int nr_feature = get_nr_feature(initial_model);
         predict_x_space = Malloc(struct feature_node,  nr_feature+2);
         model_ =  initial_model;
         SVMtrained = true;
    }
    else
    {
        model_ = warm_start_train(&prob_, &param, initial_model);
        free_model_content(initial_model);
        initial_model = model_;
    }
    free(prob_.y);
    free(prob_.x);
    free(x_space);
}

void  FHOGFeature::SVMclassify(std::vector<cv::Mat> & States,std::vector<float> & prob)
{
    double *prob_estimates=NULL;
    int nr_class=get_nr_class(model_);
    CV_Assert(check_probability_model(model_));
    prob_estimates = (double *) malloc(nr_class*sizeof(double));

    int posCounts  = States.size();
    prob.resize(posCounts);
    int nr_feature= get_nr_feature(model_);
    int n = nr_feature;
    if(model_->bias>=0)
        n = nr_feature + 1;
    for(int i = 0 ; i < posCounts ; i++)
    {
        if(prob[i]!=-FLT_MAX)
        {
            int j = 0 ;
            float * p = (float*)States[i].data;
            for(int k = 0 ; k < nr_feature ; k++)
            {
                if(p[k]!=0)
                {
                    predict_x_space[j].index = k + 1 ;
                    predict_x_space[j].value  = p[k];
                    ++j;
                }
            }
            if(model_->bias>=0)
            {
                predict_x_space[j].index = n;
                predict_x_space[j].value = model_->bias;
                j++;
            }
            predict_x_space[j].index = -1;
            predict_probability(model_,predict_x_space,prob_estimates);
            prob[i] = prob_estimates[0];
        }
    }
    free(prob_estimates);
}
void  FHOGFeature::classifier(std::vector<cv::Mat> & States,std::vector<float> & prob)
{
    switch (learningAlgorithm)
    {
    case RatioClassifier: ratioClassifier(States,prob); break;
    case BoostingMILClassifier: Milclassify(States,prob); break;
    case SVMClassifier: SVMclassify(States,prob);break;
    default: break;
    }
}

void  FHOGFeature::classifierUpdate(std::vector<cv::Mat> & positiveStates, std::vector<cv::Mat> & negativeStates)
{
    switch (learningAlgorithm)
    {
    case RatioClassifier: ratioclassifierUpdate(positiveStates,negativeStates); break;
    case BoostingMILClassifier: MilClassifierUpdate(positiveStates,negativeStates); break;
    case SVMClassifier: SVMClassifyUpdate(positiveStates,negativeStates);break;
    default: break;
    }
}

double kernel(int i, int j, void *kparam)
{
    double dot;
    //static intkcalcs = 0 ;
  //  kcalcs++;

    dot=lasvm_sparsevector_dot_product(X_Samples[i],X_Samples[j]);

    // sparse, linear kernel
    switch(kernel_type)
    {
    case LINEAR:
        return dot;
    case POLY:
        return pow(kgamma*dot+coef0,degree);
    case RBF:
        return exp(-kgamma*(x_square[i]+x_square[j]-2*dot));
    case SIGMOID:
        return tanh(kgamma*dot+coef0);
    }
    return 0;
}

double testkernel(int i, int j, void *kparam)
{
    double dot;
    dot=lasvm_sparsevector_dot_product(Test_Samples[i],Xsv[j]);

    // sparse, linear kernel
    switch(kernel_type)
    {
    case LINEAR:
        return dot;
    case POLY:
        return pow(kgamma*dot+coef0,degree);
    case RBF:
        return exp(-kgamma*(Test_square[i]+xsv_square[j]-2*dot));
    case SIGMOID:
        return tanh(kgamma*dot+coef0);
    }
    return 0;
}

void FHOGFeature::prepareLaSVMdata(std::vector<cv::Mat>& posfeatures, std::vector<cv::Mat>& negfeatures)
{
    lasvm_sparsevector_t* v;
    int posCounts  = posfeatures.size();
    int negCounts  = negfeatures.size();
    int numsfoSamples = (posCounts + negCounts) /2 * 2;
    int max_index = 0;
    for(int i = 0; i < numsfoSamples; i++ )
    {
          v=lasvm_sparsevector_create();
          X_Samples.push_back(v);
          Y_Samples.push_back(1);
    }
    int featurecols  = posfeatures[0].cols;
    for(int i = 0; i < numsfoSamples/2; i++ )
    {
        float * p;
        int  currentFg = -1;
        if( i < posCounts)
        {
            p = (float*)posfeatures[i].data;
            currentFg = 1;
        }
        else
            p = (float*)negfeatures[i -posCounts].data;
        Y_Samples[history_+2*i] = (currentFg);
        for(int k = 0 ; k < featurecols ; k++)
        {
            if(p[k]!=0)
            {
                lasvm_sparsevector_set(X_Samples[history_+2*i],k + 1,p[k]);
                if (k+1>max_index) max_index=k+1;
            }
        }
        float * p2;
        currentFg = -1;
        int mapPosition = (int)(i +  numsfoSamples / 2);
        if( mapPosition < posCounts)
        {
            p2 = (float*)posfeatures[mapPosition].data;
            currentFg = 1;
        }
        else
            p2 = (float*)negfeatures[mapPosition -posCounts].data;
        Y_Samples[history_+2*i+1] = (currentFg);
        for(int k = 0 ; k < featurecols ; k++)
        {
            if(p2[k]!=0)
            {
                lasvm_sparsevector_set(X_Samples[history_+2*i+1],k + 1,p2[k]);
                if (k+1>max_index) max_index=k+1;
            }
        }
    }
    //std::cout<<negfeatures[0]<<std::endl;
    //std::cout<<posfeatures[0]<<std::endl;
    current_samples_ = numsfoSamples;
    learningSamples_ = history_+current_samples_;

    /***在原来的文件中是先保存再统一录入学习，我们这儿不需要再次保存  */
    if(kernel_type==RBF)
    {
        x_square.clear();
        x_square.resize(learningSamples_);
        for(int i=0; i<learningSamples_; i++)
            x_square[i]=lasvm_sparsevector_dot_product(X_Samples[i],X_Samples[i]);
    }

    if(kgamma==-1)
        kgamma=1.0/ ((double) max_index); // same default as LIBSVM

    accumulateSamples_ += numsfoSamples; //一直以来的所有样本

}
void FHOGFeature::prepareLaSVMdata(std::vector<cv::Mat>& features)
{
    lasvm_sparsevector_t* v;
    int numsfoSamples = features.size();
    int max_index = 0;
    if(numsfoSamples==0) return;
    Test_Samples.clear();
    for(int i = 0; i < numsfoSamples; i++ )
    {
          v=lasvm_sparsevector_create();
          Test_Samples.push_back(v);
    }
    int featurecols  = features[0].cols;
    for(int i = 0; i < numsfoSamples; i++ )
    {
        float * p = (float*)features[i].data;
        for(int k = 0 ; k < featurecols ; k++)
        {
            if(p[k]!=0)
            {
                lasvm_sparsevector_set(Test_Samples[i],k + 1,p[k]);
                if (k+1>max_index) max_index=k+1;
            }
        }
    }
    /***在原来的文件中是先保存再统一录入学习，我们这儿不需要再次保存  */
    if(kernel_type==RBF)
    {
        Test_square.resize(numsfoSamples);
        for(int i=0; i<numsfoSamples; i++)
            Test_square[i]=lasvm_sparsevector_dot_product(Test_Samples[i],Test_Samples[i]);
    }
    if(kgamma==-1)
        kgamma=1.0/ ((double) max_index); // same default as LIBSVM

}

void  FHOGFeature::LaSVMClassifyUpdate(std::vector<cv::Mat> & positiveStates, std::vector<cv::Mat> & negativeStates)
{
    prepareLaSVMdata(positiveStates,negativeStates);
    stopwatch *sw; // start measuring time after loading is finished
    sw=new stopwatch;    // save timing information
    if(incrLearningMode_==IncrementWithoutPersistence && SVMtrained)//without persistence
    {
        // everything is new when we start
        for(int i=0; i<learningSamples_; i++) inew.push_back(i);
        make_old(0); //dummy , else iold remains uninitialized later and therefore gives error

        kcache=lasvm_kcache_create(kernel, NULL);
        lasvm_kcache_set_maximum_size(kcache, cache_size*1024*1024);
        sv=lasvm_create(kcache,use_b0,C*C_pos,C*C_neg);
        printf("set cache size %lld\n",cache_size);
        // first add 5 examples of each class, just to balance the initial set
        int c1=0,c2=0;
        for(int i=1; i<learningSamples_; i++)
        {
            if(Y_Samples[i]==1 && c1<5) {lasvm_process(sv,i,(double) Y_Samples[i]); c1++; make_old(i);}
            if(Y_Samples[i]==-1 && c2<5){lasvm_process(sv,i,(double) Y_Samples[i]); c2++; make_old(i);}
            if(c1==5 && c2==5) break;
        }

        int index2;
        //process all the old SVs history_
        for(int i=1; i< history_; i++) // Old Svs + new samples
        {
            if(inew.size()==0) break;

            if(i<=inew.size())
            {
                lasvm_process(sv,i,(double) Y_Samples[i]); c1++; make_old(i);
            }
            else
            {
                index2=inew.size()-1;
                lasvm_process(sv,index2,(double) Y_Samples[index2]);  make_old(index2);
            }
            //	lasvm_reprocess(sv,epsgr); //TODO: decide whether to reprocess or not
        }
    }
    else if(incrLearningMode_==IncrementWithPersistence && SVMtrained)//with persistence old Samples + new Samples
    {
        // everything is new when we start
        for(int i=0; i<learningSamples_; i++) inew.push_back(i);
        make_old(0); //dummy , else iold remains uninitialized later and therefore gives error
        lasvm_kcache_set_maximum_size(kcache, cache_size*1024*1024);
        printf("set cache size %lld\n",cache_size);
        int index2;
        for(int i = 1; i < history_; i++)
        {
            if(inew.size()==0) break;
            if(i<=inew.size())
                make_old(i);
            else
            {
                index2=inew.size()-1;
                make_old(inew[index2]);
            }
            //	lasvm_reprocess(sv,epsgr); //TODO: decide whether to reprocess or not
        }
    }
    else // New Samples
    {
        SVMtrained  = true;
        kcache=lasvm_kcache_create(kernel, NULL);
        lasvm_kcache_set_maximum_size(kcache, cache_size*1024*1024);
        sv=lasvm_create(kcache,use_b0,C*C_pos,C*C_neg);
        // everything is new when we start
        for(int i=0;i< learningSamples_;i++) inew.push_back(i);
        // first add 5 examples of each class, just to balance the initial set
        int c1=0,c2=0;
        for(int i=0;i<learningSamples_;i++)
        {
            if(Y_Samples[i]==1 && c1<5) {lasvm_process(sv,i,(double) Y_Samples[i]); c1++; make_old(i);}
            if(Y_Samples[i]==-1 && c2<5){lasvm_process(sv,i,(double) Y_Samples[i]); c2++; make_old(i);}
            if(c1==5 && c2==5) break;
        }
    }

    for(int j=0;j<epochs;j++)
    {
        for(int i=history_; i < learningSamples_; i++)
        {
            if(inew.size()==0) break; // nothing more to select
            int s=select(sv);            // selection strategy, select new point
            int t1=lasvm_process(sv,s,(double) Y_Samples[s]);
            int t2 = 0;
            if (deltamax<=1000) // potentially multiple calls to reprocess..
            {
                t2=lasvm_reprocess(sv,epsgr);// at least one call to reprocess
                while (lasvm_get_delta(sv)>deltamax && deltamax<1000)
                {
                    t2=lasvm_reprocess(sv,epsgr);
                }
            }

            if (verbosity==2)
            {
                int l=(int) lasvm_get_l(sv);
                printf("l=%d process=%d reprocess=%d\n",l,t1,t2);
            }
            else
                if(verbosity==1)
                    if( (i%1000)==0){ fprintf(stdout, "..%d",i); fflush(stdout); }

            int l=(int) lasvm_get_l(sv);

            for(int k=0; k<(int)select_size.size(); k++)
            {
                if( (termination_type==ITERATIONS && i==select_size[k])|| (termination_type==SVS && l>=select_size[k])
                        || (termination_type==TIME && sw->get_time()>=select_size[k])  )
                {
                    select_size[k]=select_size[select_size.size()-1];
                    select_size.pop_back();
                }
            }
            if(select_size.size()==0) break; // early stopping, all intermediate models saved
        }

        inew.resize(0);iold.resize(0); // start again for next epoch..
        //TODO: Do we reprocess entire data or only new data? ？？？？
        for(int i=0;i<learningSamples_;i++) inew.push_back(i);
    }

    //i//f(saves<2) //提取支持向两的alpha
   // {
        // finish 提取处理alpha
     finish(sv); // if haven't done any intermediate saves, do final save
        //timer+=sw->get_time();
        //f << m << " " << count_svs() << " " << kcalcs << " " << timer << endl;
   // }

    if(verbosity>0) printf("\n");
    nums0fsvs = count_svs();
    printf("nSVs=%d\n",nums0fsvs);
    printf("||w||^2=%g\n",lasvm_get_w2(sv)); //分界面

    if(incrLearningMode_!=IncrementWithPersistence)// 必须保存
    {
        lasvm_destroy(sv);
        lasvm_kcache_destroy(kcache);
    }
    saveModel();
    /*std::vector<float> prob;
    prob.resize(negativeStates.size());
    LaSVMclassify(negativeStates,prob);*/
}

void FHOGFeature::saveModel()
{
    //** 什么都不许要操作，仅仅需要拷贝支持向两
    for(int i = 0; i < Xsv.size(); i++ )
        lasvm_sparsevector_destroy(Xsv[i]);
    xsv_square.resize(nums0fsvs);
    Xsv.clear();
    Xsv.reserve(nums0fsvs);
    lasvm_sparsevector_t* v;
    for(int i = 0; i < nums0fsvs; i++ )
    {
        v=lasvm_sparsevector_create();
        Xsv.push_back(v);
    }
    alpha_sv.resize(nums0fsvs);

    //训练的所有样本
    int index = 0;
    for(int j=0;j<2;j++)
        for(int i=0;i<learningSamples_;i++)
        {
            if (j==0 && Y_Samples[i]==-1) continue;
            if (j==1 && Y_Samples[i]==1) continue;
            if (alpha[i]*Y_Samples[i]< alpha_tol) continue; // not an SV
            lasvm_sparsevector_pair_t *p1 = X_Samples[i]->pairs;
            alpha_sv[index] = alpha[i];
            xsv_square[index] = x_square[i];
            while (p1)
            {
                lasvm_sparsevector_set(Xsv[index],p1->index,p1->data);
                p1 = p1->next;
            }
            index++;
        }

    if(incrLearningMode_ != IncrementWithPersistence)  // all svs samples are recorded;
    {
        Y_Samples.clear();
        for(int i = 0; i < X_Samples.size(); i++ )
            lasvm_sparsevector_destroy(X_Samples[i]);
        X_Samples.clear();
        for(int i = 0; i < nums0fsvs; i++ )
        {
            v=lasvm_sparsevector_create();
            X_Samples.push_back(v);
        }
        for(int i = 0; i < nums0fsvs; i++ )
        {
            lasvm_sparsevector_pair_t *p1 = Xsv[i]->pairs;
            while (p1)
            {
                lasvm_sparsevector_set(X_Samples[i],p1->index,p1->data);
                p1 = p1->next;
            }
        }
        history_ = nums0fsvs;
    }
    else  // all previous samples are recorded;
    {
        history_ = learningSamples_;
    }

}
int FHOGFeature::count_svs()
{
    int sv1,sv2;
    double max_alpha;
    max_alpha=0;
    sv1=0;sv2=0;

    for(int i=0; i<learningSamples_; i++) 	// Count svs..
    {
        if(alpha[i]>max_alpha)  max_alpha=alpha[i];
        if(-alpha[i]>max_alpha) max_alpha=-alpha[i];
    }

    alpha_tol=max_alpha/1000.0;

    for(int i=0;i<learningSamples_;i++)
    {
        if(Y_Samples[i]>0)  //正的支持向两数量
        {
            if(alpha[i] >= alpha_tol) sv1++;
        }
        else //负的支持向两数量
        {
            if(-alpha[i] >= alpha_tol) sv2++;
        }
    }
    return sv1+sv2;
}

void FHOGFeature::finish(lasvm_t *_sv)
{
    int i,l;
    if (optimizer==ONLINE_WITH_FINISHING)
    {
        fprintf(stdout,"..[finishing]");

        int iter=0;

        do {
            iter += lasvm_finish(_sv, epsgr);
        } while (lasvm_get_delta(_sv)>epsgr);

    }

    l=(int) lasvm_get_l(_sv);
    int *svind;
    svind= new int[l];

    int svs=lasvm_get_sv(_sv,svind);

    alpha.resize(learningSamples_);
    for(int i=0; i<learningSamples_; i++) alpha[i]=0;
    double *svalpha;
    svalpha=new double[l];
    lasvm_get_alpha(_sv,svalpha);
    for(i=0;i<svs;i++) alpha[svind[i]]=svalpha[i];
    b0=lasvm_get_b(_sv);
}

void  FHOGFeature::LaSVMclassify(std::vector<cv::Mat> & States,std::vector<float> & prob)
{
    prepareLaSVMdata(States);
    int nums  = Test_square.size();
    double y = 0;
    for(int i=0; i< nums; i++)
    {
        y=-b0;
        for(int j=0;j<nums0fsvs;j++)
        {
            y+=alpha_sv[j]*testkernel(i,j,NULL);
        }
        if(y>=0)
            prob[i]=1;
        else
            prob[i]=-1;
    }

    for(int i = 0 ;i < nums;i++)
        lasvm_sparsevector_destroy(Test_Samples[i]);
    Test_square.clear();
    Test_square.clear();
}
void FHOGFeature::make_old(int val)// move index <val> from new set into old set
{
    int i,ind=-1;
    for(i=0;i<(int)inew.size();i++)
    {
        if(inew[i]==val) {ind=i; break;}
    }

    if (ind>=0)
    {
        inew[ind]=inew[inew.size()-1];
        inew.pop_back();
        iold.push_back(val);
    }
}

int FHOGFeature::select(lasvm_t *sv) // selection strategy
{
    int s=-1;
    int t,i,r,j;
    double tmp,best; int ind=-1;

    switch(selection_type)
    {
    case RANDOM:   // pick a random candidate
        s=rand() % inew.size();
        break;

    case GRADIENT: // pick best gradient from 50 candidates
        j=candidates; if((int)inew.size()<j) j=inew.size();
        r=rand() % inew.size();
        s=r;
        best=1e20;
        for(i=0;i<j;i++)
        {
            r=inew[s];
            tmp=lasvm_predict(sv, r);
            tmp*=Y_Samples[r];
            //printf("%d: example %d   grad=%g\n",i,r,tmp);
            if(tmp<best) {best=tmp;ind=s;}
            s=rand() % inew.size();
        }
        s=ind;
        break;

    case MARGIN:  // pick closest to margin from 50 candidates
        j=candidates; if((int)inew.size()<j) j=inew.size();
        r=rand() % inew.size();
        s=r;
        best=1e20;
        for(i=0;i<j;i++)
        {
            r=inew[s];
            tmp=lasvm_predict(sv, r);
            if (tmp<0) tmp=-tmp;
            //printf("%d: example %d   grad=%g\n",i,r,tmp);
            if(tmp<best) {best=tmp;ind=s;}
            s=rand() % inew.size();
        }
        s=ind;
        break;
    }

    t=inew[s];
    inew[s]=inew[inew.size()-1];
    inew.pop_back();
    iold.push_back(t);
    //printf("(%d %d)\n",iold.size(),inew.size());
    return t;
}
