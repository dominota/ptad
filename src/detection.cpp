/*
 * Detection.cpp
 *
 *  Created on: Jun 9, 2011
 *      Author: alantrrs
 */

#include <detection.h>
#include <stdio.h>
#include <omp.h>
using namespace cv;
using namespace std;


Detection::Detection(){
}
Detection::~Detection(){
    clear();
}

Detection::Detection(const FileNode& file){
    read(file);
}

void Detection::read(const FileNode& file){
    ///Bounding Box Parameters
    min_win = (int)file["min_win"];
    ///Genarator Parameters
    //initial parameters for positive examples
    patch_size = (int)file["patch_size"];
    num_closest_init = (int)file["num_closest_init"];
    num_warps_init = (int)file["num_warps_init"];
    noise_init = (int)file["noise_init"];
    angle_init = (float)file["angle_init"];
    shift_init = (float)file["shift_init"];
    scale_init = (float)file["scale_init"];
    //update parameters for positive examples
    num_closest_update = (int)file["num_closest_update"];
    num_warps_update = (int)file["num_warps_update"];
    noise_update = (int)file["noise_update"];
    angle_update = (float)file["angle_update"];
    shift_update = (float)file["shift_update"];
    scale_update = (float)file["scale_update"];
    //parameters for negative examples
    bad_overlap = (float)file["overlap"];
    bad_patches = (int)file["num_patches"];
    directionalFilter = (int)file["dvarfilter"];
    increase_ncc_samples = (int)file["increase_ncc_samples"];
    pyramidlevels = (int)file["pyramidlevels"];
    is_debug = (int)file["show_debug_info"];
    classifier.read(file);
    if(pyramidlevels<=0) pyramidlevels = 5;
    scales.resize(2*pyramidlevels+1);
    scales[pyramidlevels] = 1;
    for(int i = 1 ; i <= pyramidlevels ; i++)
    {
        float pyramid_layer   = std::pow(1.2,i);
        scales[pyramidlevels-i] = float(1.0/pyramid_layer);
        scales[pyramidlevels+i] = float(pyramid_layer);
    }
}
void Detection::clear()
{
#if defined(FernFeature)  //select one features
    nXT.clear();
#elif defined(CompressiveFeature)
    fX.clear();
#elif defined(HaarLikeFeature)
    fX.clear();
#endif
#ifdef using_pyramid   //using image pyramids
    timages.clear();
    filterImages.clear();
    iisum.clear();
    iisqsum.clear();
    gridpyramids.clear();
    scalespyramids.clear();
    scalesgridnums.clear();
#else
    timages.release();
    filterImages.release();
    iisum.release();
    iisqsum.release();
#endif
    pX.clear();
    nX.clear();
    tmp.patt.clear();
    tmp.patt1.clear();
    tmp.candidateIndex.clear();
    tmp.conf.clear();
    frameImage.release();
    dbb.clear();
    dconf.clear();
    grid.clear();
    BoundingBoxScales.clear();
    //scales.clear();
    scalesizes.clear();
    good_boxes.clear();
    bad_boxes.clear();
    classifier.clear();
}

void Detection::init(const Mat& frame, const Rect& box){  

    buildGrid(frame,box);
    printf("Created %d bounding boxes\n",(int)grid.size());

    dconf.reserve(100);
    dbb.reserve(100);
    bbox_step =7;

    // initialization for different features
#if defined(FernFeature)
    tmp.conf = vector<float>(grid.size());
    tmp.patt = vector<vector<int> >(grid.size(),vector<int>(classifier.getNumStructs(),0));
    dt.bb.reserve(grid.size());
#elif defined(CompressiveFeature)
    tmp.conf = vector<float>(grid.size());
    tmp.patt = vector<vector<float> >(grid.size(),vector<float>(classifier.compDetector.featureNum,0));
    dt.bb.reserve(grid.size());
    fX.reserve(grid.size());
#elif defined(HaarLikeFeature)
    classifier.haarfeature.initHaarFeature(HaarFeature::AdaBoostinglassifierAlgorithm);
    tmp.conf =  vector<float>(grid.size());
    // Error!!!  All elements point to the same storage location.
    //tmp.patt1 =  vector<cv::Mat >(grid.size(),Mat_<float>(Size(1, classifier.haarfeature.featureNum))); //row features;
    tmp.patt1.clear();
    tmp.patt1.reserve(grid.size());
    for(int i = 0; i < grid.size() ; i++)
        tmp.patt1.push_back(cv::Mat_<float>(1,classifier.haarfeature.featureNum));
    dt.bb.reserve(grid.size());
    tmp.candidateIndex.reserve(grid.size());
#elif defined(HOGFeature)
    cv::Size initia_patch(box.width,box.height);
    classifier.fhogfeature.fhog_.init(4,initia_patch);// fhog feature set
    classifier.fhogfeature.initHOGFeature(FHOGFeature::BoostingMILClassifier);//BoostingMILClassifier
    classifier.fhogfeature.featureNum =  classifier.fhogfeature.fhog_.featureNums;
    tmp.conf = vector<float>(grid.size());
    tmp.patt1.clear();
    tmp.patt1.reserve(grid.size());
    for(int i = 0; i < grid.size() ; i++)
        tmp.patt1.push_back(cv::Mat_<float>(1,classifier.fhogfeature.featureNum));
    dt.bb.reserve(grid.size());
#endif

    good_boxes.reserve(grid.size());
    bad_boxes.reserve(grid.size());
    pEx.create(patch_size,patch_size,CV_64F);

    getOverlappingBoxes(box,num_closest_init);
    printf("Found %d good boxes, %d bad boxes\n",(int)good_boxes.size(),(int)bad_boxes.size());
    printf("Best Box: %f %f %f %f\n",best_box.x,best_box.y,best_box.width,best_box.height);
    printf("Bounding box hull: %f %f %f %f\n",bbhull.x,bbhull.y,bbhull.width,bbhull.height);

#ifdef using_pyramid
    iisum.resize(scalespyramids.size());
    iisqsum.resize(scalespyramids.size());
    for(int i = 0; i < scalespyramids.size(); i++)
    {
        iisum[i].create(scalespyramids[i]+cv::Size(1,1),CV_32F);
        iisqsum[i].create(scalespyramids[i]+cv::Size(1,1),CV_64F);
    }
    constructImagePyramids(frame); //构造高斯金字塔
#else
    timages = frame; // timages orignal image
    GaussianBlur(frame,filterImages,Size(9,9),1.5);
//#if defined(FernFeature)  //only process orignal image
//    GaussianBlur(frame,filterImages,Size(9,9),1.5);
//#elif defined(CompressiveFeature)|| defined(HaarLikeFeature)
    //cv::equalizeHist(frame, filterImages); //直方图均衡化
//#endif
    iisum.create(frame.rows+1,frame.cols+1,CV_32F);
    iisqsum.create(frame.rows+1,frame.cols+1,CV_64F);
    integral(frame,iisum,iisqsum);
#endif


    //fprintf(bb_file,"%d,%d,%d,%d,%f\n",lastbox.x,lastbox.y,lastbox.br().x,lastbox.br().y,lastconf);
    //Prepare Classifier
#ifdef using_pyramid
    int index = scaleindex.at(1.0);
#if defined(FernFeature)
    classifier.prepare(scalesizes[index]);
#elif defined(CompressiveFeature)
    classifier.compDetector.prepareCompressive(scalesizes[index]);
#elif defined(HaarLikeFeature)
    classifier.haarfeature.prepareHaar(scalesizes[index]);
    classifier.haarSamples.reserve(grid.size()); //记录所有的数据
#endif

#else

#if defined(FernFeature)
    classifier.prepareScale(scalesizes); //fern features
#elif defined(CompressiveFeature)
    classifier.compDetector.prepareCompressiveScale(scalesizes);
#elif defined(HaarLikeFeature)
    classifier.haarfeature.prepareHaarScale(scalesizes);
#endif

#endif

    lastbox=best_box;
    lastconf=1;
    lastvalid=true;
    lastdetect = true;    //Print

    pX.reserve(300);
    nX.reserve(grid.size());

    generatePositiveData(num_warps_init);  // Generate positive data

    Scalar stdev, mean;     // Set variance threshold
    meanStdDev(frame(best_box),mean,stdev);
    var = pow(stdev.val[0],2)*0.5; //getVar(best_box,iisum,iisqsum);

    cout << "variance: " << var << endl;


#ifdef using_pyramid
    index = scaleindex.at(1.0);
    double vr =  getVar(best_box,iisum[index],iisqsum[index])*0.5;
    col_var = getColVar(best_box,iisum[index],iisqsum[index])*0.5;
    row_var = getRowVar(best_box,iisum[index],iisqsum[index])*0.5;
#else
    double vr =  getVar(best_box,iisum,iisqsum)*0.5;
    col_var = getColVar(best_box,iisum,iisqsum)*0.5;
    row_var = getRowVar(best_box,iisum,iisqsum)*0.5;
#endif
    cout << "check variance: " << vr << " col_variance: "<<col_var<<" row_variance: "<<row_var<<endl;

    generateNegativeData();     // Generate negative data
    ///Split Negative NN Examples into Training and Testing sets  for NCC classifiers
    int half = (int)nEx.size()*0.5f;
    nExT.assign(nEx.begin()+half,nEx.end());
    nEx.resize(half);
    vector<cv::Mat> nn_data(nEx.size()+nn_examples.size());
    nn_posnums = nn_examples.size();
    for(int i = 0 ;i < nn_examples.size() ;i++)
        nn_data[i] = nn_examples[i];
    for (int i=0;i<nEx.size();i++){
        nn_data[i+nn_examples.size()]= nEx[i];
    }

#if defined(FernFeature)  //Fern Features     //Merge Negative Data with Positive Data and shuffle it
    half = (int)nX.size()*0.5f;
    nXT.assign(nX.begin()+half,nX.end());
    nX.resize(half);
    vector<pair<vector<int>,int> > ferns_data(nX.size()+pX.size());
    vector<int> idx = index_shuffle(0,ferns_data.size());  //随机选择样本训练
    int a=0;
    for (int i=0;i<pX.size();i++){
        ferns_data[idx[a]] = pX[i];
        a++;
    }
    for (int i=0;i<nX.size();i++){
        ferns_data[idx[a]] = nX[i];
        a++;
    }
    classifier.trainF(ferns_data,2); //bootstrap = 2
#elif defined(CompressiveFeature)
    //classifier.compDetector.classifierUpdate(ct_px,ct_nx);
    std::vector<std::vector<float> > nxsamples = nX;
    if(nX.size() >400) // select 400 samples for training
        nxsamples.resize(400);
    classifier.compDetector.MilClassifierUpdate(pX,nxsamples);
    fX.reserve(grid.size());
#elif defined(HaarLikeFeature)
    std::vector<cv::Mat> nxsamples;
    if(nX.size() > 400)
        nxsamples.assign(nX.begin(),nX.begin()+400);
    else
        nxsamples = nX;
    classifier.haarfeature.classifierUpdate(pX,nxsamples);
#elif defined(HOGFeature)
    if(nX.size() > 200)
        nX.resize(200);
    classifier.fhogfeature.classifierUpdate(pX,nX);
    // classifier.fhogfeature.SVMClassifyUpdate(fhog_px_vec,fhog_nx_vec);
#endif
    classifier.trainNN(nn_data,nn_posnums);
    ///Threshold Evaluation on testing sets
#ifdef FernFeature
    classifier.evaluateFernTh(nXT);
#endif
    classifier.evaluateNCCTh(nExT);
    if(is_debug)
        classifier.show();
    detected=false;
    adjustingTypes = NotChangingTrackingSate;
    nn_examples.clear();
}
bool Detection::filterVar(const BoundingBox& box,const Mat& sum,const Mat& sqsum, double factor)
{
    bool passFilter = true;
    double Avar = getVar(box,sum,sqsum) ;
    double ratio = 1;
    if(Avar<var *factor*ratio /*|| Avar > var *factor*/)
        passFilter = false;
    else if(directionalFilter)
    {
        double Cvar = getColVar(box,sum,sqsum) ;
        if(Cvar<col_var *factor*ratio /*|| Cvar >col_var *factor*/) // for not rotating object
            passFilter = false;
        else
        {
            double Rvar = getRowVar(box,sum,sqsum);
            if(Rvar<row_var *factor*ratio /*|| Rvar >row_var*/) // for not rotating object
                passFilter = false;
        }
        if(!passFilter) directionFilterValid++;
    }
    return passFilter;
}

double Detection::getVar(const BoundingBox& box,const Mat& sum,const Mat& sqsum){
    double brs = sum.at<int>(box.y+box.height,box.x+box.width);
    double bls = sum.at<int>(box.y+box.height,box.x);
    double trs = sum.at<int>(box.y,box.x+box.width);
    double tls = sum.at<int>(box.y,box.x);
    double brsq = sqsum.at<double>(box.y+box.height,box.x+box.width);
    double blsq = sqsum.at<double>(box.y+box.height,box.x);
    double trsq = sqsum.at<double>(box.y,box.x+box.width);
    double tlsq = sqsum.at<double>(box.y,box.x);
    double mean = (brs+tls-trs-bls)/((double)box.area());
    double sqmean = (brsq+tlsq-trsq-blsq)/((double)box.area());
    // var = mean(I"(B)) - mean(I'(B))^2
    // mean(I"(B)) = (brsq+tlsq-trsq-blsq)/((double)box.area())
    // mean(I'(B)) =  (brs+tls-trs-bls)/((double)box.area())
    return (sqmean-mean*mean);
}
double Detection::getColVar(const BoundingBox& box,const Mat& sum,const Mat& sqsum){
    double colVar = 0;
    double mean = 0;
    for(int i = 0 ; i < box.width; i++)//column variance  row
    {
        double brs = sum.at<int>(box.y+box.height,box.x+i+1);
        double bls = sum.at<int>(box.y+box.height,box.x+i);
        double trs = sum.at<int>(box.y,box.x+i+1);
        double tls = sum.at<int>(box.y,box.x+i);
        double sumI = (brs+tls-trs-bls);
        mean +=sumI;
        colVar += sumI *sumI;
    }
    mean = mean / box.area();
    colVar = colVar /( box.height * box.area()) - mean *mean;
    return colVar;
}
double Detection::getRowVar(const BoundingBox& box,const cv::Mat& sum,const cv::Mat& sqsum)
{
    double rowVar = 0;
    double mean = 0;
    for(int i = 0 ; i < box.height; i++)//column variance  row
    {
        double brs = sum.at<int>(box.y+i+1,box.x+box.width);
        double bls = sum.at<int>(box.y+i+1,box.x);
        double trs = sum.at<int>(box.y+i,box.x+box.width);
        double tls = sum.at<int>(box.y+i,box.x);
        double sumI = (brs+tls-trs-bls);
        mean +=sumI;
        rowVar += sumI *sumI;
    }
    mean = mean / box.area();
    rowVar = rowVar /( box.width * box.area()) - mean *mean;
    return rowVar;
}
BoundingBox  Detection::getDetectBox(){
    return lastbox;
}
bool Detection::getReinitialization(){
    return adjustingTypes;
}
#ifdef using_pyramid
void Detection::constructImagePyramids(const cv::Mat & image) //image gray image
{
    cv::Mat imagetmp = image.clone(); //imagetmp color
    timages.clear();
    timages.resize(scalespyramids.size());
#if defined(FernFeature) || defined(CompressiveFeature)|| defined(HaarLikeFeature)
    filterImages.resize(scalespyramids.size());
    //HOG process orignal images
#endif

#pragma omp parallel for num_threads(6)
    for(int i = 0; i < scalespyramids.size(); i++)  //Hog color Other gray
    {
        if(scalespyramids[i] == imagetmp.size() )
            timages[i] = imagetmp.clone();
        else
            cv::resize(imagetmp,  timages[i], scalespyramids[i]);
#if defined(FernFeature)  //only process orignal image
        //cv::equalizeHist(timages[i], filterImages[i]); //直方图均衡化
        GaussianBlur(timages[i],filterImages[i],Size(9,9),1.5);
#elif defined(CompressiveFeature)|| defined(HaarLikeFeature)
        cv::equalizeHist(timages[i], filterImages[i]); //直方图均衡化
#endif
    }

#if defined(FernFeature)  //only process orignal image
    int index = scaleindex.at(1.0);
    integral(timages[index],iisum[index],iisqsum[index]);
    //for(int i = 0; i < timages.size(); i++)  //Hog color Other gray
    // integral(timages[i],iisum[i],iisqsum[i]);
#elif defined(CompressiveFeature)
#pragma omp parallel for num_threads(4)
    for(int i = 0; i < timages.size(); i++)  //Hog color Other gray
        integral(timages[i],iisum[i],iisqsum[i]);
#elif defined(HaarLikeFeature) //Only for Haar Features and Compressive Heafure
#pragma omp parallel for num_threads(4)
    for(int i = 0; i < timages.size(); i++)  //Hog color Other gray
        integral(timages[i],iisum[i],iisqsum[i]);
#elif defined(HOGFeature)
    int index = scaleindex.at(1.0);
    integral(timages[index],iisum[index],iisqsum[index]);
#endif
}
#endif

void Detection::determinateTrackingState(cv::Mat & image, BoundingBox & tboundingBox, bool & tracked,bool & dtracked, bool & tvalid,float & tconf)
{
    //// tracking successful or failed
    tconf = 0;
    Mat tpattern;
    Scalar mean, stdev;
    Rect bb;
    bb.x = max(int(tboundingBox.x),0);
    bb.y = max(int(tboundingBox.y),0);
    bb.width  = min(min(double(image.cols-bb.x),tboundingBox.width),min(tboundingBox.width,tboundingBox.br().x));
    bb.height = min(min(double(image.rows-bb.y),tboundingBox.height),min(tboundingBox.height,tboundingBox.br().y));
    if( bb.width <= 0 ||  bb.height <=0 )
    {
        std::cout<<"Tracking state1(width): "<<tracked<<"   ";
        tvalid = false;
      //  tracked = false;
        std::cout<<"Tracking state2(width): "<<tracked<<"   "<<std::endl;
        processNumsofLevels++;
        return;
    }
    else
    {
#ifdef using_pyramid
        int index = scaleindex.at(1.0);    // orignal image detection
        if(!filterVar(BoundingBox(bb),iisum[index],iisqsum[index])) //false pass var test
            return;
        getPattern(timages[index](bb),tpattern,mean,stdev);
#else
        if(!filterVar(BoundingBox(bb),iisum,iisqsum))
        {
            //double var1 = getVar(BoundingBox(bb),iisum,iisqsum);
            //std::cout<<"Tracking state1: "<<tracked<<"   ";
            tvalid = false;
            //tracked = false;
           // std::cout<<"Tracking state2: "<<tracked<<"   "<<var1<<std::endl;
            processNumsofLevels++;
            return;
        }
        getPattern(timages(bb),tpattern,mean,stdev);
#endif
        int scaleIndex = middleClassifier(BoundingBox(bb),dtracked);
        float dummy;
        vector<int> isin;
        classifier.NNConf(tpattern,isin,dummy,tconf); //Conservative Similarity
        //tconf = (tconf + dummy)/2.0;
        if(( (dtracked && tracked) && tconf>classifier.thr_nn_valid -0.05 )/*||(tconf>classifier.thr_nn_valid -0.1)*/)
            tvalid =true;
       // float nn_th = classifier.getNNTh() ; // 大于0.7才算成功
        if(dtracked && tconf>classifier.thr_nn_valid) tracked = true;
        if(tvalid || (tracked &&dtracked)) // 当前尺度层
        {
            levelOfTracker = scaleIndex;
            processNumsofLevels = 2;
        }
        else
            processNumsofLevels++;
    }
}
int Detection::middleClassifier(BoundingBox bb ,bool & is_pass)
{
    cv::Mat warp;
#ifdef using_pyramid
    int index = scaleindex.at(1.0);
#if defined(FernFeature) || defined(CompressiveFeature) ||defined(HaarLikeFeature)
    warp = filterImages[index](bb).clone();
#elif defined(HOGFeature)
    warp = timages[index](bb).clone();
#endif

#else
    warp = filterImages(bb).clone();
#endif

    int scaleIndex = -1;
    double minfactor = FLT_MAX;
    for(int k = 0 ; k < scalesizes.size(); k++) // 在原始图像上的box大小
    {
        double factor = fabs(std::sqrt(double(bb.area())/ double(scalesizes[k].area())) -1.0);
        if(factor < minfactor)
        {
            scaleIndex = k;
            minfactor = factor;
        }
    }

#ifdef using_pyramid
    cv::Size sizetmp(gridpyramids[0].width,gridpyramids[0].height); //将trackingROI规划到有的尺度
    cv::resize(warp,warp,sizetmp); //转换到金字塔的长宽度
#else
    cv::resize(warp,warp,scalesizes[scaleIndex]); //trackingROI调整大小到滑动窗口中有的尺度
#endif

#if defined(FernFeature)
    int numtrees = classifier.getNumStructs();
    float fern_th = classifier.getFernTh();
    vector <int> ferns(numtrees);
#ifdef using_pyramid
    classifier.getFeatures(warp,ferns);
#else
    classifier.getFeaturesScale(warp,scaleIndex,ferns);
#endif
    float conf = classifier.measure_forest(ferns);
    if (conf>numtrees*fern_th)
        is_pass = true;
#elif defined(CompressiveFeature)
    cv::Mat patch;
    cv::integral(warp,patch);
    std::vector<std::vector<float> > ctfs;
    ctfs.resize(1);
#ifdef using_pyramid
    classifier.compDetector.getFeatureCompressive(patch,ctfs[0]);
#else
    classifier.compDetector.getFeatureCompressiveScale(patch,scaleIndex,ctfs[0]);
#endif
    std::vector<float> prob;
    classifier.compDetector.Milclassify(ctfs,prob);
    if(prob[0] > 0)
        is_pass =true;
#elif defined(HaarLikeFeature)
    std::vector<cv::Mat> patch;
    patch.resize(1);
    std::vector<cv::Mat> features;
    features.push_back(cv::Mat_<float>(1,classifier.haarfeature.featureNum));
    cv::integral(warp,patch[0]);
#ifdef using_pyramid
    classifier.haarfeature.getFeatureHaar(patch,features);
#else
    std::vector<int> scales(1,scaleIndex);
    classifier.haarfeature.getFeatureHaarScale(patch,features,scales);
#endif
    std::vector<float> prob(1, 0.0);
    classifier.haarfeature.classifier(features,prob);
    if(prob[0] > 0)
        is_pass = true;
#elif defined(HOGFeature)
    std::vector<cv::Mat> features;
    features.resize(1);
    classifier.fhogfeature.getFeatureHog(warp,features[0]);
    std::vector<float> prob(1, 0.0);
    classifier.fhogfeature.classifier(features,prob);
    if(prob[0] > 0)
        is_pass = true;
#endif
    return scaleIndex;
}

void Detection::detectProcess(cv::Mat & image, BoundingBox & tboundingBox, bool & tracked)
{
    double t = (double) cv::getTickCount();

    frameImage = image;
    adjustingTypes = NotChangingTrackingSate;
#ifdef using_pyramid
    constructImagePyramids(image);
    if(is_debug)
    {
        t=(double)getTickCount()-t;
        printf("pyramid in %gms\n", t*1000/getTickFrequency()); //400ms
        t = (double) cv::getTickCount();
    }
#else
    timages = image;
    GaussianBlur(timages,filterImages,Size(9,9),1.5);
    integral(timages,iisum,iisqsum);
#endif
    vector<BoundingBox> cbb;
    vector<float> cconf;
    int confident_detections=0;
    int didx;
    float tconf = 0;
    bool  tvalid = false ;
    if(tracked)
        tvalid = lastvalid;
    BoundingBox bbnext = tboundingBox;
    bool dtracked = false;
    determinateTrackingState(image,tboundingBox,tracked,dtracked,tvalid,tconf);
    if(is_debug)
        std::cout<<"tconf: "<<tconf<<" tvalid: "<<tvalid<<" tracked: "<<tracked<<" dtracked: "<<dtracked<<std::endl;
    detect();
    tmplastvalid = false;
    if(tracked)
    {
        lastvalid = tvalid;
        lastconf  = tconf;
        bbnext    = tboundingBox;//tboundingBox ok
        if(detected){                                            //   if Detected
            clusterConf(dbb,dconf,cbb,cconf);                    //   cluster detections
            if(is_debug)
                printf("Found %d clusters\n",(int)cbb.size());
            float maxconf = -FLT_MAX;
            int maxindex = -1;
            float maxNearConf =  -FLT_MAX;
            int  maxNearindex = -1;
            float maxFastConf =  -FLT_MAX;
            int  maxFastindex = -1;
            if(is_debug)
                std::cout<<"conf:  ";
            for (int i=0;i<cbb.size();i++){
                //
                if(bbOverlap(tboundingBox,cbb[i])<0.5 && cconf[i]>tconf){ //  Get index of a clusters that is far from tracker and are more confident than the tracker
                    confident_detections++;
                    didx=i; //detection index
                }
                if(bbOverlap(tboundingBox,cbb[i])<0.5 &&  cconf[i]>maxFastConf) //  fast detection
                {
                    maxFastConf = cconf[i];
                    maxFastindex = i;
                }
                if(bbOverlap(tboundingBox,cbb[i])>=0.7 && cconf[i]>maxNearConf) //  near detection
                {
                    maxNearConf = cconf[i];
                    maxNearindex = i;
                }
                if(cconf[i] > maxconf && cconf[i] > tconf)
                {
                    maxconf = cconf[i];
                    maxindex = i;
                }
                if(is_debug)
                    std::cout<<cconf[i]<<" ";
            }
            if(is_debug){
                std::cout<<std::endl;
                std::cout<<"maxindex:  "<<maxindex<<std::endl;
            }
            //if there is ONE such a cluster, re-initialize the tracker
             if (confident_detections==1  && maxFastindex == didx /*&& maxFastConf > maxNearindex+0.05*/){ //maxconf > tconfig and 同时bbOverlap < 0.5
                 if(is_debug)
                     printf("Found a better match(location change)..reinitializing tracking\n");
                 // if((! & tconf < 0.6) || cconf[didx] > tconf+0.15) //maxconf > tconfig and 同时bbOverlap < 0.5
                 {
                     bbnext =cbb[didx];
                     adjustingTypes = ReInitialization;
                     lastvalid=false;
                 }
                 if(is_debug)
                     std::cout<<"tracked box"<<tboundingBox<<"  bbnext box"<<bbnext<<std::endl;
            }
            else{
                 // tconf 最大  或者是有最大的maxconf 同时bbOverlap > 0.5
                 if(is_debug)
                     printf("%d confident cluster was found\n",confident_detections);
                int cx=0,cy=0,cw=0,ch=0;
                int close_detections=0;
                float averageConf = 0;
                for (int i=0;i<dbb.size();i++){
                    if(bbOverlap(tboundingBox,dbb[i])>0.7){  //重合度大于0.7 才学习
                        cx += dbb[i].x;
                        cy += dbb[i].y;
                        cw += dbb[i].width;
                        ch += dbb[i].height;
                        close_detections++;
                        averageConf += dconf[i];
                        //printf("weighted detection: %d %d %d %d\n",dbb[i].x,dbb[i].y,dbb[i].width,dbb[i].height);
                    }
                }

                // weighted average trackers trajectory with the close detections //tracking update too slowly
                if (close_detections>0){
                    float factor = 10;
                    bbnext.x = cvRound((float)(factor*tboundingBox.x+cx)/(float)(factor+close_detections));
                    bbnext.y = cvRound((float)(factor*tboundingBox.y+cy)/(float)(factor+close_detections));
                    bbnext.width = cvRound((float)(factor*tboundingBox.width+cw)/(float)(factor+close_detections));
                    bbnext.height =  cvRound((float)(factor*tboundingBox.height+ch)/(float)(factor+close_detections));
                    //    printf("Tracker bb: %d %d %d %d\n",tboundingBox.x,tboundingBox.y,tboundingBox.width,tboundingBox.height);
                    //    printf("Average bb: %d %d %d %d\n", bbnext.x, bbnext.y, bbnext.width, bbnext.height);
                    //    printf("Weighting %d close detection(s) with tracker..\n",close_detections);
                    //    lastvalid = true;  // detection and tracking overlaps

                    if(is_debug)
                        printf("ReCorrecting tconf: %f \n",tconf);
                    adjustingTypes = ReCorrecting;
                    averageConf = averageConf/close_detections;
                    if(averageConf > classifier.thr_nn_valid - 0.05)
                        lastvalid = true;
                    if(is_debug)
                        std::cout<<"tracked box"<<tboundingBox<<"  bbnext box"<<bbnext<<"lastvalid: "<<lastvalid<<" averageConf:  "<<averageConf<<std::endl;
                }
                else{
                    if(is_debug){
                        printf("Not change  tconf: %f\n", tconf);
                        if(cbb.size()>0)
                            std::cout<<"tracked box"<<tboundingBox<<"  bbnext box"<<cbb[maxindex]<<std::endl;
                    }
                }
            }
        }
    }
    else{
        if(is_debug)//   If NOT tracking
            printf("Not tracking..\n");
        lastvalid = false;
        if(detected){                           //  and detector is defined
            clusterConf(dbb,dconf,cbb,cconf);   //  cluster detections
            if(is_debug)
                printf("Found %d clusters\n",(int)cbb.size());
            if (cconf.size()==1){
                bbnext=cbb[0];
                lastconf=cconf[0];
                if(is_debug)
                    printf("Confident detection..reinitializing tracker\n");
                // lastboxfound = true;
                adjustingTypes = ReInitialization;
            }
        }
    }
    //！加一个判断,就是上一帧检测到，目标也被检测到，检测和跟踪区域非常接近，则直接可以学习
    // 前后两帧检测到，然后跟踪的比较相近就学习， bbOverlap(bbnext,lastbox)>0.5 这个有点问题
    // 检测和跟踪应该比较近
    // if(lastdetect && detected)//&& bbOverlap(bbnext,lastbox)>0) //
    //     lastvalid = true;
    lastbox  = bbnext;
    lastdetect = detected;
    //t=(double)getTickCount()-t;
    // printf("detection processing time %gms\n", t*1000/getTickFrequency());
}


void Detection::detect(){
    dbb.clear();
    dconf.clear();
    candidatedbb.clear();
    candidatedconf.clear();

    dt.bb.clear();
    tmp.candidateIndex.clear();
    double maxtmpconf = -FLT_MAX;
    Mat patch;
    int a=0;
    double t = (double)getTickCount();
#if defined(FernFeature)
    float conf;
    int numtrees = classifier.getNumStructs();
    float fern_th = classifier.getFernTh();
    vector <int> ferns(numtrees);
#elif defined(CompressiveFeature)
    fX.clear();
    vector<float> ctfs;
#elif defined(HaarLikeFeature)
    classifier.haarSamples.resize(grid.size()); //清空以前的数据；
    tmp.candidateIndex.clear();
#endif
    directionFilterValid = 0;
 //   cv::Mat VariantFilterimage = timages[index].clone();
#ifdef using_pyramid
   // processNumsofLevels = 5;
    int index = scaleindex.at(1.0);    // orignal image detection
    for (int i=0;i<gridpyramids.size();i++){//FIXME: BottleNeck

        if(grid[i].sidx > levelOfTracker + processNumsofLevels
                || grid[i].sidx < levelOfTracker - processNumsofLevels)
        {
#if defined(FernFeature)
#elif defined(CompressiveFeature)
#elif defined(HaarLikeFeature)
            classifier.haarSamples[i].release(); //消除以前
#endif
            tmp.conf[i]= -FLT_MAX;
            continue;
        }

        BoundingBox & bb = gridpyramids[i];
        if(filterVar(grid[i],iisum[index],iisqsum[index])){  //pyramids
            a++;
#if defined(FernFeature)
            patch = filterImages[bb.sidx](bb);
            classifier.getFeatures(patch,ferns);
            conf = classifier.measure_forest(ferns);
            tmp.conf[i]=conf;
            tmp.patt[i]=ferns;
            if (conf>numtrees*fern_th){
                dt.bb.push_back(i);
                if(tmp.conf[i] > maxtmpconf)
                    maxtmpconf = tmp.conf[i];
            }
#elif defined(CompressiveFeature)
            patch =iisum[bb.sidx](bb);
            classifier.compDetector.getFeatureCompressive(patch,ctfs);
            fX.push_back(ctfs);
            tmp.conf[i] = -1;
            tmp.patt[i] = ctfs;
#elif defined(HaarLikeFeature)
            classifier.haarSamples[i] = iisum[bb.sidx](bb);
            //tmp.candidateIndex.push_back(i);
            tmp.conf[i]= 0;
#elif defined(HOGFeature)
            classifier.fhogfeature.getFeatureHog(timages[bb.sidx](bb),tmp.patt1[i]);
            tmp.conf[i]=-1;
#endif
            // drawBox(VariantFilterimage,grid[i]);
        }
        else
        {
#if defined(FernFeature)
#elif defined(CompressiveFeature)
#elif defined(HaarLikeFeature)
            classifier.haarSamples[i].release(); //消除以前
#endif
            tmp.conf[i]= -FLT_MAX;
        }
    }

#else
    processNumsofLevels = 5;
    for (int i=0;i<grid.size();i++){
        if(grid[i].sidx > levelOfTracker + processNumsofLevels
                || grid[i].sidx < levelOfTracker - processNumsofLevels)
        {
#if defined(FernFeature)
#elif defined(CompressiveFeature)
#elif defined(HaarLikeFeature)
            classifier.haarSamples[i].release(); //消除以前
#endif
            tmp.conf[i]= -FLT_MAX;
            continue;
        }
        if (filterVar(grid[i],iisum,iisqsum)){
            a++;
#if defined(FernFeature)
            patch = filterImages(grid[i]);
            classifier.getFeaturesScale(patch,grid[i].sidx,ferns);
            conf = classifier.measure_forest(ferns);
            tmp.conf[i]=conf;
            tmp.patt[i]=ferns;
            if (conf>numtrees*fern_th){
                dt.bb.push_back(i);
                if(tmp.conf[i] > maxtmpconf)
                    maxtmpconf = tmp.conf[i];
            }
#elif defined(CompressiveFeature)
            patch =iisum((grid[i]));
            classifier.compDetector.getFeatureCompressiveScale(iisum((grid[i])),grid[i].sidx,ctfs);
            fX.push_back(ctfs);
            tmp.conf[i] = -1;
            tmp.patt[i] = ctfs;
#elif defined(HaarLikeFeature)
            classifier.haarSamples[i] = iisum(grid[i]);
            tmp.conf[i]=0;
#elif defined(HOGFeature)
            classifier.fhogfeature.getFeatureHog(timages(grid[i]),tmp.patt1[i]);
            tmp.conf[i]=-1;
#endif
        }
        else
        {
#if defined(FernFeature)
#elif defined(CompressiveFeature)
#elif defined(HaarLikeFeature)
            classifier.haarSamples[i].release(); //消除以前
#endif
            tmp.conf[i]= -FLT_MAX;
        }
    }
#endif
    if(is_debug)
        printf("Bounding boxes passed the variance filter: %d  grid size:  %d filterNums: %d \n",a, grid.size(),directionFilterValid);
#if defined(FernFeature)

#elif defined(CompressiveFeature)
    std::vector<float> prob;
    classifier.compDetector.Milclassify(fX,prob);
    //classifier.compDetector.radioClassifier(ct_fx,prob);
    int indexBox = 0;
    for (int i=0;i<grid.size();i++)
    {
        // pass variant
        if (tmp.conf[i]==-1){
            tmp.conf[i] = prob[indexBox];
            if (prob[indexBox]>0){
                dt.bb.push_back(i);
                if(tmp.conf[i] > maxtmpconf)
                    maxtmpconf = tmp.conf[i];
            }
            indexBox++;
        }
    }
#elif defined(HaarLikeFeature)
#ifdef using_pyramid
    classifier.haarfeature.getFeatureHaar(classifier.haarSamples,tmp.patt1);
#else
    classifier.haarfeature.getFeatureHaarScale(classifier.haarSamples,tmp.patt1,BoundingBoxScales);
    if(is_debug)
    {
        t=(double)getTickCount()-t;
        printf("frature extracking %gms\n", t*1000/getTickFrequency());
        t = (double) getTickCount();
    }
#endif
    classifier.haarfeature.classifier(tmp.patt1,tmp.conf); // tmp.conf < -1  or FLT_MIN  0 represent valid>
    for (int i=0;i<grid.size();i++)
    {
        if(tmp.conf[i]>0)
        {
            dt.bb.push_back(i);
            if(tmp.conf[i] > maxtmpconf)
                maxtmpconf = tmp.conf[i];
        }
    }
#elif defined(HOGFeature)
    classifier.fhogfeature.classifier(tmp.patt1,tmp.conf);
    for (int i=0;i<grid.size();i++)
    {
        if(tmp.conf[i]!=-FLT_MAX)
        {
            dt.bb.push_back(i);
            if(tmp.conf[i] > maxtmpconf)
                maxtmpconf = tmp.conf[i];
        }
    }
#endif
    int detections = dt.bb.size();
    if(is_debug)
       // imshow("VariantFilterimage",VariantFilterimage);
        printf("%d Initial detection from Classifiers(Fern,MIL,BOOST,SVM)\n",detections);
    int selectedNums = 100;
    if (detections>selectedNums){
        std::partial_sort(dt.bb.begin(),dt.bb.begin()+selectedNums,dt.bb.end(),CComparator(tmp.conf));
        //std::nth_element(dt.bb.begin(),dt.bb.begin()+selectedNums,dt.bb.end(),CComparator(tmp.conf)); //选取100个最相似的
        dt.bb.resize(selectedNums);
        detections=selectedNums;
    }
    else
        std::sort(dt.bb.begin(),dt.bb.end(),CComparator(tmp.conf));
    double mintmpconf = tmp.conf[dt.bb[selectedNums-1]];
#ifdef using_pyramid
    int scaleIndex = scaleindex.at(1.0);     // orignal scale
    cv::Mat img =timages[scaleIndex].clone();
#else
    cv::Mat img = timages.clone();
#endif
    for (int i=0;i<detections;i++){
        drawBox(img,grid[dt.bb[i]]);
    }
    if (detections==0){
        detected=false;
        if(is_debug) imshow("detections",img);
        return;
    }
    if(is_debug)
    {
        printf("Object detector made %d detections ",detections);
        t=(double)getTickCount()-t;
        printf("in %gms\n", t*1000/getTickFrequency());
    }

#if defined(FernFeature)
    dt.patt = vector<vector<int> >(detections,vector<int>(classifier.getNumStructs(),0));        //  Corresponding codes of the Ensemble Classifier
#elif defined(CompressiveFeature)
    dt.patt = vector<vector<float> >(detections,vector<float>(classifier.compDetector.featureNum,0));        //  Corresponding codes of the Ensemble Classifier
#elif defined(HaarLikeFeature)
    dt.patt1 = vector<cv::Mat >(detections);
#elif defined(HOGFeature)
    dt.patt1 = vector<cv::Mat >(detections);
#endif

    dt.conf1 = vector<float>(detections);                                //  Relative Similarity (for final nearest neighbour classifier)
    dt.conf2 =vector<float>(detections);                                 //  Conservative Similarity (for integration with tracker)
    dt.isin = vector<vector<int> >(detections,vector<int>(3,-1));        //  Detected (isin=1) or rejected (isin=0) by nearest neighbour classifier
    dt.patch = vector<Mat>(detections,Mat(patch_size,patch_size,CV_32F));//  Corresponding patches
    int idx;
    Scalar mean, stdev;
    float nn_th = classifier.getNNTh() ; // 大于0.7才算成功
    float factor = 0.0;
#ifdef using_pyramid
    for (int i=0;i<detections;i++){                                         //  for every remaining detection
        idx=dt.bb[i];                                                       //  Get the detected bounding box index
        patch = timages[gridpyramids[idx].sidx](gridpyramids[idx]);
        getPattern(patch,dt.patch[i],mean,stdev);                //  Get pattern within bounding box
        classifier.NNConf(dt.patch[i],dt.isin[i],dt.conf1[i],dt.conf2[i]);  //  Evaluate nearest neighbour classifier
#if defined(FernFeature)
        dt.patt[i]=tmp.patt[idx];
#elif defined(CompressiveFeature)
        dt.patt[i]=tmp.patt[idx];
#elif defined(HaarLikeFeature)
        dt.patt1[i]=tmp.patt1[idx];
#elif defined(HOGFeature)
        dt.patt1[i]=tmp.patt1[idx];
#endif
        if(is_debug)
            printf("Testing feature %d, conf:%f isin:(%d|%d|%d)\n",i,dt.conf1[i],dt.isin[i][0],dt.isin[i][1],dt.isin[i][2]);
        if (dt.conf1[i]>nn_th -  tmp.conf[idx]/12.0 * factor /*(tmp.conf[idx]-mintmpconf)/(maxtmpconf-mintmpconf) * factor*/){                                               //  idx = dt.conf1 > tld.model.thr_nn; % get all indexes that made it through the nearest neighbour
           // if( tmp.conf[idx] >0.9){
            candidatedbb.push_back(grid[idx]);                                         //  BB    = dt.bb(:,idx); % bounding boxes
            candidatedconf.push_back(dt.conf2[i]); // }                                   //  Conf  = dt.conf2(:,idx); % conservative confidences
            if(dt.conf1[i]>nn_th){ //strict limit
                dbb.push_back(grid[idx]);                                         //  BB    = dt.bb(:,idx); % bounding boxes
                dconf.push_back(dt.conf2[i]) /*+ dt.conf1[i])*0.5))*/;                                     //  Conf  = dt.conf2(:,idx); % conservative confidences
            }
        }
    }
#else
    for (int i=0;i<detections;i++){                                         //  for every remaining detection
        idx=dt.bb[i];                                                       //  Get the detected bounding box index
        patch = timages(grid[idx]);
        getPattern(patch,dt.patch[i],mean,stdev);                           //  Get pattern within bounding box
        classifier.NNConf(dt.patch[i],dt.isin[i],dt.conf1[i],dt.conf2[i]);  //  Evaluate nearest neighbour classifier
#if defined(FernFeature)
        dt.patt[i]=tmp.patt[idx];
#elif defined(CompressiveFeature)
        dt.patt[i]=tmp.patt[idx];
#elif defined(HaarLikeFeature)
        dt.patt1[i]=tmp.patt1[idx].clone();
#elif defined(HOGFeature)
        dt.patt1[i]=tmp.patt1[idx].clone();
#endif
        if(is_debug)
            printf("Testing feature %d, rconf:%f cconf:%f isin:(%d|%d|%d)\n",i,dt.conf1[i],dt.conf2[i],dt.isin[i][0],dt.isin[i][1],dt.isin[i][2]);
        if (dt.conf1[i]>nn_th - tmp.conf[idx]/maxtmpconf * factor){                                               //  idx = dt.conf1 > tld.model.thr_nn; % get all indexes that made it through the nearest neighbour
            candidatedbb.push_back(grid[idx]);                                         //  BB    = dt.bb(:,idx); % bounding boxes
            candidatedconf.push_back(dt.conf2[i]);                                     //  Conf  = dt.conf2(:,idx); % conservative confidences
            if(dt.conf1[i]>nn_th){
                dbb.push_back(grid[idx]);                                         //  BB    = dt.bb(:,idx); % bounding boxes
                dconf.push_back((dt.conf2[i] + dt.conf1[i])*0.5);                                     //  Conf  = dt.conf2(:,idx); % conservative confidences
            }
        }
    }
#endif
    bool candinate = false;
    if(dbb.size() ==0 && candidatedbb.size() >0)
    {
        if(is_debug)  std::cout<<"using candidatedabb"<<std::endl;
          dbb=candidatedbb;
          dconf = candidatedconf;
          candinate = true;
    }

    if(is_debug){
        int thick =1;
        if(candinate) thick =2;
        for (int i=0;i<dbb.size();i++){
            drawBox(img,dbb[i], cv::Scalar(0,0,0),thick);
        }
        imshow("detections",img);
    }
    if (dbb.size()>0){
        if(is_debug)
            printf("Found %d NN matches\n",(int)dbb.size());
        detected=true;
    }
    else{
        if(is_debug)
            printf("No NN matches found.\n");
        detected=false;
    }
   /* static int frameNumber =0;
    frameNumber++;
    std::stringstream ss;
    ss.str("");
    ss.clear();
    ss<<"/home/nubot/result/detection"<<frameNumber<<".jpg";
    cv::imwrite(ss.str(),img);
    ss.str("");
    ss.clear();
    ss<<"/home/nubot/result/detectionoriginal"<<frameNumber<<".jpg";
    cv::imwrite(ss.str(),timages);*/
}

void Detection::evaluate(){
}

void Detection::learn(){

    double t = (double) getTickCount();
    if (!lastvalid && !tmplastvalid)
        return;
    if(is_debug) printf("[Learning] \n");

    BoundingBox bb;
    bb.x = max(lastbox.x,0.0);
    bb.y = max(lastbox.y,0.0);
    bb.width = min(min(frameImage.cols-lastbox.x,lastbox.width),min(lastbox.width,lastbox.br().x));
    bb.height = min(min(frameImage.rows-lastbox.y,lastbox.height),min(lastbox.height,lastbox.br().y));
    Scalar mean, stdev;
    Mat pattern;
#ifdef using_pyramid
    int index = scaleindex.at(1.0);    // orignal scale
    getPattern(timages[index](bb),pattern,mean,stdev);
#else
    getPattern(timages(bb),pattern,mean,stdev);
#endif
    bool is_pass=false;
    middleClassifier(bb,is_pass);
   // middleClassifier(timages(bb) ,bool & pass, int scaleIndex);
    vector<int> isin;
    float dummy, conf;
    float maxP = classifier.NNConf(pattern,isin,conf,dummy);
    if (conf<0.5 && maxP > 0.6) {
        if(is_debug)
            printf("Fast change..not training\n");
        lastvalid =false;
        return;
    }
    if (pow(stdev.val[0],2)<var){
        if(is_debug)
            printf("Low variance..not training\n");
        lastvalid=false;
        return;
    }
    if(!is_pass)
    {
        if(is_debug)
            printf("Middle classifier..not training\n");
       lastvalid=false;
       return;
    }
    if(isin[2]==1){
        if(is_debug)
            printf("Patch in negative data..not traing");
        lastvalid=false;
        return;
    }
    /// Data generation
    for (int i=0;i<grid.size();i++){
        grid[i].overlap = bbOverlap(lastbox,grid[i]);
    }
    good_boxes.clear();
    bad_boxes.clear();
    getOverlappingBoxes(lastbox,num_closest_update);
    if (good_boxes.size()>0)
        generatePositiveData(num_warps_update);
    else{
        lastvalid = false;
        if(is_debug)
            printf("No good boxes..Not training");
        return;
    }
    random_shuffle(bad_boxes.begin(),bad_boxes.end());
    int idx = 0;
#if defined(FernFeature)
    vector<pair<vector<int>,int> > fern_examples;
    fern_examples.reserve(pX.size()+bad_boxes.size());
    fern_examples.assign(pX.begin(),pX.end()); // positive sample
    for (int i=0;i<bad_boxes.size();i++){
        idx=bad_boxes[i];
        if (tmp.conf[idx]>=1){ // noly consider negative positive samples
            fern_examples.push_back(make_pair(tmp.patt[idx],0));
        }
    }
#elif defined(CompressiveFeature)
    nX.clear();
    for (int i=0;i<bad_boxes.size();i++){
        idx=bad_boxes[i];
        if (tmp.conf[idx]>=0){
            nX.push_back(tmp.patt[idx]);
        }
    }
#elif defined(HaarLikeFeature)
    nX.clear();
    for (int i=0;i<bad_boxes.size();i++){
        idx=bad_boxes[i];
        if (tmp.conf[idx] >= 0){
            nX.push_back(tmp.patt1[idx]);
        }
    }
#elif defined(HOGFeature)
    nX.clear();
    for (int i=0;i<bad_boxes.size();i++){
        idx=bad_boxes[i];
        if (tmp.conf[idx]>=0){
            nX.push_back(tmp.patt1[idx]);
        }
    }
#endif
    nn_posnums = nn_examples.size();
    for (int i=0;i<dt.bb.size();i++){
        idx = dt.bb[i];
        if (bbOverlap(lastbox,grid[idx]) < bad_overlap)
            nn_examples.push_back(dt.patch[i]);
    }
#if defined(FernFeature)
    classifier.trainF(fern_examples,2);
#elif defined(CompressiveFeature)
    classifier.compDetector.MilClassifierUpdate(pX,nX);
#elif defined(HaarLikeFeature)
    if(nX.size() >300)
        nX.resize(300);
    classifier.haarfeature.classifierUpdate(pX,nX);
#elif defined(HOGFeature)
    if(nX.size() >200)
        nX.resize(200);
    classifier.fhogfeature.classifierUpdate(pX,nX);
#endif
    // classifier.compDetector.classifierUpdate(ct_px,ct_nx);
    classifier.trainNN(nn_examples,nn_posnums);
    if(is_debug)  classifier.show();
    frameImage.release();
    if(is_debug) {
        t=(double)getTickCount()-t;
        printf("learning time %gms\n", t*1000/getTickFrequency());
    }
}


void Detection::buildGrid(const cv::Mat& img, const cv::Rect& box){
    const float SHIFT = 0.05 *2;
    int width, height, min_bb_side;
    int min_bb_edge = min(box.height,box.width);
    BoundingBox bbox;
#ifdef using_pyramid
    int scales_number = 0;
#endif
    Size scale;
    int sc = 0;
    for (int s=0;s<scales.size();s++){  // 尺度太小的目标无须考虑, 尺度从小到到大， 有些层数没有
        width  = round(box.width*scales[s]);
        height = round(box.height*scales[s]);
        min_bb_side = min(height,width);
        if (min_bb_side < min_win || width > img.cols || height > img.rows)
            continue;
        int img_cols = round(img.cols / scales[s]);
        int img_rows = round(img.rows / scales[s]);

        scale.width = width;
        scale.height = height;
        scalesizes.push_back(scale);

        // pyramids construction
        for (int y=0;y<img_rows-box.height;y+=round(SHIFT*min_bb_edge)){
            for (int x=0;x<img_cols- box.width;x+=round(SHIFT*min_bb_edge)){
                bbox.x = round(x * scales[s]); //
                bbox.y = round(y * scales[s]);
                bbox.width  = width;
                bbox.height = height;
                bbox.overlap = bbOverlap(bbox,BoundingBox(box));
                bbox.sidx = sc;
                grid.push_back(bbox);
                BoundingBoxScales.push_back(bbox.sidx); //record Box Scales
#ifdef using_pyramid
                bbox.x = x;
                bbox.y = y;
                bbox.width  = box.width;
                bbox.height = box.height;
                gridpyramids.push_back(bbox);
                scales_number++;
#endif
            }
        }
#ifdef using_pyramid
        // scale can be used to look up sc ,then look up scales_nums，scales_ori and so on
        scaleindex.insert(std::make_pair(scales[s], sc)); //scales[s]  与原始框的尺度比 Key
        scalesgridnums.push_back(scales_number);
        scalespyramids.push_back(cv::Size(img_cols,img_rows)); //金字塔图像的大小
#endif
        if(scales[s]==1) levelOfTracker = sc; // intial level
        sc++;
    }
        processNumsofLevels = 1;
}

float Detection::bbOverlap(const BoundingBox& box1,const BoundingBox& box2){
    if (box1.x > box2.x+box2.width) { return 0.0; }
    if (box1.y > box2.y+box2.height) { return 0.0; }
    if (box1.x+box1.width < box2.x) { return 0.0; }
    if (box1.y+box1.height < box2.y) { return 0.0; }

    float colInt =  min(box1.x+box1.width,box2.x+box2.width) - max(box1.x, box2.x);
    float rowInt =  min(box1.y+box1.height,box2.y+box2.height) - max(box1.y,box2.y);

    float intersection = colInt * rowInt;
    float area1 = box1.width*box1.height;
    float area2 = box2.width*box2.height;
    return intersection / (area1 + area2 - intersection);
}

void Detection::getOverlappingBoxes(const cv::Rect& box1,int num_closest){
    // bestbox using_pyramid
    float max_overlap = 0;
    for (int i=0;i<grid.size();i++){
        if (grid[i].overlap > max_overlap) {
            max_overlap = grid[i].overlap;
            best_box = grid[i];
#ifdef using_pyramid
            best_pyramidbox = gridpyramids[i];
#endif
        }
        if (grid[i].overlap > 0.7){
            good_boxes.push_back(i);
        }
        else if (grid[i].overlap < bad_overlap){
            bad_boxes.push_back(i);
        }
    }
    //Get the best num_closest (10) boxes and puts them in good_boxes
    if (good_boxes.size()>num_closest){
        // std::nth_element(good_boxes.begin(),good_boxes.begin()+num_closest,good_boxes.end(),OComparator(grid));
        std::partial_sort(good_boxes.begin(),good_boxes.begin()+num_closest,good_boxes.end(),OComparator(grid));
        good_boxes.resize(num_closest);
    }
    //std::sort(good_boxes.begin(),good_boxes.end(),)
#ifndef using_pyramid //not pyramids
    getBBHull(); //可能再不同的尺度下的图像；
#endif
}
//将所有的goodbox组合为一个大的
void Detection::getBBHull(){
    int x1=INT_MAX, x2=0;
    int y1=INT_MAX, y2=0;
    int idx;
    for (int i=0;i<good_boxes.size();i++){
        idx= good_boxes[i];
        x1=min(grid[idx].x,double(x1));
        y1=min(grid[idx].y,double(y1));
        x2=max(grid[idx].x+grid[idx].width,double(x2));
        y2=max(grid[idx].y+grid[idx].height,double(y2));
    }
    bbhull.x = x1;
    bbhull.y = y1;
    bbhull.width = x2-x1;
    bbhull.height = y2 -y1;
}
/* Generate Positive data
 * Inputs:
 * - good_boxes (bbP)
 * - best_box (bbP0)
 * - frame (im0)
 * Outputs:
 * - Positive fern features (pX)
 * - Positive NN examples (pEx)
 */
void Detection::generatePositiveData(int num_warps){
    nn_examples.clear();
    Scalar mean;
    Scalar stdev;
    cv::Mat_<uchar> warped;
    pX.clear(); // positive samples
    int idx;
    RNG& rng = theRNG();

#ifdef using_pyramid
    int index = scaleindex.at(1.0);    // orignal scale
    getPattern(timages[index](best_box),pEx,mean,stdev);
#else
    cv::Mat frame = timages; //orignal gray image
    cv::Mat img   = filterImages; // filter image
    getPattern(frame(best_box),pEx,mean,stdev);
    warped = img(bbhull).clone(); //必须使用clone
#endif

#if defined(FernFeature)
    vector<int> fern(classifier.getNumStructs());
    if (pX.capacity()<num_warps*good_boxes.size())
        pX.reserve(num_warps*good_boxes.size());
#elif defined(CompressiveFeature)
    vector<float> compressive(classifier.compDetector.featureNum);
#ifndef using_pyramid
    cv::Mat warpiisum = iisum(bbhull).clone();
#endif
#elif defined(HaarLikeFeature)
    classifier.haarSamples.resize(good_boxes.size() * num_warps);
    int countIndex = 0;
#ifndef using_pyramid
    cv::Mat warpiisum = iisum(bbhull).clone(); // orignal image
    std::vector<int> scalesCount(good_boxes.size() * num_warps,0);
#endif
#elif defined(HOGFeature)
    int countIndex = 0;
    pX.resize(good_boxes.size() * num_warps);
#endif
//#ifdef using_pyramid
    for (int i = 0; i < (int)good_boxes.size(); i++)
    {
        warped.release();
        idx=good_boxes[i];
#ifdef using_pyramid
        BoundingBox bb = gridpyramids[idx];        //金字塔图像 对应的原始图像爱那个
#if defined(FernFeature) || defined(CompressiveFeature) || defined(HaarLikeFeature)
        cv::Mat blurimage = filterImages[bb.sidx];
#endif
        cv::Mat originalimage = timages[bb.sidx];
#else
        BoundingBox bb = grid[idx];        //金字塔图像 对应的原始图像爱那个
        cv::Mat blurimage = filterImages;
        cv::Mat originalimage = timages;
#endif

        Size2f size = Size(bb.width,bb.height);

#if defined(FernFeature)
        warped = blurimage(bb).clone();  //must clone
#ifdef using_pyramid
        classifier.getFeatures(warped,fern);
#else
        classifier.getFeaturesScale(warped,bb.sidx,fern);
#endif
        pX.push_back(make_pair(fern,1));
#elif defined(CompressiveFeature)
        warped = blurimage(bb).clone();  //must clone
        cv::Mat warpiisumTmp;
        cv::integral(warped,warpiisumTmp);
#ifdef using_pyramid
        classifier.compDetector.getFeatureCompressive(warpiisumTmp,compressive);
#else
        classifier.compDetector.getFeatureCompressiveScale(warpiisumTmp,bb.sidx,compressive);
#endif
        pX.push_back(compressive);
#elif defined(HaarLikeFeature)
        warped = blurimage(bb).clone();  //must clone
        cv::integral(warped,classifier.haarSamples[countIndex]);
        countIndex++;
#elif defined(HOGFeature)
        warped = originalimage(bb).clone();  //must clone
        classifier.fhogfeature.getFeatureHog(warped,pX[countIndex]); // 有归一化
        countIndex++;
#endif
        getPattern(originalimage(bb),pEx,mean,stdev); //blur images
        nn_examples.push_back(pEx.clone());           // best images
        for (int j = 1; j < num_warps; j++)
        {
            Point2f center;
            center.x = (float)(bb.x + bb.width  * (0.5 + rng.uniform(-0.01, 0.01)));
            center.y = (float)(bb.y + bb.height * (0.5 + rng.uniform(-0.01, 0.01)));
            size.width  = (float)(bb.width  * rng.uniform(1.0-scale_init, 1.0+scale_init));
            size.height = (float)(bb.height * rng.uniform(1.0-scale_init, 1.0+scale_init));
            float angle = (float)rng.uniform(-angle_init, angle_init);
#if defined(FernFeature) || defined(CompressiveFeature) || defined(HaarLikeFeature)
            resample(blurimage, RotatedRect(center, size, angle), warped);
            for (int y = 0; y < warped.rows; y++)
                for (int x = 0; x < warped.cols; x++)
                    warped(y, x) += (uchar)rng.gaussian(noise_init);
#endif

#if defined(FernFeature)
#ifdef using_pyramid
            classifier.getFeatures(warped,fern);
#else
            classifier.getFeaturesScale(warped,bb.sidx,fern);
#endif
            pX.push_back(make_pair(fern,1));
#elif defined(CompressiveFeature)
             cv::integral(warped,warpiisumTmp);
#ifdef using_pyramid
            classifier.compDetector.getFeatureCompressive(warpiisumTmp,compressive);
#else
            classifier.compDetector.getFeatureCompressiveScale(warpiisumTmp,bb.sidx,compressive);
#endif
            pX.push_back(compressive);
#elif defined(HaarLikeFeature)
            cv::integral(warped,classifier.haarSamples[countIndex]);
            countIndex++;
#elif defined(HOGFeature)
            resample(originalimage, RotatedRect(center, size, angle), warped);
            classifier.fhogfeature.getFeatureHog(warped,pX[countIndex]);
            countIndex++;
#endif
            if(i <=increase_ncc_samples)
            {
                 float angleLearning = (float)rng.uniform(-angle_init *0.75, angle_init *0.75); //0.75
#if defined(FernFeature) || defined(CompressiveFeature) || defined(HaarLikeFeature)
                resample(originalimage, RotatedRect(center, size, angleLearning), warped);
#endif
                getPattern(warped,pEx,mean,stdev); //blur images
                nn_examples.push_back(pEx.clone());
            }
        }
    }

   /*
    for (int i=0;i<num_warps;i++){
        if(i > 0)
        {
            Size2f size;
            Point2f center;
            center.x = (float)(bbhull.x + (bbhull.width -1)  * (0.5 + rng.uniform(-0.01, 0.01)));
            center.y = (float)(bbhull.y + (bbhull.width -1)  * (0.5 + rng.uniform(-0.01, 0.01)));
            size.width  = (float)(bbhull.width  * rng.uniform(1.0-scale_init , 1.0+scale_init ));
            size.height = (float)(bbhull.height * rng.uniform(1.0-scale_init , 1.0+scale_init ));
            float angle = (float)rng.uniform(-angle_init+10, angle_init-10);
            resample(img, RotatedRect(center, size, angle), warped);
            for (int y = 0; y < warped.rows; y++)
                for (int x = 0; x < warped.cols; x++)
                    warped(y, x) += (uchar)rng.gaussian(noise_init); // orignal image
#if defined(FernFeature)
#elif defined(CompressiveFeature) || defined(HaarLikeFeature)
            cv::integral(warped,warpiisum);
#endif
            resample(timages, RotatedRect(center, size, angle), nccwarped);
        }
         cv::imshow("img1",nccwarped);
        for (int b = 0; b < good_boxes.size(); b++) {
            idx=good_boxes[b];
            Rect region(grid[idx].x-bbhull.x, grid[idx].y - bbhull.y, grid[idx].width, grid[idx].height);
#if defined(FernFeature)
            classifier.getFeaturesScale(warped(region), grid[idx].sidx, fern);
            pX.push_back(make_pair(fern,1));
#elif defined(CompressiveFeature)
            classifier.compDetector.getFeatureCompressiveScale(warpiisum(region),grid[idx].sidx,compressive);
            pX.push_back(compressive);
#elif defined(HaarLikeFeature)
            classifier.haarSamples[countIndex] = warpiisum(region);
            scalesCount[countIndex] = grid[idx].sidx;
            countIndex++;
#endif
            if(b <= increase_ncc_samples) //i num_warps
            {
                getPattern(nccwarped(region),pEx,mean,stdev); //orignal images for NCC
             //   nn_examples.push_back(pEx.clone());
            }
        }
    }
    */
#if defined(FernFeature)
#elif defined(CompressiveFeature)
#elif defined(HaarLikeFeature)
    //pX = std::vector<cv::Mat>(classifier.haarSamples.size(),cv::Mat_<float>(1,classifier.haarfeature.featureNum));
    // 上面initialization is error: all elements point to the same location
    pX.reserve( classifier.haarSamples.size());
    for(int i = 0 ; i < classifier.haarSamples.size() ; i++ )
        pX.push_back(cv::Mat_<float>(1,classifier.haarfeature.featureNum));
#ifdef using_pyramid
    classifier.haarfeature.getFeatureHaar(classifier.haarSamples,pX);
#else
    classifier.haarfeature.getFeatureHaarScale(classifier.haarSamples,pX,scalesCount); //no pyramids
#endif
#elif defined(HOGFeature)
    // classifier.fhogfeature.getFeatureHog(classifier.fhogSamples,pX);
#endif
    if(is_debug)
        printf("Positive examples generated: (ferns Haar HOG) :%d NN:1\n",(int)pX.size());
}
void Detection::resample(const Mat& img, const Rect2d& r2, Mat_<uchar>& samples)
{
    Mat_<float> M(2, 3);
    M(0, 0) = (float)(samples.cols / r2.width); M(0, 1) = 0.0f; M(0, 2) = (float)(-r2.x * samples.cols / r2.width);
    M(1, 0) = 0.0f; M(1, 1) = (float)(samples.rows / r2.height); M(1, 2) = (float)(-r2.y * samples.rows / r2.height);
    warpAffine(img, samples, M, samples.size());
}
void Detection::resample(const Mat& img, const RotatedRect& r2, Mat_<uchar>& samples)
{
    Mat_<float> M(2, 3), R(2, 2), Si(2, 2), s(2, 1), o(2, 1);
    R(0, 0) = (float)cos(r2.angle * CV_PI / 180); R(0, 1) = (float)(-sin(r2.angle * CV_PI / 180));
    R(1, 0) = (float)sin(r2.angle * CV_PI / 180); R(1, 1) = (float)cos(r2.angle * CV_PI / 180);
    Si(0, 0) = (float)(samples.cols / r2.size.width); Si(0, 1) = 0.0f;
    Si(1, 0) = 0.0f; Si(1, 1) = (float)(samples.rows / r2.size.height);
    s(0, 0) = (float)samples.cols; s(1, 0) = (float)samples.rows;
    o(0, 0) = r2.center.x; o(1, 0) = r2.center.y;
    Mat_<float> A(2, 2), b(2, 1);
    A = Si * R;
    b = s / 2.0 - Si * R * o;
    A.copyTo(M.colRange(Range(0, 2)));
    b.copyTo(M.colRange(Range(2, 3)));
    warpAffine(img, samples, M, samples.size());
}


void Detection::getPattern(const Mat& img, Mat& pattern,Scalar& mean,Scalar& stdev){
    //Output: resized Zero-Mean patch
    resize(img,pattern,Size(patch_size,patch_size));
    meanStdDev(pattern,mean,stdev);
    pattern.convertTo(pattern,CV_32F);
    pattern = pattern-mean.val[0];
}

/* Inputs:
* - Image
* - bad_boxes (Boxes far from the bounding box)
* - variance (pEx variance)
* Outputs
* - Negative fern features (nX)
* - Negative NN examples (nEx)
*/
void Detection::generateNegativeData(){
    // cv::Mat img;
    cv::Mat frame;
#ifdef using_pyramid
    int index = scaleindex.at(1.0);    // orignal image
    frame = timages[index];
    // img = filterImages[index];
#else
    frame = timages;
    //  img = filterImages;
#endif
    random_shuffle(bad_boxes.begin(),bad_boxes.end());//Random shuffle bad_boxes indexes
    int idx;
    //Get Fern Features of the boxes with big variance (calculated using integral images)
    int a=0;
    //int num = std::min((int)bad_boxes.size(),(int)bad_patches*100); //limits the size of bad_boxes to try
    printf("negative data generation started.\n");

#if defined(FernFeature)
    vector<int> fern(classifier.getNumStructs());
    nX.reserve(bad_boxes.size());
#elif defined(CompressiveFeature)
    vector<float> ct(classifier.compDetector.featureNum);
    nX.reserve(bad_boxes.size());
#elif defined(HaarLikeFeature)
    nX.reserve(bad_boxes.size());
    classifier.haarSamples.clear();
    classifier.haarSamples.reserve(bad_boxes.size());
#ifndef using_pyramid
    std::vector<int> scalesCount;
    scalesCount.reserve(bad_boxes.size());
#endif
#elif defined(HOGFeature)
    nX.reserve(bad_boxes.size());
#endif

    Mat patch;
    for (int j=0;j<bad_boxes.size();j++){  // 提取金字塔上的视觉特征

        idx = bad_boxes[j];

#ifdef using_pyramid //pyramids
        int index = scaleindex.at(1.0);
        cv::Mat iisumTmp   =  iisum[index];
        cv::Mat iisqsumTmp =  iisqsum[index];

        if (!filterVar(grid[idx],iisumTmp ,iisqsumTmp,0.5)/*<var*0.5f*/) //grid <--> gridpyramids consistent
            continue;

#if defined(FernFeature)
        patch =  timages[gridpyramids[idx].sidx](gridpyramids[idx]); // orginal image
        classifier.getFeatures(patch,fern);
        nX.push_back(make_pair(fern,0));
#elif defined(CompressiveFeature)
        patch =  iisum[gridpyramids[idx].sidx](gridpyramids[idx]);
        classifier.compDetector.getFeatureCompressive(patch,ct);
        nX.push_back(ct);
#elif defined(HaarLikeFeature)
        classifier.haarSamples.push_back(iisum[gridpyramids[idx].sidx](gridpyramids[idx]));
#elif defined(HOGFeature)
        nX.resize(nX.size()+1);
        classifier.fhogfeature.getFeatureHog(timages[gridpyramids[idx].sidx](gridpyramids[idx]),nX[nX.size()-1]);
#endif
        a++;

#else  // not pyramids

        if (!filterVar(grid[idx],iisum,iisqsum,0.5))
            continue;
#if defined(FernFeature)
        patch =  frame(grid[idx]);
        classifier.getFeaturesScale(patch,grid[idx].sidx,fern);
        nX.push_back(make_pair(fern,0));
#elif defined(CompressiveFeature)
        classifier.compDetector.getFeatureCompressiveScale(iisum(grid[idx]),grid[idx].sidx,ct);
        nX.push_back(ct);
#elif defined(HaarLikeFeature)
        scalesCount.push_back(grid[idx].sidx);
        classifier.haarSamples.push_back(iisum(grid[idx]));
#elif defined(HOGFeature)
        nX.resize(nX.size()+1);
        classifier.fhogfeature.getFeatureHog(frame(grid[idx]),nX[nX.size()-1]);
#endif
        a++;
#endif
    }

#if defined(FernFeature)
#elif defined(CompressiveFeature)
#elif defined(HaarLikeFeature)
    nX.reserve( classifier.haarSamples.size());
    for(int i = 0 ; i < classifier.haarSamples.size() ; i++ )
        nX.push_back(cv::Mat_<float>(1,classifier.haarfeature.featureNum));
#ifdef using_pyramid
    classifier.haarfeature.getFeatureHaar(classifier.haarSamples,nX);
#else
    classifier.haarfeature.getFeatureHaarScale(classifier.haarSamples,nX,scalesCount);
#endif
#endif
    printf("Negative examples generated: ferns: %d ",a);

    //random_shuffle(bad_boxes.begin(),bad_boxes.begin()+bad_patches);//Randomly selects 'bad_patches' and get the patterns for NN;
    Scalar dum1, dum2;
    nEx=vector<Mat>(bad_patches); //100 = bad_patches；
    for (int i=0;i<bad_patches;i++){
        idx=bad_boxes[i];
        patch = frame(grid[idx]);  //grid <--> gridpyramids consistent
        getPattern(patch,nEx[i],dum1,dum2);
    }
    printf("NN: %d\n",(int)nEx.size());
}
bool bbcomp(const BoundingBox& box1,const BoundingBox& box2){
    double over_lap = -FLT_MAX;
    if (box1.x > box2.x+box2.width) { over_lap =  0.0; }
    if (box1.y > box2.y+box2.height){ over_lap =  0.0; }
    if (box1.x+box1.width < box2.x) { over_lap =  0.0; }
    if (box1.y+box1.height < box2.y){ over_lap =  0.0; }

    if(over_lap!=0.0)
    {
        float colInt =  min(box1.x+box1.width,box2.x+box2.width) - max(box1.x, box2.x);
        float rowInt =  min(box1.y+box1.height,box2.y+box2.height) - max(box1.y,box2.y);

        float intersection = colInt * rowInt;
        float area1 = box1.width*box1.height;
        float area2 = box2.width*box2.height;
        over_lap = intersection / (area1 + area2 - intersection);
    }
    if (over_lap<0.5)
        return false;
    else
        return true;
    /*Detection t;
    if (t.bbOverlap(b1,b2)<0.5)
      return false;
    else
      return true;*/
}
int Detection::clusterBB(const vector<BoundingBox>& dbb,vector<int>& indexes){
    //FIXME: Conditional jump or move depends on uninitialised value(s)
    const int c = dbb.size();
    //1. Build proximity matrix
    Mat D(c,c,CV_32F);
    float d;
    for (int i=0;i<c;i++){
        for (int j=i+1;j<c;j++){
            d = 1-bbOverlap(dbb[i],dbb[j]);
            D.at<float>(i,j) = d;
            D.at<float>(j,i) = d;
        }
    }
    //2. Initialize disjoint clustering
    float L[c-1]; //Level
    int nodes[c-1][2];
    int belongs[c];
    int m=c;
    for (int i=0;i<c;i++){
        belongs[i]=i;
    }
    for (int it=0;it<c-1;it++){
        //3. Find nearest neighbor
        float min_d = 1;
        int node_a, node_b;
        for (int i=0;i<D.rows;i++){
            for (int j=i+1;j<D.cols;j++){
                if (D.at<float>(i,j)<min_d && belongs[i]!=belongs[j]){
                    min_d = D.at<float>(i,j);
                    node_a = i;
                    node_b = j;
                }
            }
        }
        if (min_d>0.5){
            int max_idx =0;
            bool visited;
            for (int j=0;j<c;j++){
                visited = false;
                for(int i=0;i<2*c-1;i++){
                    if (belongs[j]==i){
                        indexes[j]=max_idx;
                        visited = true;
                    }
                }
                if (visited)
                    max_idx++;
            }
            return max_idx;
        }

        //4. Merge clusters and assign level
        L[m]=min_d;
        nodes[it][0] = belongs[node_a];
        nodes[it][1] = belongs[node_b];
        for (int k=0;k<c;k++){
            if (belongs[k]==belongs[node_a] || belongs[k]==belongs[node_b])
                belongs[k]=m;
        }
        m++;
    }
    return 1;

}

void Detection::clusterConf(const vector<BoundingBox>& dbb,const vector<float>& dconf,vector<BoundingBox>& cbb,vector<float>& cconf){
    int numbb =dbb.size();
    vector<int> T;
    float space_thr = 0.5;
    int c=1;
    switch (numbb){
    case 1:
        cbb=vector<BoundingBox>(1,dbb[0]);
        cconf=vector<float>(1,dconf[0]);
        return;
        break;
    case 2:
        T =vector<int>(2,0);
        if (1-bbOverlap(dbb[0],dbb[1])>space_thr){
            T[1]=1;
            c=2;
        }
        break;
    default:
        T = vector<int>(numbb,0);
        c = partition(dbb,T,(*bbcomp));
        //c = clusterBB(dbb,T);
        break;
    }
    cconf=vector<float>(c);
    cbb=vector<BoundingBox>(c);
    if(is_debug)
        printf("Cluster indexes: ");
    BoundingBox bx;
    for (int i=0;i<c;i++){
        float cnf=0;
        int N=0,mx=0,my=0,mw=0,mh=0;
        for (int j=0;j<T.size();j++){
            if (T[j]==i){
                if(is_debug)
                    printf("%d ",i);
                cnf=cnf+dconf[j];
                mx=mx+dbb[j].x;
                my=my+dbb[j].y;
                mw=mw+dbb[j].width;
                mh=mh+dbb[j].height;
                N++;
            }
        }
        if (N>0){
            cconf[i]=cnf/N;
            bx.x=cvRound(mx/N);
            bx.y=cvRound(my/N);
            bx.width=cvRound(mw/N);
            bx.height=cvRound(mh/N);
            cbb[i]=bx;
        }
    }
    printf("\n");
}



