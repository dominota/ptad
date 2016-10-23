/*
 * Classifier.cpp
 *
 *  Created on: Jun 14, 2011
 *      Author: alantrrs
 */

#include <Classifier.h>

using namespace cv;
using namespace std;

void Classifier::read(const FileNode& file){
    ///Classifier Parameters
    valid = (float)file["valid"];
    ncc_thesame = (float)file["ncc_thesame"];
    nstructs = (int)file["num_trees"]; //trees
    structSize = (int)file["num_features"]; //features
    thr_fern = (float)file["thr_fern"];
    thr_nn = (float)file["thr_nn"];
    thr_nn_valid = (float)file["thr_nn_valid"];
   // compDetector.initCT();
}

void Classifier::prepare(const Size& scales){
  acum = 0;
  //Initialize test locations for features
  int totalFeatures = nstructs*structSize;
  features.resize(1);
  features[0] = vector<Feature> (totalFeatures);
  RNG& rng = theRNG();
  //float x1f,x2f,y1f,y2f;
  int x1, x2, y1, y2;
  for (int i=0;i<totalFeatures;i++){
      x1 = (float)rng* scales.width;;
      y1 = (float)rng* scales.height;;
      x2 = (float)rng* scales.width;;
      y2 = (float)rng* scales.height;
      features[0][i] = Feature(x1, y1, x2, y2);
  }
  //Thresholds
  thrN = 0.5*nstructs;

  //Initialize Posteriors
  for (int i = 0; i<nstructs; i++) {
      posteriors.push_back(vector<float>(pow(2.0,structSize), 0));
      pCounter.push_back(vector<int>(pow(2.0,structSize), 0));
      nCounter.push_back(vector<int>(pow(2.0,structSize), 0));
  }
}
void Classifier::getFeatures(const cv::Mat& image, vector<int>& fern){
  if(features.size()==0)  return;
  int leaf;
  for (int t=0;t<nstructs;t++){
      leaf=0;
      for (int f=0; f<structSize; f++){
          leaf = (leaf << 1) + features[0][t*structSize+f](image);
      }
      fern[t]=leaf;
  }
}
void Classifier::prepareScale(const vector<Size>& scales){
  acum = 0;
  //Initialize test locations for features
  int totalFeatures = nstructs*structSize;
  features = vector<vector<Feature> >(scales.size(),vector<Feature> (totalFeatures));
  RNG& rng = theRNG();
  float x1f,x2f,y1f,y2f;
  int x1, x2, y1, y2;
  for (int i=0;i<totalFeatures;i++){
      x1f = (float)rng;
      y1f = (float)rng;
      x2f = (float)rng;
      y2f = (float)rng;
      for (int s=0;s<scales.size();s++){
          x1 = x1f * scales[s].width;
          y1 = y1f * scales[s].height;
          x2 = x2f * scales[s].width;
          y2 = y2f * scales[s].height;
          features[s][i] = Feature(x1, y1, x2, y2);
      }

  }
  //Thresholds
  thrN = 0.5*nstructs;

  //Initialize Posteriors
  for (int i = 0; i<nstructs; i++) {
      posteriors.push_back(vector<float>(pow(2.0,structSize), 0));
      pCounter.push_back(vector<int>(pow(2.0,structSize), 0));
      nCounter.push_back(vector<int>(pow(2.0,structSize), 0));
  }
}

void Classifier::getFeaturesScale(const cv::Mat& image,const int& scale_idx, vector<int>& fern){
  int leaf;
  for (int t=0;t<nstructs;t++){
      leaf=0;
      for (int f=0; f<structSize; f++){
          leaf = (leaf << 1) + features[scale_idx][t*structSize+f](image);
      }
      fern[t]=leaf;
  }
}

float Classifier::measure_forest(vector<int> fern) {
  float votes = 0;
  for (int i = 0; i < nstructs; i++) {
      votes += posteriors[i][fern[i]];
  }
  return votes;
}

void Classifier::update(const vector<int>& fern, int C, int N) {
  int idx;
  for (int i = 0; i < nstructs; i++) {
      idx = fern[i];
      (C==1) ? pCounter[i][idx] += N : nCounter[i][idx] += N;
      if (pCounter[i][idx]==0) {
          posteriors[i][idx] = 0;
      } else {
          posteriors[i][idx] = ((float)(pCounter[i][idx]))/(pCounter[i][idx] + nCounter[i][idx]);
      }
  }
}

void Classifier::trainF(const vector<std::pair<vector<int>,int> >& ferns,int resample){
  // Conf = function(2,X,Y,Margin,Bootstrap,Idx)
  //                 0 1 2 3      4         5
  //  double *X     = mxGetPr(prhs[1]); -> ferns[i].first
  //  int numX      = mxGetN(prhs[1]);  -> ferns.size()
  //  double *Y     = mxGetPr(prhs[2]); ->ferns[i].second
  //  double thrP   = *mxGetPr(prhs[3]) * nTREES; ->threshold*nstructs
  //  int bootstrap = (int) *mxGetPr(prhs[4]); ->resample
  thrP = thr_fern*nstructs;                                                          // int step = numX / 10;
  //for (int j = 0; j < resample; j++) {                      // for (int j = 0; j < bootstrap; j++) {
      for (int i = 0; i < ferns.size(); i++){               //   for (int i = 0; i < step; i++) {
                                                            //     for (int k = 0; k < 10; k++) {
                                                            //       int I = k*step + i;//box index                                                            //       double *x = X+nTREES*I; //tree index
          if(ferns[i].second==1){                           //       if (Y[I] == 1) {
            if(measure_forest(ferns[i].first)<=thrP)      //         if (measure_forest(x) <= thrP)
                update(ferns[i].first,1,1);                 //             update(x,1,1);
          }else{                                            //        }else{
              if (measure_forest(ferns[i].first) >= thrN)   //         if (measure_forest(x) >= thrN)
                update(ferns[i].first,0,1);                 //             update(x,0,1);
          }
      }
  //}
}

void Classifier::trainNN(const vector<cv::Mat>& nn_examples,int nums_possamples){
    float conf,dummy;
    vector<int> y(nn_examples.size(),0);
    for(int i = 0; i < nums_possamples;i++)
        y[i]=1;
    vector<int> isin;
    int nums_halfsamples = nn_examples.size()/2;
    int res = 0;
    float maxPconf = 0;
    if(nn_examples.size() % 2 != 0 ) res = nums_halfsamples *2; //奇数
    for (int i=0;i< nums_halfsamples;i++){                          //  For each example
        maxPconf = NNConf(nn_examples[i],isin,conf,dummy);                      //  Measure Relative similarity
        if (y[i]==1 && conf<=thr_nn+0.05  && conf >0.5){                                //    if y(i) == 1 && conf1 <= tld.model.thr_nn % 0.65 + 0.1
            if (isin[1]<0){                                          //      if isnan(isin(2))
                pEx = vector<Mat>(1,nn_examples[i]);                 //        tld.pex = x(:,i);
                continue;                                            //        continue;
            }                                                        //      end
            //pEx.insert(pEx.begin()+isin[1],nn_examples[i]);        //      tld.pex = [tld.pex(:,1:isin(2)) x(:,i) tld.pex(:,isin(2)+1:end)]; % add to model
            pEx.push_back(nn_examples[i]);
        }

        int j = 2*nums_halfsamples -i-1; //从最后到
        maxPconf = NNConf(nn_examples[j],isin,conf,dummy);
        if (y[j]==1 && conf<=thr_nn+0.05  && conf >0.5){                                //    if y(i) == 1 && conf1 <= tld.model.thr_nn % 0.65 + 0.1
            if (isin[1]<0){                                          //      if isnan(isin(2))
                pEx = vector<Mat>(1,nn_examples[j]);                 //        tld.pex = x(:,i);
                continue;                                            //        continue;
            }                                                        //      end
            //pEx.insert(pEx.begin()+isin[1],nn_examples[i]);        //      tld.pex = [tld.pex(:,1:isin(2)) x(:,i) tld.pex(:,isin(2)+1:end)]; % add to model
            pEx.push_back(nn_examples[j]);
        }
        if(y[j]==0 && conf>0.5)                                      //  if y(i) == 0 && conf1 > 0.5
            nEx.push_back(nn_examples[j]);                             //    tld.nex = [tld.nex x(:,i)];
    }

    if(res != 0)
    {
         maxPconf = NNConf(nn_examples[res],isin,conf,dummy);
        if (y[res]==1 && conf<=thr_nn+0.05  && conf >0.5){                                //    if y(i) == 1 && conf1 <= tld.model.thr_nn % 0.65 + 0.1
            pEx.push_back(nn_examples[res]);
        }
        if(y[res]==0 && conf>0.5)                                      //  if y(i) == 0 && conf1 > 0.5
            nEx.push_back(nn_examples[res]);                             //    tld.nex = [tld.nex x(:,i)];
    }
    acum++;

    static RNG rngforSamples;
    while(pEx.size() > 800)
    {
        int eraseElment =  rngforSamples.uniform(0,pEx.size()-1);
        std::vector<cv::Mat>::iterator it = pEx.begin()+eraseElment;
        pEx.erase(it);
    }
    printf("model updateing: %d. Trained NN examples: %d positive %d negative\n",acum,(int)pEx.size(),(int)nEx.size());
}                                                                  //  end


float Classifier::NNConf(const Mat& example, vector<int>& isin,float& rsconf,float& csconf){
  /*Inputs:
   * -NN Patch
   * Outputs:
   * -Relative Similarity (rsconf), Conservative Similarity (csconf), In pos. set|Id pos set|In neg. set (isin)
   */
  isin=vector<int>(3,-1);
  if (pEx.empty()){ //if isempty(tld.pex) % IF positive examples in the model are not defined THEN everything is negative
      rsconf = 0.55; //    conf1 = zeros(1,size(x,2));
      csconf=0.55;
      return 1;
  }
  if (nEx.empty()){ //if isempty(tld.nex) % IF negative examples in the model are not defined THEN everything is positive
      rsconf = 1;   //    conf1 = ones(1,size(x,2));
      csconf=1;
      return 0;
  }
  Mat ncc(1,1,CV_32F);
  float nccP,csmaxP,maxP=0;
  bool anyP=false;
  int maxPidx,validatedPart = ceil(pEx.size()*valid);
  float nccN, maxN=0;

  float averP = 0;
  bool anyN=false;
  for (int i=0;i<pEx.size();i++){
      matchTemplate(pEx[i],example,ncc,CV_TM_CCORR_NORMED);      // measure NCC to positive examples
      nccP=(((float*)ncc.data)[0]+1)*0.5; //nccP >0.5
      if (nccP>ncc_thesame)// 0.9 similarity samples
        anyP=true;
      if(nccP > maxP){
          maxP=nccP;
          maxPidx = i;
          if(i<validatedPart)
            csmaxP=maxP;
      }
      averP = averP + ((float*)ncc.data)[0];
  }
  for (int i=0;i<nEx.size();i++){
      matchTemplate(nEx[i],example,ncc,CV_TM_CCORR_NORMED);     //measure NCC to negative examples
      nccN=(((float*)ncc.data)[0]+1)*0.5; //nccN >0.5
      if (nccN>ncc_thesame)
        anyN=true;
      if(nccN > maxN)
        maxN=nccN;
  }
  //set isin
  if (anyP) isin[0]=1;  //if he query patch is highly correlated with any positive patch in the model then it is considered to be one of them
  isin[1]=maxPidx;      //get the index of the maximall correlated positive patch
  if (anyN) isin[2]=1;  //if  the query patch is highly correlated with any negative patch in the model then it is considered to be one of them
  //Measure Relative Similarity
  float dN=1-maxN;
  float dP=1-maxP;
  rsconf = (float)dN/(dN+dP);
  //Measure Conservative Similarity
  dP = 1 - csmaxP;
  csconf =(float)dN / (dN + dP);
  averP  = averP / pEx.size();
  return 1;
}

void Classifier::evaluateNCCTh(const vector<cv::Mat>& nExT){
    vector <int> isin;
    float conf,dummy;
    for (int i=0;i<nExT.size();i++){
        NNConf(nExT[i],isin,conf,dummy);
        if (conf>thr_nn)
            thr_nn=conf;
    }
    if (thr_nn>thr_nn_valid)
        thr_nn_valid = thr_nn;
}

void Classifier::evaluateFernTh(const vector<pair<vector<int>,int> >& nXT){
    float fconf;
    for (int i=0;i<nXT.size();i++){
        fconf = (float) measure_forest(nXT[i].first)/nstructs;
        if (fconf>thr_fern)
            thr_fern=fconf;
    }
}


void Classifier::show(){
  int widthAndHeight = std::ceil(sqrt(pEx.size()));
  Mat examples =cv::Mat::zeros((int)widthAndHeight*pEx[0].rows,(int)widthAndHeight*pEx[0].cols,CV_8U);
  double minval;
  Mat ex(pEx[0].rows,pEx[0].cols,pEx[0].type());
  int rowindex=0;
  int colindex=0;
  for (int i=0;i<pEx.size();i++){
    minMaxLoc(pEx[i],&minval);
    pEx[i].copyTo(ex);
    ex = ex-minval;
    cv::Rect roi(colindex*pEx[i].cols,rowindex*pEx[i].rows,pEx[0].cols,pEx[0].rows);
    Mat tmp = examples(roi); //examples.rowRange(Range(i*pEx[i].rows,(i+1)*pEx[i].rows));
    ex.convertTo(tmp,CV_8U);
    rowindex++;
    if(rowindex == widthAndHeight)
    {
        rowindex = 0;
        colindex++;
    }
  }

  int NwidthAndHeight = std::ceil(sqrt(nEx.size()));
  Mat Nexamples =cv::Mat::zeros((int)NwidthAndHeight*nEx[0].rows,(int)NwidthAndHeight*nEx[0].cols,CV_8U);
  double Nminval;
  Mat Nex(nEx[0].rows,nEx[0].cols,nEx[0].type());
  int Nrowindex=0;
  int Ncolindex=0;
  for (int i=0;i<nEx.size();i++){
    minMaxLoc(nEx[i],&Nminval);
    nEx[i].copyTo(Nex);
    Nex = Nex-Nminval;
    cv::Rect Nroi(Ncolindex*nEx[i].cols,Nrowindex*nEx[i].rows,nEx[0].cols,nEx[0].rows);
    Mat tmp = Nexamples(Nroi); //examples.rowRange(Range(i*pEx[i].rows,(i+1)*pEx[i].rows));
    Nex.convertTo(tmp,CV_8U);
    Nrowindex++;
    if(Nrowindex == NwidthAndHeight)
    {
        Nrowindex = 0;
        Ncolindex++;
    }
  }
 // examples.convertTo(examples,CV_8U);
  imshow("PosExamples",examples);
  imshow("NegExamples",Nexamples);
}

void Classifier::clear()
{
    features.clear();
    nCounter.clear();
    pCounter.clear();; //positive counter
    posteriors.clear();; //Ferns posteriors
    pEx.clear();; //NN positive examples
    nEx.clear();; //NN negative examples
    haarSamples.clear();
}


