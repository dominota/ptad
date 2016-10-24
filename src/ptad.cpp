#include <opencv2/opencv.hpp>
#include "opencv2/datasets/track_vot.hpp"
#include <utils.h>
#include <iostream>
#include <sstream>
#include <system.h>
#include <stdio.h>
#include "vot.hpp"
using namespace cv;
using namespace std;
#include <X11/Xlib.h>
//Global variables
#define VOT_DATA
Rect box;
bool drawing_box = false;
bool gotBB = false;
bool tl = true;
bool rep = false;
bool fromfile=false;
string video;

void readBB(char* file){
    ifstream bb_file (file);
    string line;
    getline(bb_file,line);
    istringstream linestream(line);
    string x1,y1,x2,y2;
    getline (linestream,x1, ',');
    getline (linestream,y1, ',');
    getline (linestream,x2, ',');
    getline (linestream,y2, ',');
    int x = atoi(x1.c_str());// = (int)file["bb_x"];
    int y = atoi(y1.c_str());// = (int)file["bb_y"];
    int w = atoi(x2.c_str())-x;// = (int)file["bb_w"];
    int h = atoi(y2.c_str())-y;// = (int)file["bb_h"];
    box = Rect(x,y,w,h);
}
//bounding box mouse callback
void mouseHandler(int event, int x, int y, int flags, void *param){
    switch( event ){
    case CV_EVENT_MOUSEMOVE:
        if (drawing_box){
            box.width = x-box.x;
            box.height = y-box.y;
        }
        break;
    case CV_EVENT_LBUTTONDOWN:
        drawing_box = true;
        box = Rect( x, y, 0, 0 );
        break;
    case CV_EVENT_LBUTTONUP:
        drawing_box = false;
        if( box.width < 0 ){
            box.x += box.width;
            box.width *= -1;
        }
        if( box.height < 0 ){
            box.y += box.height;
            box.height *= -1;
        }
        gotBB = true;
        break;
    }
}

void print_help(char** argv){
    printf("use:\n     %s -p /path/parameters.yml\n",argv[0]);
    printf("-s    source video\n-b        bounding box file\n-tl  track and learn\n-r     repeat\n");
}

void read_options(int argc, char** argv,VideoCapture& capture,FileStorage &fs){
    for (int i=0;i<argc;i++){
        if (strcmp(argv[i],"-b")==0){
            if (argc>i){
                readBB(argv[i+1]);
                gotBB = true;
            }
            else
                print_help(argv);
        }
        if (strcmp(argv[i],"-s")==0){
            if (argc>i){
                video = string(argv[i+1]);
                capture.open(video);
                fromfile = true;
            }
            else
                print_help(argv);

        }
        if (strcmp(argv[i],"-p")==0){
            if (argc>i){
                fs.open(argv[i+1], FileStorage::READ);
            }
            else
                print_help(argv);
        }
        if (strcmp(argv[i],"-no_tl")==0){
            tl = false;
        }
        if (strcmp(argv[i],"-r")==0){
            rep = true;
        }
    }
}

int main(int argc, char * argv[]){

    XInitThreads();

    std::vector<std::string> paths = FileFunctions::Dir("/home/nubot/PTAD/TLD",true);
    std::string selectedTrackingAlgorithm;
#if defined(FernFeature)
    selectedTrackingAlgorithm = "Fern";
#elif defined(CompressiveFeature)
    selectedTrackingAlgorithm = "CT";
#elif defined(HaarLikeFeature)
    selectedTrackingAlgorithm = "Haar";
#elif defined(HOGFeature)
    selectedTrackingAlgorithm = "HOG";
#endif

#if defined(TRACKER_LKAlgorithm)
    selectedTrackingAlgorithm = selectedTrackingAlgorithm+"_LK";
#elif defined(KCFTrackerAlgorithm)
    selectedTrackingAlgorithm = selectedTrackingAlgorithm+"_KCF";
#elif defined(BoostingTrackerAlgorithm)
    selectedTrackingAlgorithm = selectedTrackingAlgorithm+"_Boosting";
#elif defined(MILTrackerAlgorithm)
    selectedTrackingAlgorithm = selectedTrackingAlgorithm+"_MIL";
#elif defined(CompressiveTrackerAlgorithm)
    selectedTrackingAlgorithm = selectedTrackingAlgorithm+"_CT";
#endif
  //  selectedTrackingAlgorithm = selectedTrackingAlgorithm+"5fps";
    std::vector<std::string> rpaths = FileFunctions::Dir(paths[paths.size()-1].c_str(),true);
    paths.erase(paths.end());
    int i = 8;
    // for(int i = 8; i < paths.size(); i++)
    {
#ifdef VOT_DATA
        std::string path =paths[i];
        std::string output = rpaths[i] + "/" + selectedTrackingAlgorithm.c_str();
        std::cout<<path<<std::endl;
        std::string image;
        std::string right =".jpg";
        if(i == 6) right = ".png";
        VOT vot_io(path,image,output,right);
        VideoCapture capture;
        FileStorage fs;
        read_options(argc,argv,capture,fs);
#else

        VideoCapture capture;
        capture.open(0);
        FileStorage fs;
        //Read options
        read_options(argc,argv,capture,fs);
        if (!capture.isOpened())
        {
            cout << "capture device failed to open!" << endl;
            return 1;
        }
#endif
        //Register mouse callback to draw the bounding box
        cvNamedWindow("PTAD",CV_WINDOW_AUTOSIZE);
        cvSetMouseCallback( "PTAD", mouseHandler, NULL );
        //TLD framework
        static System tsystem;
        tsystem.detection_.clear();
        //Read parameters file
        tsystem.detection_.read(fs.getFirstTopLevelNode());

        cv::Mat oframe;
        Mat frame;
        Mat last_gray;
        Mat first;
#ifdef VOT_DATA
        vot_io.getNextImage(oframe); //color_image
        float wscale = 480 / float(oframe.cols);
        float hscale = 320 / float(oframe.rows);
        float scaleFactor = std::max(wscale,hscale);
        if(scaleFactor>1)
            scaleFactor =1;
        int width  =int(oframe.cols *scaleFactor);
        int height =int(oframe.rows *scaleFactor);
        if(scaleFactor <1 )
            cv::resize(oframe,frame,Size(width,height));
        else
            frame = oframe;
        if(frame.channels() > 1 )
            cvtColor(frame, last_gray, CV_RGB2GRAY);
        else
            last_gray = frame;
        frame.copyTo(first);
        box = vot_io.getInitRectangle();
        vot_io.outputBoundingBox(cv::Rect2d(box),true);
        box.x = box.x * scaleFactor;
        box.y = box.y * scaleFactor;
        box.width = box.width * scaleFactor;
        box.height = box.height * scaleFactor;
        gotBB = true;

#else
        if (fromfile){
            capture >> frame;
            cvtColor(frame, last_gray, CV_RGB2GRAY);
            frame.copyTo(first);
        }else{
            capture.set(CV_CAP_PROP_FRAME_WIDTH,640);
            capture.set(CV_CAP_PROP_FRAME_HEIGHT,480);
        }
#endif
        ///Initialization
GETBOUNDINGBOX:
        while(!gotBB)
        {
            if (!fromfile){
                capture >> frame;
            }
            else
                first.copyTo(frame);
            //  cvtColor(frame, last_gray, CV_RGB2GRAY);
            cv::Mat img = frame.clone();
            drawBox(img,box);
            imshow("PTAD", img);
            if (cvWaitKey(33) == 'q')
                return 0;
        }
        if (min(box.width,box.height)<(int)fs.getFirstTopLevelNode()["min_win"]){
            cout << "Bounding box too small, try again." << endl;
            gotBB = false;
            goto GETBOUNDINGBOX;
        }
        //Remove callback
        cvSetMouseCallback( "PTAD", NULL, NULL );
        printf("Initial Bounding Box = x:%d y:%d h:%d w:%d\n",box.x,box.y,box.width,box.height);
        //Output file
        //FILE  *bb_file = fopen("bounding_boxes.txt","w");
        //TLD initialization

        tsystem.initImpl(frame, box); //rgb--image;
        //.init(last_gray,box,bb_file);
        ///Run-time
        Rect2d pbox = box;
        int frames = 1;
        double tStart = getTickCount();
        double  tDuration ;
        bool tracked = true;
REPEAT:
#ifdef VOT_DATA
        while(vot_io.getNextImage(oframe)==1){
            if(scaleFactor <1 )
                cv::resize(oframe,frame,Size(width,height));
            else
                frame = oframe.clone();
            tsystem.Run(frame,pbox,tracked);
            drawBox(frame,pbox,Scalar(255, 0, 255),2);
            Rect2d trackingBox(pbox.x/scaleFactor, pbox.y/scaleFactor, pbox.width/scaleFactor, pbox.height/scaleFactor);
            vot_io.outputBoundingBox(trackingBox,tracked);
            tDuration = getTickCount() - tStart;
            // tStart = getTickCount();
            double fps = static_cast<double>(getTickFrequency() / tDuration) * frames ;
            std::stringstream ss;
            ss << "FPS: " << fps;
            putText(frame, ss.str(), Point(20, 20), FONT_HERSHEY_TRIPLEX, 0.5, Scalar(255, 0, 0));
            ss.str("");
            ss.clear();
            ss << "#" << frames;
            putText(frame, ss.str(), Point(frame.cols - 60, 20), FONT_HERSHEY_TRIPLEX, 0.5, Scalar(255, 0, 0));
            ss.str("");
            ss.clear();
            ss << "tracked: " << tracked;
            putText(frame, ss.str(), Point(20, frame.rows-20), FONT_HERSHEY_TRIPLEX, 0.5, Scalar(255, 0, 0));
            frames++;
            imshow("PTAD", frame);
           /* ss.str("");
            ss.clear();
            ss<<"/home/nubot/result/tracking"<<frames<<".jpg";
            cv::imwrite(ss.str(),frame);*/
            if (cv::waitKey(27) == 'q') //CT 15_17  LK  kcf 27
                break;
        }
        // tsystem.detect_running = false;
        // tsystem.~System();
        std::cout<<"A sequence ending"<<std::endl;
#else
        while(capture.read(frame)){
            tsystem.Run(frame,pbox,tracked);
            drawBox(frame,pbox);
            tDuration = getTickCount() - tStart;
            // tStart = getTickCount();
            double fps = static_cast<double>(getTickFrequency() / tDuration) * frames ;
            std::stringstream ss;
            ss << "FPS: " << fps;
            putText(frame, ss.str(), Point(20, 20), FONT_HERSHEY_TRIPLEX, 0.5, Scalar(255, 0, 0));
            ss.str("");
            ss.clear();
            ss << "#" << frames;
            putText(frame, ss.str(), Point(frame.cols - 60, 20), FONT_HERSHEY_TRIPLEX, 0.5, Scalar(255, 0, 0));
            frames++;
            imshow("PTAD", frame);
            if (cv::waitKey(1) == 'q')
                break;
        }
#endif
        /* if (rep){
            rep = false;
            tl = false;
            fclose(bb_file);
            bb_file = fopen("final_detector.txt","w");
            //capture.set(CV_CAP_PROP_POS_AVI_RATIO,0);
            capture.release();
            capture.open(video);
            goto REPEAT;
        }
        fclose(bb_file);*/
    }
    return 0;
}
