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
bool isVideo = false;
Rect box;
bool drawing_box = false;
bool gotBB = false;
bool tl = true;
bool rep = false;
bool fromfile=false;
string video;
string inputpath;
string outputpath;
string initpath;
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

void read_options(int argc, char** argv,FileStorage &fs){
    for (int i=0;i<argc;i++){
        if (strcmp(argv[i],"-b")==0){
            if (argc>i){
                readBB(argv[i+1]);
                gotBB = true;
            }
            else
                print_help(argv);
        }
        if (strcmp(argv[i],"-vs")==0){
            if (argc>i){
                video = string(argv[i+1]);
                fromfile = true;
                isVideo = true;
            }
            else
                print_help(argv);

        }
        if (strcmp(argv[i],"-vc")==0){
            if (argc>i){
                video = string(argv[i+1]);
                isVideo = true;
                fromfile =false;
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
        if (strcmp(argv[i],"-i")==0){
            if (argc>i){
                inputpath = string(argv[i+1]);
                fromfile = true;
            }
            else
                print_help(argv);
        }
        if (strcmp(argv[i],"-n")==0){
            if (argc>i){
                initpath = string(argv[i+1]);
            }
            else
                print_help(argv);
        }
        if (strcmp(argv[i],"-o")==0){
            if (argc>i){
                outputpath = string(argv[i+1]);
            }
            else
                print_help(argv);
        }
        if (strcmp(argv[i],"-r")==0){
            rep = true;
        }
    }
    if(outputpath.empty()) outputpath = inputpath;
    if(initpath.empty())   initpath   = inputpath;
}

int main(int argc, char * argv[]){
    XInitThreads();
    FileStorage fs;
    read_options(argc,argv,fs);
    VideoCapture capture;
    VOT vot_io(initpath,inputpath,outputpath,string(".jpg"));
    if(!isVideo)
    {
        if(inputpath.empty())
        {
            cout << "images failed to load!" << endl;
            return 1;
        }
    }
    else
    {   if(fromfile && !video.empty())
            capture.open(video);
        else
            capture.open(0);
        if (!capture.isOpened())
        {
            cout << "capture device failed to open!" << endl;
            return 1;
        }
    }
    //Register mouse callback to draw the bounding box
    cvNamedWindow("PTAD",CV_WINDOW_AUTOSIZE);
    cvSetMouseCallback( "PTAD", mouseHandler, NULL );
    System tsystem;
    //Read parameters file
    tsystem.detection_.read(fs.getFirstTopLevelNode());
    cv::Mat oframe, frame,last_gray, first;
    float scaleFactor = 1;
    int width,height;
    if(!isVideo)
    {
        vot_io.getNextImage(oframe); //color_image
        float wscale = 480 / float(oframe.cols);
        float hscale = 320 / float(oframe.rows);
        scaleFactor = std::max(wscale,hscale);
        if(scaleFactor>1)
            scaleFactor =1;
        width =int(oframe.cols *scaleFactor);
        height =int(oframe.rows *scaleFactor);

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
    }
    else
    {
        if (fromfile){
            capture >> frame;
            cvtColor(frame, last_gray, CV_RGB2GRAY);
            frame.copyTo(first);
        }else{
            capture.set(CV_CAP_PROP_FRAME_WIDTH,640);
            capture.set(CV_CAP_PROP_FRAME_HEIGHT,480);
        }
    }

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
    tsystem.initImpl(frame, box); //rgb--image;
    ///Run-time
    Rect2d pbox = box;
    int frames = 1;
    double tStart = getTickCount();
    double  tDuration ;
    bool tracked = true;
REPEAT:
    if(!isVideo)
    {
        while(vot_io.getNextImage(oframe)==1){
            if(scaleFactor <1 )
                cv::resize(oframe,frame,Size(width,height));
            else
                frame = oframe.clone();
            tsystem.Run(frame,pbox,tracked);
            if(tracked)  drawBox(frame,pbox,Scalar(255, 0, 255),2);
            Rect2d trackingBox(pbox.x/scaleFactor, pbox.y/scaleFactor, pbox.width/scaleFactor, pbox.height/scaleFactor);
            vot_io.outputBoundingBox(trackingBox,tracked);
            tDuration = getTickCount() - tStart;
            double fps = static_cast<double>(getTickFrequency() / tDuration) * (frames-1);
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
            if (cv::waitKey(27) == 'q') //CT 15_17 20 LK  kcf 27
                break;
        }
    }
    else
    {
        while(capture.read(frame)){
            tsystem.Run(frame,pbox,tracked);
            if(tracked)  drawBox(frame,pbox,Scalar(255, 0, 255),2);
            tDuration = getTickCount() - tStart;
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
            if (cv::waitKey(27) == 'q')
                break;
        }
    }
}
