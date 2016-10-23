/* 
 *  Author : Tomas Vojir
 *  Date   : 2013-06-05
 *  Desc   : Simple class for parsing VOT inputs and providing 
 *           interface for image loading and storing output.
 */ 

#ifndef CPP_VOT_H
#define CPP_VOT_H

#include <string>
#include <fstream>
#include <iostream>
#include "opencv2/opencv.hpp"
#include "FileFunctions.h"

class VOT
{
    const double NaNdata =std::numeric_limits<double>::signaling_NaN();
public:
    VOT(  std::string & region_file1, std::string & images,std::string & ouput,
          std::string & right)
    {
        std::string region_file = region_file1 +"/init.txt";
        p_region_stream.open(region_file.c_str());
        if (p_region_stream.is_open()){
            float x, y, w, h;
            char ch;
            p_region_stream >> x >> ch >> y >> ch >> w >> ch >> h;
            p_init_rectangle = cv::Rect(x, y, w-x, h-y);
        }else{
            std::cerr << "Error loading initial region in file " << region_file << "!" << std::endl;
            p_init_rectangle = cv::Rect(0, 0, 0, 0);
        }
      /*  p_region_stream.open(region_file.c_str());
        if (p_region_stream.is_open()){
            std::vector<float> x(4,0);
            std::vector<float> y(4,0);
            char ch;
            p_region_stream >> x[0] >> ch >> y[0] >> ch;
            p_region_stream >> x[1] >> ch >> y[1] >> ch;
            p_region_stream >> x[2] >> ch >> y[2] >> ch;
            p_region_stream >> x[3] >> ch >> y[3] >> ch;
            std::sort(x.begin(),x.end());
            std::sort(y.begin(),y.end());
            p_init_rectangle = cv::Rect(x[0], y[0], x[3]-x[0], y[3]-y[0]);
        }else{
            std::cerr << "Error loading initial region in file " << region_file << "!" << std::endl;
            p_init_rectangle = cv::Rect(0, 0, 0, 0);
        }*/
        std::cout<<region_file<<std::endl;
        images = region_file1 +"/";
        imagefiles = FileFunctions::Dir(images.c_str(), right.c_str(), true);
        /*p_images_stream.open(images.c_str());
        if (!p_images_stream.is_open())
            std::cerr << "Error loading image file " << images << "!" << std::endl;*/
        std::cout<<images<<std::endl;

        std::string path1 = ouput +"Rect.txt";
        p_output_stream.open(path1.c_str());
        if (!p_output_stream.is_open())
            std::cerr << "Error opening output file " << path1 << "!" << std::endl;
        std::cout<<path1<<std::endl;
        std::string path2 = ouput +".txt";
        p_output_stream2.open(path2.c_str());
        if (!p_output_stream2.is_open())
            std::cerr << "Error opening output1 file " << path2 << "!" << std::endl;
        std::cout<<path2<<std::endl;
        label = 0;
    }

    /*VOT(const std::string & region_file, const std::string & images  , const std::string & ouput )

        p_region_stream.open(region_file.c_str());
        if (p_region_stream.is_open()){
            float x, y, w, h;
            char ch;
            p_region_stream >> x >> ch >> y >> ch >> w >> ch >> h;
            p_init_rectangle = cv::Rect(x, y, w, h);
        }else{
            std::cerr << "Error loading initial region in file " << region_file << "!" << std::endl;
            p_init_rectangle = cv::Rect(0, 0, 0, 0);
        }

        p_images_stream.open(images.c_str());
        if (!p_images_stream.is_open())
            std::cerr << "Error loading image file " << images << "!" << std::endl;

        p_output_stream.open(ouput.c_str());
        if (!p_output_stream.is_open())
            std::cerr << "Error opening output file " << ouput << "!" << std::endl;
    }*/

    ~VOT()
    {
        p_region_stream.close();
        p_images_stream.close();
        p_output_stream.close();
        p_output_stream2.close();
    }

    inline cv::Rect getInitRectangle() const 
    {   return p_init_rectangle;    }

    inline void outputBoundingBox(const cv::Rect & bbox, const bool & tracked)
    {
        if(bbox.x>=0 && bbox.y >=0  && bbox.width>0 && bbox.height>0 && bbox.x + bbox.width < width && bbox.y + bbox.height <height)
        {
            if(tracked ) //tracked
                p_output_stream2<< bbox.x << ", " << bbox.y << ", " << bbox.x + bbox.width << ", " << bbox.y + bbox.height<<std::endl;
            else
                p_output_stream2<< NaNdata << ", " << NaNdata << ", " << NaNdata << ", " << NaNdata<<std::endl;
            p_output_stream<< bbox.x << ", " << bbox.y << ", " << bbox.x + bbox.width << ", " << bbox.y + bbox.height<<std::endl;
        }
        else
        {
            p_output_stream2<< NaNdata << ", " << NaNdata << ", " << NaNdata << ", " << NaNdata<<std::endl;
            p_output_stream<< NaNdata << ", " << NaNdata << ", " << NaNdata << ", " << NaNdata<<std::endl;
        }
    }

    inline void outputBoundingBox(const cv::Rect2d & bbox, const bool & tracked)
    {
        //   p_output_stream << bbox.x << ", " << bbox.y << ", " << bbox.width << ", " << bbox.height << std::endl;
        /*p_output_stream2<< bbox.x << ", " << bbox.y << ", " << bbox.x + bbox.width << ", " << bbox.y<<", "
                        << bbox.x + bbox.width << ", " << bbox.y + bbox.height<< ", " << bbox.x   << ", " << bbox.y + bbox.height
                          <<std::endl;*/
        if(bbox.x>=0 && bbox.y >=0  && bbox.width>0 && bbox.height>0 && bbox.x + bbox.width < width && bbox.y + bbox.height <height)
        {
            if(tracked ) //tracked
                p_output_stream2<< bbox.x << ", " << bbox.y << ", " << bbox.x + bbox.width << ", " << bbox.y + bbox.height<<std::endl;
            else
                p_output_stream2<< NaNdata << ", " << NaNdata << ", " << NaNdata << ", " << NaNdata<<std::endl;
            p_output_stream<< bbox.x << ", " << bbox.y << ", " << bbox.x + bbox.width << ", " << bbox.y + bbox.height<<std::endl;
        }
        else
        {
            p_output_stream2<< NaNdata << ", " << NaNdata << ", " << NaNdata << ", " << NaNdata<<std::endl;
            p_output_stream<< NaNdata << ", " << NaNdata << ", " << NaNdata << ", " << NaNdata<<std::endl;
        }
    }

    inline int getNextImage(cv::Mat & img)
    {
        if (label >= imagefiles.size() )
            return -1;

        img = cv::imread(imagefiles[label], CV_LOAD_IMAGE_COLOR);
        label++;
        //printf("Processing frame ID:  %d \n",label);
        if(label ==1)
        {
            width = img.cols;
            height = img.rows;
        }
        return 1;
    }

   /* inline int getNextImage(cv::Mat & img)
    {
        static int label = 0;
        if (p_images_stream.eof() || !p_images_stream.is_open())
            return -1;

        std::string line;
        std::getline (p_images_stream, line);
		img = cv::imread(line, CV_LOAD_IMAGE_COLOR);

		printf("Processing");
		printf(line.c_str());
		printf("\n");
		
		return 1;
    }*/

    cv::Rect p_init_rectangle;
    std::ifstream p_region_stream;
    std::ifstream p_images_stream;
    std::ofstream p_output_stream;
    std::ofstream p_output_stream2;
    std::vector<std::string> imagefiles;
    int label ;
    int width;
    int height;

};

#endif //CPP_VOT_H
