#include <stdio.h>
#include <Python.h>

#include <opencv2/opencv.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

int main( int argc, char** argv )
{
    if( argc != 2)
    {
        std::cout <<" Usage: display_image ImageToLoadAndDisplay" << std::endl;
        return -1;
    }

    cv::Mat image;
    image = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);

    if(!image.data)
    {
        std::cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    system("python3 ../predict-docker-serving.py");

    //cv::imwrite("test-for-classification.jpg", image);

    //cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );
    //cv::imshow( "Display window", image );
    //cv::waitKey(0);

    return 0;
}
