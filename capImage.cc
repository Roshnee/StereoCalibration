#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include<opencv2/core/core.hpp>
#include<iostream>
#include<fstream>

using namespace cv;
using namespace std;

int main(int argc,char **argv)
{
  VideoCapture cap;
  Mat image;
  const std::string videoStreamAddress="hhtp://192.168.0.100:9090/stream/video.mpeg";
  if(!cap.open(videoStreamAddress))
  {
    cout<<"Error opening the video stream"<<endl;
  }

  for(;;)
  {
    if(!cap.read(image)){
      cout<<"No frames"<<endl;
      waitKey();
    }
    cap.read(image);
    cv:: imshow(image);
    if(cv::waitKey(10)=='c')
    {
      imwrite("/home/roshnee/Thesis/rpicode/Stream/calib%02d",image);
    }
    else if(cv::waitkey(10)=='q')
    {
      break;
    }
  }
cap.release();
return 1;
}
