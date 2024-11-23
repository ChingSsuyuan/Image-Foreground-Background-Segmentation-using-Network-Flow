#ifndef SEGMENTATION_H
#define SEGMENTATION_H

#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;

// 定义图像分割函数
cv::Mat performSegmentation(const cv::Mat& inputImage);

#endif // SEGMENTATION_H