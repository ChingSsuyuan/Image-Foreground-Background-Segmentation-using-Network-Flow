#ifndef FEATUREEXTRACTOR_H
#define FEATUREEXTRACTOR_H

#include <opencv2/opencv.hpp>
#include <vector>

struct PixelFeature {
    cv::Vec3b colorRGB;      // 
    cv::Point position;      // 
    float gradientMagnitude; //
};

std::vector<std::vector<PixelFeature>> extractFeatures(const cv::Mat& image);

#endif
