#ifndef FEATURE_EXTRACTOR_H
#define FEATURE_EXTRACTOR_H

#include <opencv2/opencv.hpp>
#include <vector>

struct PixelFeature {
    cv::Vec3b colorRGB;    // RGB colour values
    cv::Point position;    // image element position
    float gradient;        // gradient range
    float intensity;       // greyscale value
};

std::vector<std::vector<PixelFeature>> extractFeatures(const cv::Mat& image);

#endif // FEATURE_EXTRACTOR_H
