#ifndef FEATUREEXTRACTOR_H
#define FEATUREEXTRACTOR_H

#include <opencv2/opencv.hpp>
#include <vector>

struct PixelFeature {
    cv::Vec3b colorRGB;      // RGB 颜色
    cv::Point position;      // 像素位置
    float gradientMagnitude; // 梯度幅值 (可选，如果需要使用梯度信息)
};

std::vector<std::vector<PixelFeature>> extractFeatures(const cv::Mat& image);

#endif
