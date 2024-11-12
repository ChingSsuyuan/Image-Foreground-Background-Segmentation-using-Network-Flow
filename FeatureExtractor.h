// FeatureExtractor.h
#ifndef FEATURE_EXTRACTOR_H
#define FEATURE_EXTRACTOR_H

#include <opencv2/opencv.hpp>
#include <vector>

struct PixelFeature {
    cv::Vec3b colorRGB;    // RGB 颜色值
    cv::Point position;    // 像素位置
    float gradient;        // 梯度幅度
    float intensity;       // 灰度值
};

// 提取图像特征的函数声明
std::vector<std::vector<PixelFeature>> extractFeatures(const cv::Mat& image);

#endif // FEATURE_EXTRACTOR_H
