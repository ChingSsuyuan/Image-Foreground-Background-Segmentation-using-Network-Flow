#ifndef GRAPHMATRIX_H
#define GRAPHMATRIX_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>

// 常量定义
const int C = 100;              // 权重放大常量
const double SIGMA_RGB = 10.0;  // RGB距离衰减系数
const double SIGMA_GRAD = 5.0;  // 梯度幅值距离衰减系数
const double ALPHA = 0.5;       // RGB差异权重
const double BETA = 0;          // 梯度差异权重

// 像素特征结构
struct PixelFeature {
    cv::Vec3b colorRGB;          // 像素的 RGB 值
    double gradientMagnitude;    // 梯度幅值
    cv::Point position;          // 像素位置
};

// 计算权重函数声明
int calculateSimilarityWeight(const PixelFeature& feature1, const PixelFeature& feature2);

// 生成邻接矩阵函数声明
std::vector<std::vector<int>> generateGraphMatrix(const cv::Mat& image, const std::vector<std::vector<PixelFeature>>& features);

#endif // GRAPHMATRIX_H
