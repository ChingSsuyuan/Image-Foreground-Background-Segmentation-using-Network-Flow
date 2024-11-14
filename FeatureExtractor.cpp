#include "FeatureExtractor.h"
#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;

std::vector<std::vector<PixelFeature>> extractFeatures(const Mat& image) {
    Mat gray, gradX, gradY, gradMag;

    cvtColor(image, gray, COLOR_BGR2GRAY);

    // 计算 x 和 y 方向上的梯度
    Sobel(gray, gradX, CV_32F, 1, 0, 3);  // x 方向梯度
    Sobel(gray, gradY, CV_32F, 0, 1, 3);  // y 方向梯度

    // 计算梯度幅值
    magnitude(gradX, gradY, gradMag);  // 计算梯度幅值
    
    // 初始化特征矩阵
    std::vector<std::vector<PixelFeature>> features(image.rows, std::vector<PixelFeature>(image.cols));

    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            PixelFeature pf;
            
            // 颜色信息
            pf.colorRGB = image.at<Vec3b>(y, x);
            
            // 位置
            pf.position = Point(x, y);
            
            // 梯度幅值
            pf.gradientMagnitude = gradMag.at<float>(y, x);  // 获取该点的梯度幅值
            
            // 保存特征信息
            features[y][x] = pf;
        }
    }

    return features;
}
