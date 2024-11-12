// FeatureExtractor.cpp
#include "FeatureExtractor.h"
#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;

std::vector<std::vector<PixelFeature>> extractFeatures(const Mat& image) {
    Mat gray, gradX, gradY, gradMag;
    cvtColor(image, gray, COLOR_BGR2GRAY);

    std::vector<std::vector<PixelFeature>> features(image.rows, std::vector<PixelFeature>(image.cols));
  
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            PixelFeature pf;
            pf.colorRGB = image.at<Vec3b>(y, x);
            pf.position = Point(x, y);
            features[y][x] = pf;
        }
    }

    return features;
}
