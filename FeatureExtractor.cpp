#include "FeatureExtractor.h"
#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
// struct PixelFeature {
//     Vec3b colorRGB;
//     Point position;
//     float gradientMagnitude; //
// };
std::vector<std::vector<PixelFeature>> extractFeatures(const Mat& image) {
    Mat gray, gradX, gradY, gradMag;

    cvtColor(image, gray, COLOR_BGR2GRAY);

    // Calculate the gradient in the x and y directions
    Sobel(gray, gradX, CV_32F, 1, 0, 3);  // x 
    Sobel(gray, gradY, CV_32F, 0, 1, 3);  // y 

    // Calculate the gradient magnitude
    magnitude(gradX, gradY, gradMag);  
    

    std::vector<std::vector<PixelFeature>> features(image.rows, std::vector<PixelFeature>(image.cols));

    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            PixelFeature pf;
            

            pf.colorRGB = image.at<Vec3b>(y, x);
            
   
            pf.position = Point(x, y);
            
     
            pf.gradientMagnitude = gradMag.at<float>(y, x);  // 
            
   
            features[y][x] = pf;
        }
    }

    return features;
}
