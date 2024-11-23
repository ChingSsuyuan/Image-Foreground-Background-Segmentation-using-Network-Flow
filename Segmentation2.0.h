#ifndef SEGMENTATION_H
#define SEGMENTATION_H

#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;

// Structure for compressed image data
struct CompressedImage {
    Mat compressed;       // Compressed image
    int originalRows;     // Original image height
    int originalCols;     // Original image width
    vector<vector<Vec3b>> blockColors;  // Original colors of each pixel
};

// Function declarations
cv::Mat performSegmentation(const cv::Mat& inputImage);
cv::Mat performCompressedSegmentation(const cv::Mat& inputImage, int blockSize = 2);

#endif // SEGMENTATION_H
