#include <opencv2/opencv.hpp>
#include <iostream>
#include "FeatureExtractor.h"
#include "BST.h"

using namespace cv;
using namespace std;

int main() {
    Mat image = imread("input.jpg");
    if (image.empty()) {
        cout << "Fail to read Image" << endl;
        return -1;
    }

    auto features = extractFeatures(image);

    BST bst;

    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            bst.insert(features[y][x]);
        }
    }

    cout << "BST data：" << endl;
    bst.inorderTraversal();

    return 0;
}
