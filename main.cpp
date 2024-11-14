#include <opencv2/opencv.hpp>
#include <iostream>
#include "FeatureExtractor.h"
// #include "BST.h"

using namespace cv;
using namespace std;
int main() {
    
    string path = "Pictures/p4.png";
    Mat image = imread(path, IMREAD_COLOR);

    if (!image.data)
    {
        cout << "This is en empty image"<<endl;
        return -1;
    }
    vector<vector<PixelFeature>> extraction = extractFeatures(image);
    for (vector<PixelFeature> i : extraction){
        for (PixelFeature j : i){
            cout << "colorRGB " <<j.colorRGB << endl;
            cout << "gradientMagnitude " << j.gradientMagnitude << endl;
            cout << "position " << j.position << endl;
        }
    }
    namedWindow("image", WINDOW_AUTOSIZE);
    imshow("image", image);
    waitKey(0);
    return 0;
}
