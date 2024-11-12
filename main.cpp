#include <opencv2/opencv.hpp>
#include <iostream>
#include "FeatureExtractor.h"
#include "BST.h"

using namespace cv;
using namespace std;

int main() {
    // 加载输入图像
    Mat image = imread("input.jpg");
    if (image.empty()) {
        cout << "Fail to read Image" << endl;
        return -1;
    }

    // 提取图像特征
    auto features = extractFeatures(image);

    // 创建BST对象
    BST bst;

    // 将特征数据存入BST
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            bst.insert(features[y][x]);
        }
    }

    // 遍历BST，输出特征数据
    cout << "BST data：" << endl;
    bst.inorderTraversal();

    return 0;
}
