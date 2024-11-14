#include <opencv2/opencv.hpp>
#include "FeatureExtractor.h"
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    // 读取图像
    Mat image = imread("Image-Foreground-Background-Segmentation-using-Network-Flow-main/Pictures/wallhaven-p8yedm_1920x1080.png"); // 替换为图像的实际路径

    if (image.empty()) {
        cout << "图像加载失败，请检查路径。" << endl;
        return -1;
    }

    // 调用特征提取函数
    vector<vector<PixelFeature>> features = extractFeatures(image);

    // 打印一些特征信息（示例）
    for (int y = 0; y < image.rows; y += 10) { // 每10行打印一次，避免输出过多
        for (int x = 0; x < image.cols; x += 10) { // 每10列打印一次
            PixelFeature pf = features[y][x];
            cout << "位置: (" << pf.position.x << ", " << pf.position.y << "), ";
            cout << "颜色: (" << (int)pf.colorRGB[0] << ", " << (int)pf.colorRGB[1] << ", " << (int)pf.colorRGB[2] << "), ";
            cout << "梯度幅值: " << pf.gradientMagnitude << endl;
        }
    }

    return 0;
}
