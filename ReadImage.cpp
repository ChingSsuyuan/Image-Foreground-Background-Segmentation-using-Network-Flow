#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

struct PixelFeature {
    Vec3b colorRGB;    // RGB color 
    Point position;    // image element position
    float gradient;    // gradient range
    float intensity;   // Grayscale value
};

vector<vector<PixelFeature>> extractFeatures(const Mat& image) {
    Mat gray, gradX, gradY, gradMag;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    Sobel(gray, gradX, CV_32F, 1, 0); // X 方向梯度
    Sobel(gray, gradY, CV_32F, 0, 1); // Y 方向梯度
    magnitude(gradX, gradY, gradMag); // 梯度幅度

    vector<vector<PixelFeature>> features(image.rows, vector<PixelFeature>(image.cols));

    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            PixelFeature pf;
            pf.colorRGB = image.at<Vec3b>(y, x);     // 提取颜色
            pf.position = Point(x, y);               // 提取位置
            pf.gradient = gradMag.at<float>(y, x);   // 提取梯度
            pf.intensity = gray.at<uchar>(y, x);     // 提取亮度

            features[y][x] = pf;
        }
    }

    return features;
}

int main() {
    Mat image = imread(".jpg");
    if (image.empty()) {
        cout << "Unable to read picture" << endl;
        return -1;
    }

    auto features = extractFeatures(image);

    // 后续代码：将特征输入图分割或最大流算法进行分割
}
