#include <opencv2/opencv.hpp>
#include <iostream>
#include "FeatureExtractor.h"
#include "BST.h"

using namespace cv;
int main() {
    cv::Mat image = cv::imread("path/to/your/image.jpg");

    cv::Mat gradMag; // 假设梯度幅值已计算
    // 生成特征矩阵
    std::vector<std::vector<PixelFeature>> features(image.rows, std::vector<PixelFeature>(image.cols));
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            cv::Vec3b color = image.at<cv::Vec3b>(y, x);
            float gradient = gradMag.at<float>(y, x);
            features[y][x] = PixelFeature(color, cv::Point(x, y), gradient);
        }
    }

    // 使用邻接表结合哈希表存储图像的邻接信息
    std::unordered_map<std::string, std::shared_ptr<ListNode>> adjacencyList;

    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            PixelFeature& pf = features[y][x];
            std::string nodeName = "(" + std::to_string(pf.position.x) + ", " + std::to_string(pf.position.y) + ")";
            adjacencyList[nodeName] = createAdjacencyList(pf, features, image.rows, image.cols);
        }
    }

    // 打印邻接表
    for (const auto& entry : adjacencyList) {
        std::cout << "像素 " << entry.first << " 的邻接点:" << std::endl;
        std::shared_ptr<ListNode> current = entry.second;
        while (current) {
            std::cout << " -> " << current->name << " [权重: " << current->weight << "]";
            current = current->next;
        }
        std::cout << std::endl;
    }

    return 0;
}
