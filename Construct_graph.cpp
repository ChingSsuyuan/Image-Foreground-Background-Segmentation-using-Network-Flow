#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <memory>
#include <string>
#include "FeatureExtractor.h"
#include <unordered_map>

const int C = 100;           // Weight amplification constant
const double SIGMA_RGB = 10.0;  // RGB Distance attenuation factor
const double SIGMA_GRAD = 5.0;  // Gradient amplitude distance attenuation factor
const double ALPHA = 0.5;    // RGB Weighting of Differences
const double BETA = 0.5;     // Weighting of gradient differences

int calculateSimilarityWeight(const PixelFeature& feature1, const PixelFeature& feature2) {
    // Calculating RGB Differences
    int dx = feature1.colorRGB[0] - feature2.colorRGB[0];
    int dy = feature1.colorRGB[1] - feature2.colorRGB[1];
    int dz = feature1.colorRGB[2] - feature2.colorRGB[2];
    double rgbDistance = std::sqrt(dx * dx + dy * dy + dz * dz);
    
    // Calculating gradient differences
    double gradDistance = std::abs(feature1.gradientMagnitude - feature2.gradientMagnitude);
    
    // Combined calculation of weights
    int weight = static_cast<int>(C * std::exp(- (ALPHA * (rgbDistance / SIGMA_RGB) + BETA * (gradDistance / SIGMA_GRAD))));
    return weight;
}
// Using a hash table to store an adjacency table
using AdjacencyList = std::unordered_map<std::string, std::shared_ptr<ListNode>>;

// linked list node structure
struct ListNode {
    int x;                      // Node x-coordinate
    int y;                      // Node y-coordinate
    int weight;                 // Weights to neighbouring points
    std::shared_ptr<ListNode> next; // Points to the next node in the chain

    ListNode(int x_, int y_, int w) : x(x_), y(y_), weight(w), next(nullptr) {}
};

// 创建邻接表的辅助函数
std::shared_ptr<ListNode> createAdjacencyList(const PixelFeature& feature, cv::Mat& image, const std::vector<std::vector<PixelFeature>>& features, AdjacencyList& adjList) {
    std::string nodeName = "(" + std::to_string(feature.position.x) + ", " + std::to_string(feature.position.y) + ")";
    std::shared_ptr<ListNode> head = std::make_shared<ListNode>(nodeName, 0);
    std::shared_ptr<ListNode> current = head;

    int x = feature.position.x;
    int y = feature.position.y;
    int dx[] = {0, 1, 0, -1}; // 右、下、左、上
    int dy[] = {-1, 0, 1, 0};

    for (int i = 0; i < 4; ++i) {
        int nx = x + dx[i];
        int ny = y + dy[i];

        if (nx >= 0 && nx < image.cols && ny >= 0 && ny < image.rows) {
            const PixelFeature& neighborFeature = features[ny][nx];
            int weight = calculate3DSimilarityWeight(feature, neighborFeature);
            std::string neighborName = "(" + std::to_string(nx) + ", " + std::to_string(ny) + ")";

            std::shared_ptr<ListNode> newNode = std::make_shared<ListNode>(neighborName, weight);
            current->next = newNode;
            current = newNode;
        }
    }

    adjList[nodeName] = head->next;  // 不保存当前节点自身，只保存其邻居节点
    return head;
}

// 示例：生成整张图的邻接表
AdjacencyList generateGraph(cv::Mat& image, const std::vector<std::vector<PixelFeature>>& features) {
    AdjacencyList adjList;

    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            createAdjacencyList(features[y][x], image, features, adjList);
        }
    }

    return adjList;
}

// 插入接口函数
void BST::insert(const PixelFeature& feature) {
    root = insertNode(root, feature);
}

// 外部调用的中序遍历接口
void BST::inorderTraversal() const {
    inorderTraversal(root);
}
