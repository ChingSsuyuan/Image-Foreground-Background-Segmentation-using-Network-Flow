#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <memory>
#include <string>
#include "BST.h"
#include "FeatureExtractor.h"

const int C = 100;           // 权重放大常数
const double SIGMA_RGB = 10.0;  // RGB 距离衰减系数
const double SIGMA_GRAD = 5.0;  // 梯度幅值距离衰减系数
const double ALPHA = 0.5;    // RGB 差异的权重
const double BETA = 0.5;     // 梯度差异的权重

// 计算 3D 相似度权重，结合 RGB 差异和梯度差异
int calculateSimilarityWeight(const PixelFeature& feature1, const PixelFeature& feature2) {
    // 计算 RGB 差异
    int dx = feature1.colorRGB[0] - feature2.colorRGB[0];
    int dy = feature1.colorRGB[1] - feature2.colorRGB[1];
    int dz = feature1.colorRGB[2] - feature2.colorRGB[2];
    double rgbDistance = std::sqrt(dx * dx + dy * dy + dz * dz);
    
    // 计算梯度差异
    double gradDistance = std::abs(feature1.gradientMagnitude - feature2.gradientMagnitude);
    
    // 综合计算权重
    int weight = static_cast<int>(C * std::exp(- (ALPHA * (rgbDistance / SIGMA_RGB) + BETA * (gradDistance / SIGMA_GRAD))));
    return weight;
}
// 链表节点结构体
struct ListNode {
    std::string name;            // 节点名字 (x, y)
    int weight;                  // 到相邻点的权重
    std::shared_ptr<ListNode> next; // 指向链表下一个节点

    ListNode(const std::string& n, int w) : name(n), weight(w), next(nullptr) {}
};

// 创建链表的辅助函数
std::shared_ptr<ListNode> createAdjacencyList(const PixelFeature& feature, cv::Mat& image, const std::vector<std::vector<PixelFeature>>& features) {
    std::shared_ptr<ListNode> head = std::make_shared<ListNode>("(" + std::to_string(feature.position.x) + ", " + std::to_string(feature.position.y) + ")", 0);
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

            std::shared_ptr<ListNode> newNode = std::make_shared<ListNode>("(" + std::to_string(nx) + ", " + std::to_string(ny) + ")", weight);
            current->next = newNode;
            current = newNode;
        }
    }
    
    return head;
}


// 插入节点（根据颜色相似度权重进行排序）
// std::shared_ptr<BSTNode> BST::insertNode(std::shared_ptr<BSTNode> node, const PixelFeature& feature) {
//     if (!node) return std::make_shared<BSTNode>(feature);

//     // 使用颜色相似度计算权重
//     int featureWeight = calculateColorSimilarityWeight(feature, node->data);

//     if (featureWeight > 50) { // 假设权重高于 50 表示颜色相似度较高，则插入左子树
//         node->left = insertNode(node->left, feature);
//     } else {
//         node->right = insertNode(node->right, feature);
//     }
//     return node;
// }


// 插入接口函数
void BST::insert(const PixelFeature& feature) {
    root = insertNode(root, feature);
}


// void BST::inorderTraversal(std::shared_ptr<BSTNode> node) const {
//     if (node) {
//         inorderTraversal(node->left);
//         std::cout << "RGB: (" << node->data.color[0] << ", " 
//                   << node->data.color[1] << ", " << node->data.color[2] << ") "
//                   << "位置: (" << node->data.position.x << ", " << node->data.position.y << ")" << std::endl;
//         inorderTraversal(node->right);
//     }
// }

// 外部调用的中序遍历接口
void BST::inorderTraversal() const {
    inorderTraversal(root);
}
