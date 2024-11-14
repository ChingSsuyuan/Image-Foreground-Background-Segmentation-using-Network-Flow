#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <memory>
#include <string>
#include "BST.h"

const int C = 100; // Constant for amplifying weights
const double SIGMA = 10.0; // Parameters controlling the decay rate

int calculate3DSimilarityWeight(const PixelFeature & feature1, const PixelFeature & feature2) {
    int dx = feature1.color[0] - feature2.color[0];
    int dy = feature1.color[1] - feature2.color[1];
    int dz = feature1.color[2] - feature2.color[2];
    double distance = std::sqrt(dx * dx + dy * dy + dz * dz);
    
    //Inverse Exponential Similarity Calculation
    int weight = static_cast<int>(C * std::exp(-distance / SIGMA));
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
std::shared_ptr<ListNode> createAdjacencyList(int x, int y, cv::Mat& image) {
    // 当前像素颜色特征
    cv::Vec3b pixelColor = image.at<cv::Vec3b>(y, x); // 注意OpenCV中的行列顺序
    PixelFeature feature(pixelColor[2], pixelColor[1], pixelColor[0]);

    // 头节点，名字是当前节点坐标 (x, y)
    std::shared_ptr<ListNode> head = std::make_shared<ListNode>("(" + std::to_string(x) + ", " + std::to_string(y) + ")", 0);
    std::shared_ptr<ListNode> current = head;

    // 上下左右四个方向的坐标偏移
    int dx[] = {0, 1, 0, -1}; // 右、下、左、上
    int dy[] = {-1, 0, 1, 0};
    
    // 遍历上下左右四个方向
    for (int i = 0; i < 4; ++i) {
        int nx = x + dx[i];
        int ny = y + dy[i];

        // 检查边界条件
        if (nx >= 0 && nx < image.cols && ny >= 0 && ny < image.rows) {
            // 获取相邻点的颜色特征
            cv::Vec3b neighborColor = image.at<cv::Vec3b>(ny, nx);
            PixelFeature neighborFeature(neighborColor[2], neighborColor[1], neighborColor[0]);

            // 计算相似性权重
            int weight = calculate3DSimilarityWeight(feature, neighborFeature);

            // 创建新链表节点并连接
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
