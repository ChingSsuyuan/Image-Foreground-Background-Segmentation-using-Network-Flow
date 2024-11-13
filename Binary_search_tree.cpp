#include "BST.h"
#include <iostream>

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

// 插入节点
std::shared_ptr<BSTNode> BST::insertNode(std::shared_ptr<BSTNode> node, const PixelFeature& feature) {
    if (!node) return std::make_shared<BSTNode>(feature);

    // 基于 RGB 总和的排序
    int featureSum = calculateRGBSum(feature);
    int nodeSum = calculateRGBSum(node->data);

    if (featureSum < nodeSum) {
        node->left = insertNode(node->left, feature);
    } else {
        node->right = insertNode(node->right, feature);
    }
    return node;
}

// 插入接口函数
void BST::insert(const PixelFeature& feature) {
    root = insertNode(root, feature);
}

// 中序遍历
void BST::inorderTraversal(std::shared_ptr<BSTNode> node) const {
    if (node) {
        inorderTraversal(node->left);
        std::cout << "RGB: (" << node->data.color[0] << ", " 
                  << node->data.color[1] << ", " << node->data.color[2] << ") "
                  << "位置: (" << node->data.position.x << ", " << node->data.position.y << ")" << std::endl;
        inorderTraversal(node->right);
    }
}

// 外部调用的中序遍历接口
void BST::inorderTraversal() const {
    inorderTraversal(root);
}
