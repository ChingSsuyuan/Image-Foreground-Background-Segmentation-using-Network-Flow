#include "BST.h"
#include <iostream>

// 辅助函数：计算 RGB 的总和，用于排序
int calculateRGBSum(const PixelFeature& feature) {
    return feature.color[0] + feature.color[1] + feature.color[2];
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
