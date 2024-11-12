#ifndef BST_H
#define BST_H

#include "FeatureExtractor.h"
#include <memory>


struct BSTNode {
    PixelFeature data;
    std::shared_ptr<BSTNode> left;
    std::shared_ptr<BSTNode> right;

    BSTNode(const PixelFeature& feature) : data(feature), left(nullptr), right(nullptr) {}
};

// 定义二叉搜索树类
class BST {
public:
    BST() : root(nullptr) {}

    // 插入节点，基于 RGB 总和排序
    void insert(const PixelFeature& feature);

    // 中序遍历输出数据
    void inorderTraversal() const;

private:
    std::shared_ptr<BSTNode> root;

    std::shared_ptr<BSTNode> insertNode(std::shared_ptr<BSTNode> node, const PixelFeature& feature);
    void inorderTraversal(std::shared_ptr<BSTNode> node) const;
};

#endif 
