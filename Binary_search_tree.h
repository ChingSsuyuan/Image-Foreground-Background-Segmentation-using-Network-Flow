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

    void insert(const PixelFeature& feature);
    void inorderTraversal() const;

private:
    std::shared_ptr<BSTNode> root;

    std::shared_ptr<BSTNode> insertNode(std::shared_ptr<BSTNode> node, const PixelFeature& feature);
    void inorderTraversal(std::shared_ptr<BSTNode> node) const;
};

#endif 
