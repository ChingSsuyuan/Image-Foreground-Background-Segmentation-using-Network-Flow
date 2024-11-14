#ifndef CONSTRUCT_GRAPH_H
#define CONSTRUCT_GRAPH_H

#include <opencv2/opencv.hpp>
#include <memory>
#include <unordered_map>
#include <vector>


// PairHash，用于为 std::pair<int, int> 定义哈希
struct PairHash {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2>& p) const {
        return std::hash<T1>()(p.first) ^ std::hash<T2>()(p.second);
    }
};

// ListNode 结构，用于表示邻接链表的节点
struct ListNode {
    int x;                      // 节点 x 坐标
    int y;                      // 节点 y 坐标
    int weight;                 // 到相邻点的权重
    std::shared_ptr<ListNode> next; // 指向链表下一个节点

    ListNode(int x_, int y_, int w) : x(x_), y(y_), weight(w), next(nullptr) {}
};

// AdjacencyList 类型，使用 unordered_map 存储邻接表
using AdjacencyList = std::unordered_map<std::pair<int, int>, std::shared_ptr<ListNode>, PairHash>;

// 计算特征之间相似度权重的函数
int calculateSimilarityWeight(const PixelFeature& feature1, const PixelFeature& feature2);

// 生成整个图的邻接表
AdjacencyList generateGraph(cv::Mat& image, const std::vector<std::vector<PixelFeature>>& features);

#endif // CONSTRUCT_GRAPH_H
