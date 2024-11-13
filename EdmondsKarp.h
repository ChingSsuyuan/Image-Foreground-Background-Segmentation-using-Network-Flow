#ifndef EDMONDSKARP_H
#define EDMONDSKARP_H

#include <vector>

class EdmondsKarp {
public:
    EdmondsKarp(int numVertices); // 构造函数，初始化顶点数和图
    void addEdge(int u, int v, int capacity); // 添加带容量的边
    int maxFlow(int source, int sink); // 计算从源到汇的最大流量

private:
    int numVertices; // 顶点数量
    std::vector<std::vector<int>> capacityGraph; // 容量图的邻接矩阵表示
    bool bfs(int source, int sink, std::vector<int>& parent); // 使用BFS查找增广路径
};

#endif // EDMONDSKARP_H
