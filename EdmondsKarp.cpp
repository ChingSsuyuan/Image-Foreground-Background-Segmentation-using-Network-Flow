#include "EdmondsKarp.h"
#include <vector>
#include <queue>
#include <climits>

// 构造函数，初始化图的容量矩阵
EdmondsKarp::EdmondsKarp(int numVertices) : numVertices(numVertices), capacityGraph(numVertices, std::vector<int>(numVertices, 0)) {}

// 添加带容量的边
void EdmondsKarp::addEdge(int u, int v, int capacity) {
    capacityGraph[u][v] = capacity;
}

// 使用 BFS 查找增广路径并更新路径上的父节点
bool EdmondsKarp::bfs(int source, int sink, std::vector<int>& parent) {
    std::vector<bool> visited(numVertices, false);
    std::queue<int> q;
    q.push(source);
    visited[source] = true;
    parent[source] = -1;

    while (!q.empty()) {
        int u = q.front();
        q.pop();

        for (int v = 0; v < numVertices; ++v) {
            if (!visited[v] && capacityGraph[u][v] > 0) { // 存在剩余容量且未访问过
                q.push(v);
                parent[v] = u;
                visited[v] = true;
                if (v == sink) return true; // 到达汇点，返回 true
            }
        }
    }
    return false; // 无增广路径
}

// 计算最大流
int EdmondsKarp::maxFlow(int source, int sink) {
    int maxFlow = 0;
    std::vector<std::vector<int>> residualGraph = capacityGraph;
    std::vector<int> parent(numVertices);

    while (bfs(source, sink, parent)) {
        int pathFlow = INT_MAX;
        for (int v = sink; v != source; v = parent[v]) {
            int u = parent[v];
            pathFlow = std::min(pathFlow, residualGraph[u][v]);
        }

        for (int v = sink; v != source; v = parent[v]) {
            int u = parent[v];
            residualGraph[u][v] -= pathFlow;
            residualGraph[v][u] += pathFlow;
        }

        maxFlow += pathFlow;
    }

    return maxFlow;
}
