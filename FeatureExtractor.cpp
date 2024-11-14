#include "EdmondsKarp.h"
#include <queue>
#include <climits>

// BFS 寻找增广路径，返回是否找到增广路径
bool bfs(const AdjacencyList& adjList, std::pair<int, int> source, std::pair<int, int> sink,
         std::unordered_map<std::pair<int, int>, std::pair<int, int>, PairHash>& parent) {
    std::queue<std::pair<int, int>> queue;
    std::unordered_map<std::pair<int, int>, bool, PairHash> visited;

    queue.push(source);
    visited[source] = true;

    while (!queue.empty()) {
        auto u = queue.front();
        queue.pop();

        std::shared_ptr<ListNode> node = adjList.at(u);
        while (node) {
            std::pair<int, int> v = {node->x, node->y};

            if (!visited[v] && node->weight > 0) {
                queue.push(v);
                visited[v] = true;
                parent[v] = u;

                if (v == sink) return true;
            }
            node = node->next;
        }
    }
    return false;
}

// Edmonds-Karp 最大流算法实现
int edmondsKarp(AdjacencyList& adjList, std::pair<int, int> source, std::pair<int, int> sink) {
    std::unordered_map<std::pair<int, int>, std::pair<int, int>, PairHash> parent;
    int maxFlow = 0;

    // 寻找增广路径并计算流量
    while (bfs(adjList, source, sink, parent)) {
        int pathFlow = INT_MAX;

        // 找到路径上的最小容量
        for (auto v = sink; v != source; v = parent[v]) {
            auto u = parent[v];
            auto node = adjList[u];

            while (node && (node->x != v.first || node->y != v.second)) {
                node = node->next;
            }
            if (node) pathFlow = std::min(pathFlow, node->weight);
        }

        // 更新残余图
        for (auto v = sink; v != source; v = parent[v]) {
            auto u = parent[v];

            auto node = adjList[u];
            while (node && (node->x != v.first || node->y != v.second)) {
                node = node->next;
            }
            if (node) node->weight -= pathFlow;

            // 更新反向边
            auto reverseNode = adjList[v];
            while (reverseNode && (reverseNode->x != u.first || reverseNode->y != u.second)) {
                reverseNode = reverseNode->next;
            }
            if (reverseNode) reverseNode->weight += pathFlow;
            else {
                // 若反向边不存在，创建新的反向边
                std::shared_ptr<ListNode> newNode = std::make_shared<ListNode>(u.first, u.second, pathFlow);
                newNode->next = adjList[v];
                adjList[v] = newNode;
            }
        }

        maxFlow += pathFlow;
    }

    return maxFlow;
}
