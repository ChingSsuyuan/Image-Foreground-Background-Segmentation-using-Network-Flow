#include "EdmondsKarp.h"
#include <queue>
#include <climits>
#include <unordered_set>
#include <stack>

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
std::shared_ptr<ReachableNode> getReachableNodes(const AdjacencyList& adjList, std::pair<int, int> source) {
    std::unordered_map<std::pair<int, int>, bool, PairHash> visited;
    std::stack<std::pair<int, int>> stack;
    stack.push(source);
    visited[source] = true;

    std::shared_ptr<ReachableNode> head = nullptr;
    std::shared_ptr<ReachableNode> current = nullptr;

    while (!stack.empty()) {
        auto u = stack.top();
        stack.pop();

        std::shared_ptr<ReachableNode> newNode = std::make_shared<ReachableNode>(u.first, u.second);
        if (!head) {
            head = newNode;
            current = head;
        } else {
            current->next = newNode;
            current = newNode;
        }

        std::shared_ptr<ListNode> node = adjList.at(u);
        while (node) {
            std::pair<int, int> v = {node->x, node->y};

            if (!visited[v] && node->weight > 0) {
                stack.push(v);
                visited[v] = true;
            }
            node = node->next;
        }
    }

    return head;
}
// Edmonds-Karp Maximum Flow Algorithm Implementation
std::shared_ptr<ReachableNode> edmondsKarp(AdjacencyList& adjList, std::pair<int, int> source, std::pair<int, int> sink, std::shared_ptr<ReachableNode>& reachableNodes) {
    std::unordered_map<std::pair<int, int>, std::pair<int, int>, PairHash> parent;
    int maxFlow = 0;

    // Finding the path to amplification and calculating traffic
    while (bfs(adjList, source, sink, parent)) {
        int pathFlow = INT_MAX;

        // Find the minimum capacity on the path
        for (auto v = sink; v != source; v = parent[v]) {
            auto u = parent[v];
            auto node = adjList[u];

            while (node && (node->x != v.first || node->y != v.second)) {
                node = node->next;
            }
            if (node) pathFlow = std::min(pathFlow, node->weight);
        }

        // Updating the residual map
        for (auto v = sink; v != source; v = parent[v]) {
            auto u = parent[v];

            auto node = adjList[u];
            while (node && (node->x != v.first || node->y != v.second)) {
                node = node->next;
            }
            if (node) node->weight -= pathFlow;

            // Update Reverse Edge
            auto reverseNode = adjList[v];
            while (reverseNode && (reverseNode->x != u.first || reverseNode->y != u.second)) {
                reverseNode = reverseNode->next;
            }
            if (reverseNode) reverseNode->weight += pathFlow;
            else {
                // Update Reverse Edge
                std::shared_ptr<ListNode> newNode = std::make_shared<ListNode>(u.first, u.second, pathFlow);
                newNode->next = adjList[v];
                adjList[v] = newNode;
            }
        }

        maxFlow += pathFlow;
    }

    // Get all the nodes reachable from the source, stored in a chained table
    reachableNodes = getReachableNodes(adjList, source);

    return reachableNodes;
}
