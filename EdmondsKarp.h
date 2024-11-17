#ifndef EDMONDS_KARP_H
#define EDMONDS_KARP_H

#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <memory>
#include <vector>
#include <utility>
#include "Construct_Graph.h"

struct ReachableNode {
    int x, y;
    std::shared_ptr<ReachableNode> next;
    ReachableNode(int x_, int y_) : x(x_), y(y_), next(nullptr) {}
};

// BFS seeks for an augmentation path, returns whether the augmentation path was found or not.
bool bfs(const AdjacencyList& adjList, std::pair<int, int> source, std::pair<int, int> sink,
         std::unordered_map<std::pair<int, int>, std::pair<int, int>, PairHash>& parent);

// Edmonds-Karp algorithm to calculate the maximum flow from source to sink
std::shared_ptr<ReachableNode> edmondsKarp(AdjacencyList& adjList, std::pair<int, int> source, std::pair<int, int> sink, std::shared_ptr<ReachableNode>& reachableNodes);

#endif // EDMONDS_KARP_H
