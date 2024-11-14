#ifndef EDMONDS_KARP_H
#define EDMONDS_KARP_H

#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <memory>
#include <vector>
#include <utility>
#include "Construct_Graph.h"

// 使用类型别名表示邻接链表
using AdjacencyList = std::unordered_map<std::pair<int, int>, std::shared_ptr<ListNode>, PairHash>;

// BFS 寻找增广路径，返回是否找到增广路径
bool bfs(const AdjacencyList& adjList, std::pair<int, int> source, std::pair<int, int> sink,
         std::unordered_map<std::pair<int, int>, std::pair<int, int>, PairHash>& parent);

// Edmonds-Karp 算法，计算从 source 到 sink 的最大流
int edmondsKarp(AdjacencyList& adjList, std::pair<int, int> source, std::pair<int, int> sink);

#endif // EDMONDS_KARP_H
