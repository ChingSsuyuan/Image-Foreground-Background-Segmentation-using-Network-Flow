#include "EdmondsKarp.h"
#include <vector>
#include <queue>
#include <climits>

// Constructor to initialise the capacity matrix of the graph
EdmondsKarp::EdmondsKarp(int numVertices) : numVertices(numVertices), capacityGraph(numVertices, std::vector<int>(numVertices, 0)) {}

// Adding an edge with capacity
void EdmondsKarp::addEdge(int u, int v, int capacity) {
    capacityGraph[u][v] = capacity;
}

// Use BFS to find the augmented path and update the parent node on the path
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
            if (!visited[v] && capacityGraph[u][v] > 0) { // Remaining capacity exists and has not been accessed
                q.push(v);
                parent[v] = u;
                visited[v] = true;
                if (v == sink) return true; //Reach the meeting point, return true
            }
        }
    }
    return false; // No enhancement path
}

// Calculate maximum flow
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
