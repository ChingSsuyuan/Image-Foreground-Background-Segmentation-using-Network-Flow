#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <queue>
#include <climits>
#include <unordered_map>

using namespace std;
using namespace cv;
struct PixelFeature {
    Vec3b colorRGB;  
    Point position;   
};
const int C = 100;             
const double SIGMA_RGB = 10.0;  
const double ALPHA = 0.5;       

int Weight(const Vec3b& color1, const Vec3b& color2) {
    int dx = color1[0] - color2[0];
    int dy = color1[1] - color2[1];
    int dz = color1[2] - color2[2];
    double rgbDistance = std::sqrt(dx * dx + dy * dy + dz * dz);
    // int weight = static_cast<int>(C * std::exp(-ALPHA * (rgbDistance / SIGMA_RGB)));
    int weight = static_cast<int>(C * std::exp(-rgbDistance));
    return std::clamp(weight, 1, 100);
}

std::vector<std::vector<std::vector<int>>> Build_Matrix(const Mat& image) {
    std::vector<std::vector<PixelFeature>> features(image.rows, std::vector<PixelFeature>(image.cols));
    std::vector<std::vector<std::vector<int>>> Matrix(image.rows, std::vector<std::vector<int>>(image.cols, std::vector<int>(4, 0)));

    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            PixelFeature pf;
            pf.colorRGB = image.at<Vec3b>(y, x);
            pf.position = Point(x, y);
            features[y][x] = pf;
            if (y > 0) {
                Matrix[y][x][0] = Weight(pf.colorRGB, image.at<Vec3b>(y - 1, x)); // up
            }
            if (y < image.rows - 1) {
                Matrix[y][x][1] = Weight(pf.colorRGB, image.at<Vec3b>(y + 1, x)); // down
            }
            if (x > 0) {
                Matrix[y][x][2] = Weight(pf.colorRGB, image.at<Vec3b>(y, x - 1)); // left
            }
            if (x < image.cols - 1) {
                Matrix[y][x][3] = Weight(pf.colorRGB, image.at<Vec3b>(y, x + 1)); // right
            }
        }
    }
    return Matrix;
}
class Graph {
public:
    int V;  // number of vertices
    vector<vector<int>> capacity;  
    vector<vector<int>> flow;      
    vector<vector<int>> adj;       

    Graph(int V) : V(V) {
        capacity = vector<vector<int>>(V, vector<int>(V, 0));
        flow = vector<vector<int>>(V, vector<int>(V, 0));
        adj = vector<vector<int>>(V);
    }

    void addEdge(int u, int v, int cap) {
        capacity[u][v] = cap;
        adj[u].push_back(v);
        adj[v].push_back(u);  
    }

    bool bfs(int source, int sink, vector<int>& parent) {
        vector<bool> visited(V, false);
        queue<int> q;
        q.push(source);
        visited[source] = true;

        while (!q.empty()) {
            int u = q.front();
            q.pop();

            for (int v : adj[u]) {
                if (!visited[v] && capacity[u][v] - flow[u][v] > 0) {
                    parent[v] = u;
                    if (v == sink) return true;
                    q.push(v);
                    visited[v] = true;
                }
            }
        }
        return false;
    }

    // edmondsKarp
    int edmondsKarp(int source, int sink, vector<int>& S, vector<int>& T) {
        int maxFlow = 0;
        vector<int> parent(V);
        while (bfs(source, sink, parent)) {
            int pathFlow = INT_MAX;
            for (int v = sink; v != source; v = parent[v]) {
                int u = parent[v];
                pathFlow = min(pathFlow, capacity[u][v] - flow[u][v]);
            }

            for (int v = sink; v != source; v = parent[v]) {
                int u = parent[v];
                flow[u][v] += pathFlow;
                flow[v][u] -= pathFlow;  
            }

            maxFlow += pathFlow;
        }

        // Use BFS to find the set of nodes S that are reachable from the source node
        vector<bool> visited(V, false);
        queue<int> q;
        q.push(source);
        visited[source] = true;

        while (!q.empty()) {
            int u = q.front();
            q.pop();
            S.push_back(u);  // 加入集合S

            for (int v : adj[u]) {
                if (!visited[v] && capacity[u][v] - flow[u][v] > 0) {
                    visited[v] = true;
                    q.push(v);
                }
            }
        }
        for (int i = 0; i < V; ++i) {
            if (!visited[i]) {
                T.push_back(i);
            }
        }

        return maxFlow;
    }
};

int main() {
    Mat image = imread("Pictures/160*120.png");

    if (image.empty()) {
        cout << "Fail to read image!" << endl;
        return -1;  
    }
    int rows = image.rows;
    int cols = image.cols;


    auto start = chrono::high_resolution_clock::now();


    auto Matrix = Build_Matrix(image);


    int source = (rows / 2) * cols + (cols / 2);  
    int sink = rows * cols - 1;  


    Graph graph(rows * cols); 

    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            int u = y * cols + x;  
            if (y > 0) {
                int v = (y - 1) * cols + x;  
                graph.addEdge(u, v, Matrix[y][x][0]);  
            }
            if (y < rows - 1) {
                int v = (y + 1) * cols + x;  
                graph.addEdge(u, v, Matrix[y][x][1]);  
            }
            if (x > 0) {
                int v = y * cols + (x - 1);  
                graph.addEdge(u, v, Matrix[y][x][2]);  
            }
            if (x < cols - 1) {
                int v = y * cols + (x + 1);  
                graph.addEdge(u, v, Matrix[y][x][3]);  
            }
        }
    }


    vector<int> S, T;
    int maxFlow = graph.edmondsKarp(source, sink, S, T);
    cout << "Set S (reachable nodes): ";
    for (int node : S) {
        cout << node << " ";
    }
    cout << endl; 
    cout << "Set T : ";
    for (int node : T) {
        cout << node << " ";
    }
    cout << endl;

    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double> duration = end - start;
    cout << "Time taken: " << duration.count() << " seconds" << endl;
    return 0;
}
