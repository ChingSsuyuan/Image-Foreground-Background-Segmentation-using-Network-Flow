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
    vector<vector<int>> capacity;  // 容量矩阵
    vector<vector<int>> flow;      // 流量矩阵
    vector<vector<int>> adj;       // 邻接矩阵

    Graph(int V) : V(V) {
        capacity = vector<vector<int>>(V, vector<int>(V, 0));
        flow = vector<vector<int>>(V, vector<int>(V, 0));
        adj = vector<vector<int>>(V);
    }

    void addEdge(int u, int v, int cap) {
        capacity[u][v] = cap;
        adj[u].push_back(v);
        adj[v].push_back(u);  // 反向边
    }

    // BFS用于分层图的构建
    bool bfs(int source, vector<int>& level) {
        fill(level.begin(), level.end(), -1);
        queue<int> q;
        q.push(source);
        level[source] = 0;

        while (!q.empty()) {
            int u = q.front();
            q.pop();

            for (int v : adj[u]) {
                if (level[v] == -1 && capacity[u][v] - flow[u][v] > 0) {
                    level[v] = level[u] + 1;
                    q.push(v);
                }
            }
        }
        return level[V - 1] != -1;  // 如果汇点被标记，说明存在增广路径
    }

    // DFS搜索增广路径
    int dfs(int u, int sink, vector<int>& level, vector<int>& ptr, int flowToPush) {
        if (u == sink) return flowToPush;
        
        for (int& i = ptr[u]; i < adj[u].size(); ++i) {
            int v = adj[u][i];
            if (level[v] == level[u] + 1 && capacity[u][v] - flow[u][v] > 0) {
                int availableFlow = min(flowToPush, capacity[u][v] - flow[u][v]);
                int pushedFlow = dfs(v, sink, level, ptr, availableFlow);
                if (pushedFlow > 0) {
                    flow[u][v] += pushedFlow;
                    flow[v][u] -= pushedFlow;
                    return pushedFlow;
                }
            }
        }
        return 0;
    }

    // Dinic算法
    void dinic(int source, int sink, vector<int>& S, vector<int>& T) {
        int maxFlow = 0;
        vector<int> level(V);
        vector<int> ptr(V);

        // 进行分层图的构建和增广路径的查找
        while (bfs(source, level)) {
            fill(ptr.begin(), ptr.end(), 0);

            while (int pushed = dfs(source, sink, level, ptr, INT_MAX)) {
                maxFlow += pushed;
            }
        }

        // 使用BFS找到从源节点可以访问的节点集合S
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

        // 所有未访问的节点属于T
        for (int i = 0; i < V; ++i) {
            if (!visited[i]) {
                T.push_back(i);
            }
        }
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

    // 记录开始时间
    auto start = chrono::high_resolution_clock::now();

    // 构建权重矩阵
    auto Matrix = Build_Matrix(image);

    // 设置源节点为正中间的点
    int source = (rows / 2) * cols + (cols / 2);  
    int sink = rows * cols - 1;  // 右下角像素

    // 创建图对象
    Graph graph(rows * cols); 

    // 根据权重矩阵构建图的边
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            int u = y * cols + x;  // 当前像素的节点编号
            if (y > 0) {
                int v = (y - 1) * cols + x;  // 上方像素的节点编号
                graph.addEdge(u, v, Matrix[y][x][0]);  // 上
            }
            if (y < rows - 1) {
                int v = (y + 1) * cols + x;  // 下方像素的节点编号
                graph.addEdge(u, v, Matrix[y][x][1]);  // 下
            }
            if (x > 0) {
                int v = y * cols + (x - 1);  // 左方像素的节点编号
                graph.addEdge(u, v, Matrix[y][x][2]);  // 左
            }
            if (x < cols - 1) {
                int v = y * cols + (x + 1);  // 右方像素的节点编号
                graph.addEdge(u, v, Matrix[y][x][3]);  // 右
            }
        }
    }


    vector<int> S, T;
    graph.dinic(source, sink, S, T);
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
