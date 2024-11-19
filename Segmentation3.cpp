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

// 计算两个像素点颜色的权重
int Weight(const Vec3b& color1, const Vec3b& color2) {
    int dx = color1[0] - color2[0];
    int dy = color1[1] - color2[1];
    int dz = color1[2] - color2[2];
    double rgbDistance = std::sqrt(dx * dx + dy * dy + dz * dz);
    int weight = static_cast<int>(C * std::exp(-rgbDistance)); // 这里采用颜色距离的指数函数计算权重
    return std::clamp(weight, 1, 100);
}

// 构建图像的权重矩阵
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
                Matrix[y][x][0] = Weight(pf.colorRGB, image.at<Vec3b>(y - 1, x)); // 上
            }
            if (y < image.rows - 1) {
                Matrix[y][x][1] = Weight(pf.colorRGB, image.at<Vec3b>(y + 1, x)); // 下
            }
            if (x > 0) {
                Matrix[y][x][2] = Weight(pf.colorRGB, image.at<Vec3b>(y, x - 1)); // 左
            }
            if (x < image.cols - 1) {
                Matrix[y][x][3] = Weight(pf.colorRGB, image.at<Vec3b>(y, x + 1)); // 右
            }
        }
    }
    return Matrix;
}

class Graph {
public:
    int V;  // 顶点数
    vector<vector<pair<int, int>>> adj;  // 邻接表，每个元素是一个<pair<邻接点, 权重>>
    vector<vector<int>> capacity;  // 容量矩阵
    vector<vector<int>> flow;      // 流量矩阵

    Graph(int V) : V(V) {
        adj.resize(V);
        capacity = vector<vector<int>>(V, vector<int>(V, 0));
        flow = vector<vector<int>>(V, vector<int>(V, 0));
    }

    void addEdge(int u, int v, int cap) {
        adj[u].push_back({v, cap});
        adj[v].push_back({u, 0});  // 反向边，容量为0
        capacity[u][v] = cap;
    }

    // BFS构建层次图
    bool bfs(int source, vector<int>& level) {
        fill(level.begin(), level.end(), -1);
        queue<int> q;
        q.push(source);
        level[source] = 0;

        while (!q.empty()) {
            int u = q.front();
            q.pop();

            for (auto& edge : adj[u]) {
                int v = edge.first;
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
            int v = adj[u][i].first;
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

    // 使用Dinic算法进行最大流计算
    void dinic(int source, int sink, vector<int>& S, vector<int>& T) {
    int maxFlow = 0;
    vector<int> level(V);
    vector<int> ptr(V);

    while (bfs(source, level)) {
        fill(ptr.begin(), ptr.end(), 0);

        while (int pushed = dfs(source, sink, level, ptr, INT_MAX)) {
            maxFlow += pushed;
        }
    }

    vector<bool> visited(V, false);
    queue<int> q;
    q.push(source);
    visited[source] = true;

    while (!q.empty()) {
        int u = q.front();
        q.pop();
        S.push_back(u);

        for (auto& edge : adj[u]) {
            int v = edge.first;
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

    // 输出S和T的大小
    cout << "Set S contains " << S.size() << " nodes." << endl;
    cout << "Set T contains " << T.size() << " nodes." << endl;
}
};


int main() {
Mat image = imread("Pictures/110*100.png");

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

    // 运行 Dinic 算法
    vector<int> S, T;
    graph.dinic(source, sink, S, T);
    
    // // 输出可达的节点集合S和T
    // cout << "Set S (reachable nodes): ";
    // for (int node : S) {
    //     cout << node << " ";
    // }
    // cout << endl; 
    // cout << "Set T : ";
    // for (int node : T) {
    //     cout << node << " ";
    // }
    // cout << endl;
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;
    cout << "Time taken: " << duration.count() << " seconds" << endl;
    return 0;
}