#include "Segmentation.h"
#include <cmath>
#include <queue>
#include <climits>
#include <algorithm>

struct PixelFeature { 
    Vec3b colorRGB;  
    Point position;   
};

const int C = 100;             
const double SIGMA_RGB = 8.0;   // bigger more accurate but slower
const double ALPHA = 0.2;       // smaller more accurate but slower

// 计算权重
int Weight(const Vec3b& color1, const Vec3b& color2) {
    int dx = color1[0] - color2[0];
    int dy = color1[1] - color2[1];
    int dz = color1[2] - color2[2];
    double rgbDistance = std::sqrt(dx * dx + dy * dy + dz * dz);
    int weight = static_cast<int>(C * std::exp(-rgbDistance / SIGMA_RGB)); 
    return weight;
}

// 构建邻接矩阵
std::vector<std::vector<std::vector<int>>> Build_Matrix(const Mat& image) {
    int rows = image.rows, cols = image.cols;
    std::vector<std::vector<std::vector<int>>> Matrix(rows, std::vector<std::vector<int>>(cols, std::vector<int>(4, 0)));

    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            Vec3b color = image.at<Vec3b>(y, x);
            auto calcWeight = [&](int y2, int x2) -> int {
                Vec3b color2 = image.at<Vec3b>(y2, x2);
                double spatialDist = std::sqrt((y - y2) * (y - y2) + (x - x2) * (x - x2));
                double colorWeight = Weight(color, color2);
                return static_cast<int>(colorWeight * std::exp(-spatialDist * ALPHA));
            };
            if (y > 0) Matrix[y][x][0] = calcWeight(y - 1, x);
            if (y < rows - 1) Matrix[y][x][1] = calcWeight(y + 1, x);
            if (x > 0) Matrix[y][x][2] = calcWeight(y, x - 1);
            if (x < cols - 1) Matrix[y][x][3] = calcWeight(y, x + 1);
        }
    }
    return Matrix;
}

class Graph {
public:
    int V;
    vector<vector<pair<int, int>>> adj;
    vector<vector<int>> capacity;
    vector<vector<int>> flow;

    Graph(int V) : V(V) {
        adj.resize(V);
        capacity = vector<vector<int>>(V, vector<int>(V, 0));
        flow = vector<vector<int>>(V, vector<int>(V, 0));
    }

    void addEdge(int u, int v, int cap) {
        adj[u].emplace_back(v, cap);
        adj[v].emplace_back(u, cap); // 双向边
        capacity[u][v] = cap;
        capacity[v][u] = cap; // 反向边容量
    }

    void Kolmogorov(int source, int sink, vector<int>& S, vector<int>& T) {
        queue<int> q;
        vector<bool> visited(V, false);
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
            if (!visited[i]) T.push_back(i);
        }
    }
};

cv::Mat performSegmentation(const cv::Mat& inputImage) {
    int rows = inputImage.rows, cols = inputImage.cols;

    // 设置源点和汇点
    int source = (rows / 2) * cols + (cols / 2);  
    int sink = rows * cols - 2;
    Vec3b sourceColor = inputImage.at<Vec3b>(source / cols, source % cols);
    Vec3b sinkColor = inputImage.at<Vec3b>(sink / cols, sink % cols);

    // 构建邻接矩阵
    auto Matrix = Build_Matrix(inputImage);
    Graph graph(rows * cols);

    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            int u = y * cols + x;
            if (y > 0) graph.addEdge(u, (y - 1) * cols + x, Matrix[y][x][0]);
            if (y < rows - 1) graph.addEdge(u, (y + 1) * cols + x, Matrix[y][x][1]);
            if (x > 0) graph.addEdge(u, y * cols + (x - 1), Matrix[y][x][2]);
            if (x < cols - 1) graph.addEdge(u, y * cols + (x + 1), Matrix[y][x][3]);
        }
    }

    vector<int> S, T;
    graph.Kolmogorov(source, sink, S, T);

    // 构建结果图像
    Mat result = inputImage.clone();
    for (int i = 0; i < rows * cols; ++i) {
        int y = i / cols, x = i % cols;
        if (find(S.begin(), S.end(), i) != S.end()) {
            result.at<Vec3b>(y, x) = sourceColor;
        } else {
            result.at<Vec3b>(y, x) = sinkColor;
        }
    }

    circle(result, Point(source % cols, source / cols), 5, Scalar(0, 255, 0), -1);
    circle(result, Point(sink % cols, sink / cols), 5, Scalar(0, 0, 255), -1);
    return result;
}