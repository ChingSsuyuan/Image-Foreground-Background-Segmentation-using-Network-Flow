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
const double SIGMA_RGB = 10.0;   //bigger more accurate but slower
const double ALPHA = 0.01;       //smaller more accurate but slower

int Weight(const Vec3b& color1,  const Vec3b& color2) {
    int dx = color1[0] - color2[0];
    int dy = color1[1] - color2[1];
    int dz = color1[2] - color2[2];
    double rgbDistance = std::sqrt(dx * dx + dy * dy + dz * dz);
    int weight = static_cast<int>(C * std::exp(-rgbDistance / SIGMA_RGB)); 
    return std::clamp(weight, 1, 300);
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
            
            // 增加位置距离权重
            auto calcWeight = [&](int y2, int x2) -> int {
                Vec3b color2 = image.at<Vec3b>(y2, x2);
                double spatialDist = std::sqrt((y-y2)*(y-y2) + (x-x2)*(x-x2));
                double colorWeight = Weight(pf.colorRGB, color2);
                return static_cast<int>(colorWeight * std::exp(-spatialDist * ALPHA));
            };

            if (y > 0) Matrix[y][x][0] = calcWeight(y-1, x);
            if (y < image.rows - 1) Matrix[y][x][1] = calcWeight(y+1, x);
            if (x > 0) Matrix[y][x][2] = calcWeight(y, x-1);
            if (x < image.cols - 1) Matrix[y][x][3] = calcWeight(y, x+1);
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
        adj[u].push_back({v, cap});
        adj[v].push_back({u, cap}); // 改为双向边，权重相同
        capacity[u][v] = cap;
        capacity[v][u] = cap; // 添加反向边容量
    }

    // bool bfs(int source, vector<int>& level) {
    //     fill(level.begin(), level.end(), -1);
    //     queue<int> q;
    //     q.push(source);
    //     level[source] = 0;

    //     while (!q.empty()) {
    //         int u = q.front();
    //         q.pop();

    //         for (auto& edge : adj[u]) {
    //             int v = edge.first;
    //             if (level[v] == -1 && capacity[u][v] - flow[u][v] > 0) {
    //                 level[v] = level[u] + 1;
    //                 q.push(v);
    //             }
    //         }
    //     }
    //     return level[V - 1] != -1;
    // }

    // int dfs(int u, int sink, vector<int>& level, vector<int>& ptr, int flowToPush) {
    //     if (u == sink) return flowToPush;
        
    //     for (int& i = ptr[u]; i < adj[u].size(); ++i) {
    //         int v = adj[u][i].first;
    //         if (level[v] == level[u] + 1 && capacity[u][v] - flow[u][v] > 0) {
    //             int availableFlow = min(flowToPush, capacity[u][v] - flow[u][v]);
    //             int pushedFlow = dfs(v, sink, level, ptr, availableFlow);
    //             if (pushedFlow > 0) {
    //                 flow[u][v] += pushedFlow;
    //                 flow[v][u] -= pushedFlow;
    //                 return pushedFlow;
    //             }
    //         }
    //     }
    //     return 0;
    // }
    void Push_Relabel (int source, int sink, vector<int>& S, vector<int>& T) {
    vector<int> height(V, 0);
    vector<int> excess(V, 0);
    vector<bool> active(V, false);
    queue<int> q;

    height[source] = V;
    for (auto& edge : adj[source]) {
        int v = edge.first;
        int capacity_val = capacity[source][v];
        flow[source][v] = capacity_val;
        flow[v][source] = -capacity_val;
        excess[v] = capacity_val;
        excess[source] -= capacity_val;

        if (v != sink && excess[v] > 0) {
            active[v] = true;
            q.push(v);
        }
    }

    // Push-Relabel main algorithm
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        active[u] = false;

        // Try to push excess flow to neighbors
        for (auto& edge : adj[u]) {
            int v = edge.first;
            
            // Can only push if there's excess and capacity allows
            if (excess[u] > 0 && capacity[u][v] - flow[u][v] > 0) {
                // Determine how much can be pushed
                int push_amount = min(excess[u], capacity[u][v] - flow[u][v]);

                // If destination height is less, can push
                if (height[u] > height[v]) {
                    flow[u][v] += push_amount;
                    flow[v][u] -= push_amount;
                    excess[u] -= push_amount;
                    excess[v] += push_amount;

                    // Activate destination if it has excess and isn't source/sink
                    if (v != source && v != sink && !active[v]) {
                        active[v] = true;
                        q.push(v);
                    }
                }
            }
        }

        // Relabel if no push is possible
        if (excess[u] > 0) {
            // Find minimum height of adjacent vertices with residual capacity
            int min_height = INT_MAX;
            for (auto& edge : adj[u]) {
                int v = edge.first;
                if (capacity[u][v] - flow[u][v] > 0) {
                    min_height = min(min_height, height[v]);
                }
            }

            // Relabel if a valid height is found
            if (min_height < INT_MAX) {
                height[u] = min_height + 1;
                active[u] = true;
                q.push(u);
            }
        }
    }

    // Determine S and T sets via BFS from source
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

    // Add unreachable nodes to T
    for (int i = 0; i < V; i++) {
        if (!visited[i]) {
            T.push_back(i);
        }
    }

    cout << "Total Pixels " << S.size() + T.size() << " nodes." << endl;
    cout << "Set S contains " << S.size() << " nodes." << endl;
    cout << "Set T contains " << T.size() << " nodes." << endl;
    }
    
};

int main() {
    Mat image = imread("Pictures/200*240.png");

    if (image.empty()) { 

            cout << "Fail to read image!" << endl;
            return -1;  
        } 
        
        int rows = image.rows;
        int cols = image.cols;
        auto start = chrono::high_resolution_clock::now();
        int source = (rows / 2) * cols + (cols / 2);  
        int sink = rows * cols - 2;
        
        Vec3b sourceColor = image.at<Vec3b>(source / cols, source % cols);
        Vec3b sinkColor = image.at<Vec3b>(sink / cols, sink % cols);
        
        auto Matrix = Build_Matrix(image);
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
        graph.Push_Relabel (source, sink, S, T);
        
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> duration = end - start;
        cout << "Time: " << duration.count() << " seconds" << endl;
        cout << "Push_Relabel  Speed: " << (S.size()+T.size()) /duration.count()<< " nodes/second" << endl;
        Mat result = image.clone();
        Mat s_result = Mat::zeros(image.size(), image.type());
        Mat t_result = Mat::zeros(image.size(), image.type());
        for (int i = 0; i < rows * cols; ++i) {
            int y = i / cols;
            int x = i % cols;
            if (find(S.begin(), S.end(), i) != S.end()) {
                result.at<Vec3b>(y, x) = sourceColor; 
                // s_result.at<Vec3b>(y, x) = image.at<Vec3b>(y, x);
            } else {
                result.at<Vec3b>(y, x) = sinkColor;   
                // t_result.at<Vec3b>(y, x) = image.at<Vec3b>(y, x);
            }
        }
        circle(result, Point(source % cols, source / cols), 1, Scalar(0, 255, 0), -1);
        circle(result, Point(sink % cols, sink / cols), 1 , Scalar(0, 0, 255), -1);
        imwrite("segmentation_result.png", result);
        // imwrite("Foreground_result.png", s_result);
        // imwrite("Background_result.png", t_result);
        return 0;
}