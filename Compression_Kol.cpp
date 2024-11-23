#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <queue>
#include <climits>

using namespace std;
using namespace cv;

// Structure for compressed image data
struct CompressedImage {
    Mat compressed;       // Compressed image
    int originalRows;     // Original image height
    int originalCols;     // Original image width
    vector<vector<Vec3b>> blockColors;  // Original colors of each pixel
};

struct PixelFeature { 
    Vec3b colorRGB;  
    Point position;   
};  

const int C = 100;             
const double SIGMA_RGB = 8.0;   //bigger more accurate but slower
const double ALPHA = 0.2;       //smaller more accurate but slower
const int BLOCK_SIZE = 2;       // Compression block size

// Weight calculation function
int Weight(const Vec3b& color1, const Vec3b& color2) {
    int dx = color1[0] - color2[0];
    int dy = color1[1] - color2[1];
    int dz = color1[2] - color2[2];
    double rgbDistance = std::sqrt(dx * dx + dy * dy + dz * dz);
    int weight = static_cast<int>(C * std::exp(-rgbDistance / SIGMA_RGB)); 
    return weight;
}

// Build adjacency matrix
std::vector<std::vector<std::vector<int>>> Build_Matrix(const Mat& image) {
    std::vector<std::vector<PixelFeature>> features(image.rows, std::vector<PixelFeature>(image.cols));
    std::vector<std::vector<std::vector<int>>> Matrix(image.rows, std::vector<std::vector<int>>(image.cols, std::vector<int>(4, 0)));

    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            PixelFeature pf;
            pf.colorRGB = image.at<Vec3b>(y, x);
            pf.position = Point(x, y);
            features[y][x] = pf;
            
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
        adj[v].push_back({u, cap}); // 双向边
        capacity[u][v] = cap;
        capacity[v][u] = cap; // 反向边容量
    }

    void Kolmogorov(int source, int sink, vector<int>& S, vector<int>& T) {
        vector<int> level(V);
        vector<int> ptr(V);
        vector<int> flow_to(V);
        vector<bool> in_queue(V, false);
        vector<int> parent(V, -1);

        // Initialize flow
        fill(flow[0].begin(), flow[0].end(), 0);

        queue<int> q;
        while (true) {
            // BFS to find augmenting paths
            fill(level.begin(), level.end(), -1);
            fill(in_queue.begin(), in_queue.end(), false);
            fill(parent.begin(), parent.end(), -1);
            while (!q.empty()) q.pop();

            q.push(source);
            level[source] = 0;
            in_queue[source] = true;

            while (!q.empty()) {
                int u = q.front();
                q.pop();
                in_queue[u] = false;

                for (auto& edge : adj[u]) {
                    int v = edge.first;
                    int rem_cap = capacity[u][v] - flow[u][v];

                    if (rem_cap > 0 && level[v] == -1) {
                        level[v] = level[u] + 1;
                        parent[v] = u;
                        if (!in_queue[v]) {
                            q.push(v);
                            in_queue[v] = true;
                        }
                    }
                }
            }

            // No augmenting path found
            if (level[sink] == -1) break;

            // Find min cut in the graph
            fill(flow_to.begin(), flow_to.end(), INT_MAX);
            for (int v = sink; v != source; v = parent[v]) {
                int u = parent[v];
                flow_to[v] = min(flow_to[v], capacity[u][v] - flow[u][v]);
            }

            for (int v = sink; v != source; v = parent[v]) {
                int u = parent[v];
                flow[u][v] += flow_to[sink];
                flow[v][u] -= flow_to[sink];
            }
        }

        // Separate into S and T sets using BFS from source
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

        // Add unvisited nodes to T
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

// Compress image function
CompressedImage compressImage(const Mat& original) {
    CompressedImage result;
    result.originalRows = original.rows;
    result.originalCols = original.cols;
    
    int newRows = (original.rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int newCols = (original.cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    result.compressed = Mat(newRows, newCols, original.type());
    result.blockColors.resize(original.rows, vector<Vec3b>(original.cols));
    
    for (int y = 0; y < newRows; ++y) {
        for (int x = 0; x < newCols; ++x) {
            int count = 0;
            Vec3d sum(0, 0, 0);
            
            for (int by = y * BLOCK_SIZE; by < min((y + 1) * BLOCK_SIZE, original.rows); ++by) {
                for (int bx = x * BLOCK_SIZE; bx < min((x + 1) * BLOCK_SIZE, original.cols); ++bx) {
                    Vec3b pixel = original.at<Vec3b>(by, bx);
                    sum += Vec3d(pixel[0], pixel[1], pixel[2]);
                    result.blockColors[by][bx] = pixel;
                    count++;
                }
            }
            
            Vec3b avgColor(
                static_cast<uchar>(sum[0] / count),
                static_cast<uchar>(sum[1] / count),
                static_cast<uchar>(sum[2] / count)
            );
            result.compressed.at<Vec3b>(y, x) = avgColor;
        }
    }
    
    return result;
}

// Restore segmentation to original size while keeping original colors for foreground
Mat restoreSegmentation(const CompressedImage& compressed, const vector<int>& S, 
                       int compressedCols, const Vec3b& sourceColor, const Vec3b& sinkColor) {
    Mat result = Mat(compressed.originalRows, compressed.originalCols, CV_8UC3);
    
    for (int y = 0; y < compressed.originalRows; ++y) {
        for (int x = 0; x < compressed.originalCols; ++x) {
            int blockY = y / BLOCK_SIZE;
            int blockX = x / BLOCK_SIZE;
            int blockIndex = blockY * compressedCols + blockX;
            
            if (find(S.begin(), S.end(), blockIndex) != S.end()) {
                result.at<Vec3b>(y, x) = compressed.blockColors[y][x];  // Use original color for foreground
            } else {
                result.at<Vec3b>(y, x) = sinkColor;  // Use sink color for background
            }
        }
    }
    
    return result;
}

int main() {
    Mat image = imread("Image-Foreground-Background-Segmentation-using-Network-Flow-main/Pictures/220*250.png");

    if (image.empty()) { 
        cout << "Fail to read image!" << endl;
        return -1;  
    } 
    
    cout << "Original image size: " << image.rows << "x" << image.cols << endl;
    auto start = chrono::high_resolution_clock::now();

    // 1. Compress the image
    auto compressedImg = compressImage(image);
    cout << "Compressed image size: " << compressedImg.compressed.rows << "x" << compressedImg.compressed.cols << endl;
    
    // Save compressed image for comparison
    Mat resizedCompressed;
    resize(compressedImg.compressed, 
           resizedCompressed, 
           Size(compressedImg.originalCols, compressedImg.originalRows),
           0, 0, INTER_NEAREST);
    imwrite("compressed_original_Kol.png", resizedCompressed);
    
    // 2. Run Kolmogorov algorithm on compressed image
    int rows = compressedImg.compressed.rows;
    int cols = compressedImg.compressed.cols;
    
    int source = (rows / 2) * cols + (cols / 2);
    int sink = rows * cols - 2;
    
    Vec3b sourceColor = compressedImg.compressed.at<Vec3b>(source / cols, source % cols);
    Vec3b sinkColor = compressedImg.compressed.at<Vec3b>(sink / cols, sink % cols);
    
    auto Matrix = Build_Matrix(compressedImg.compressed);
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
    
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;
    cout << "Time: " << duration.count() << " seconds" << endl;
    cout << "Kolmogorov Speed (Compressed): " << (S.size()+T.size()) /duration.count() << " nodes/second" << endl;
    
    // 3. Restore segmentation to original size while keeping original colors for foreground
    Mat result = restoreSegmentation(compressedImg, S, cols, sourceColor, sinkColor);
    
    // Mark source and sink points
    circle(result, Point(source % cols * BLOCK_SIZE, source / cols * BLOCK_SIZE), 
           BLOCK_SIZE, Scalar(0, 255, 0), -1);  // Green for source
    circle(result, Point(sink % cols * BLOCK_SIZE, sink / cols * BLOCK_SIZE), 
           BLOCK_SIZE, Scalar(0, 0, 255), -1);  // Red for sink
    
    imwrite("compressed_segmentation_Kol_result.png", result);
    
    return 0;
}
