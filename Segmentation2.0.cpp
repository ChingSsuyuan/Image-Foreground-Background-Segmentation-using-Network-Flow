#include "Segmentation2.0.h"
#include <cmath>
#include <queue>
#include <climits>
#include <algorithm>
#include <iostream>

// Constants
const int C = 100;             
const double SIGMA_RGB = 8.0;   // bigger more accurate but slower
const double ALPHA = 0.2;       // smaller more accurate but slower

struct PixelFeature { 
    Vec3b colorRGB;  
    Point position;   
};

// Calculate weight between pixels
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
        adj[v].emplace_back(u, cap);
        capacity[u][v] = cap;
        capacity[v][u] = cap;
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

// Compress image function
CompressedImage compressImage(const Mat& original, int blockSize) {
    CompressedImage result;
    result.originalRows = original.rows;
    result.originalCols = original.cols;
    
    int newRows = (original.rows + blockSize - 1) / blockSize;
    int newCols = (original.cols + blockSize - 1) / blockSize;
    
    result.compressed = Mat(newRows, newCols, original.type());
    result.blockColors.resize(original.rows, vector<Vec3b>(original.cols));
    
    for (int y = 0; y < newRows; ++y) {
        for (int x = 0; x < newCols; ++x) {
            int count = 0;
            Vec3d sum(0, 0, 0);
            
            for (int by = y * blockSize; by < min((y + 1) * blockSize, original.rows); ++by) {
                for (int bx = x * blockSize; bx < min((x + 1) * blockSize, original.cols); ++bx) {
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

// Restore segmentation to original size
Mat restoreSegmentation(const CompressedImage& compressed, const vector<int>& S, int compressedCols) {
    Mat result = Mat(compressed.originalRows, compressed.originalCols, CV_8UC3);
    
    for (int y = 0; y < compressed.originalRows; ++y) {
        for (int x = 0; x < compressed.originalCols; ++x) {
            int blockY = y / 2;  // blockSize = 2
            int blockX = x / 2;
            int blockIndex = blockY * compressedCols + blockX;
            
            if (find(S.begin(), S.end(), blockIndex) != S.end()) {
                result.at<Vec3b>(y, x) = compressed.blockColors[y][x];  // Use original color
            } else {
                result.at<Vec3b>(y, x) = Vec3b(0, 0, 0);  // Background
            }
        }
    }
    
    return result;
}

// Original segmentation function (unchanged)
cv::Mat performSegmentation(const cv::Mat& inputImage) {
    int rows = inputImage.rows, cols = inputImage.cols;

    int source = (rows / 2) * cols + (cols / 2);  
    int sink = rows * cols - 2;
    Vec3b sourceColor = inputImage.at<Vec3b>(source / cols, source % cols);
    Vec3b sinkColor = inputImage.at<Vec3b>(sink / cols, sink % cols);

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

// New compressed segmentation function
cv::Mat performCompressedSegmentation(const cv::Mat& inputImage, int blockSize) {
    // 1. Compress image
    auto compressedImg = compressImage(inputImage, blockSize);
    
    // For visualization of compression effect
    Mat resizedCompressed;
    resize(compressedImg.compressed, 
           resizedCompressed, 
           Size(compressedImg.originalCols, compressedImg.originalRows),
           0, 0, INTER_NEAREST);
    imwrite("compressed_original.png", resizedCompressed);
    
    // 2. Run segmentation on compressed image
    int rows = compressedImg.compressed.rows;
    int cols = compressedImg.compressed.cols;
    
    int source = (rows / 2) * cols + (cols / 2);
    int sink = rows * cols - 2;
    
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
    
    // 3. Restore to original size while keeping original colors for foreground
    Mat result = restoreSegmentation(compressedImg, S, cols);
    
    // Mark source and sink points
    circle(result, Point(source % cols * blockSize, source / cols * blockSize), 
           blockSize, Scalar(0, 255, 0), -1);
    circle(result, Point(sink % cols * blockSize, sink / cols * blockSize), 
           blockSize, Scalar(0, 0, 255), -1);
    
    return result;
}
