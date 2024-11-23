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

// Constants
const int BLOCK_SIZE = 2;
const int C = 100;             // Weight upper bound
const double SIGMA_RGB = 30.0;  // Standard deviation in RGB space
const double ALPHA = 0.01;      // Spatial distance decay factor

// Structure for compressed image
struct CompressedImage {
    Mat compressed;  // Compressed image
    int originalRows;  // Original image height
    int originalCols;  // Original image width
    vector<vector<Vec3b>> blockColors;  // Original colors of each pixel
};

// Structure for pixel features
struct PixelFeature {
    Vec3b colorRGB;  // RGB color value
    Point position;   // Pixel position
};

// Calculate weight between two colors
int Weight(const Vec3b& color1, const Vec3b& color2) {
    int dx = color1[0] - color2[0];
    int dy = color1[1] - color2[1];
    int dz = color1[2] - color2[2];
    double rgbDistance = std::sqrt(dx * dx + dy * dy + dz * dz);
    int weight = static_cast<int>(C * std::exp(-rgbDistance / SIGMA_RGB)); 
    return std::clamp(weight, 1, 100);
}

// Graph class for Dinic's algorithm
class Graph {
public:
    int V;  // Number of vertices
    vector<vector<pair<int, int>>> adj;  // Adjacency list
    vector<vector<int>> capacity;  // Capacity matrix
    vector<vector<int>> flow;  // Flow matrix

    Graph(int V) : V(V) {
        adj.resize(V);
        capacity = vector<vector<int>>(V, vector<int>(V, 0));
        flow = vector<vector<int>>(V, vector<int>(V, 0));
    }

    // Add edge to the graph
    void addEdge(int u, int v, int cap) {
        adj[u].push_back({v, cap});
        adj[v].push_back({u, cap}); 
        capacity[u][v] = cap;
        capacity[v][u] = cap; 
    }

    // BFS for level graph construction
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
        return level[V - 1] != -1;
    }

    // DFS for finding blocking flow
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

    // Dinic's algorithm implementation
    void dinic(int source, int sink, vector<int>& S, vector<int>& T) {
        int maxFlow = 0;
        vector<int> level(V);
        vector<int> ptr(V);

        const int SOURCE_WEIGHT = 200;  
        const int SINK_WEIGHT = 200;    

        // Set source capacities
        for (int i = 0; i < V; i++) {
            if (abs(i - source) < V/4) {  
                capacity[source][i] = SOURCE_WEIGHT;
            }
        }

        // Set sink capacities
        for (int i = 0; i < V; i++) {
            if (abs(i - sink) < V/4) {    
                capacity[i][sink] = SINK_WEIGHT;
            }
        }

        // Find maximum flow
        while (bfs(source, level)) {
            fill(ptr.begin(), ptr.end(), 0);
            while (int pushed = dfs(source, sink, level, ptr, INT_MAX)) {
                maxFlow += pushed;
            }
        }

        // Find min cut
        vector<bool> visited(V, false);
        queue<int> q;
        q.push(source);
        visited[source] = true;

        while (!q.empty()) {
            int u = q.front();
            q.pop();

            bool hasStrongConnection = false;
            for (auto& edge : adj[u]) {
                int v = edge.first;
                if (!visited[v] && capacity[u][v] - flow[u][v] > 0) {
                    if (capacity[u][v] > C/2) { 
                        hasStrongConnection = true;
                        visited[v] = true;
                        q.push(v);
                    }
                }
            }

            // Assign vertices to S or T set
            if (hasStrongConnection) {
                S.push_back(u);
            } else {
                T.push_back(u);
            }
        }

        // Add unvisited vertices to T set
        for (int i = 0; i < V; i++) {
            if (!visited[i]) {
                T.push_back(i);
            }
        }
    }
};

// Build adjacency matrix for the graph
std::vector<std::vector<std::vector<int>>> Build_Matrix(const Mat& image) {
    std::vector<std::vector<PixelFeature>> features(image.rows, std::vector<PixelFeature>(image.cols));
    std::vector<std::vector<std::vector<int>>> Matrix(image.rows, std::vector<std::vector<int>>(image.cols, std::vector<int>(4, 0)));

    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            PixelFeature pf;
            pf.colorRGB = image.at<Vec3b>(y, x);
            pf.position = Point(x, y);
            features[y][x] = pf;
            
            // Lambda function to calculate weight between pixels
            auto calcWeight = [&](int y2, int x2) -> int {
                Vec3b color2 = image.at<Vec3b>(y2, x2);
                double spatialDist = std::sqrt((y-y2)*(y-y2) + (x-x2)*(x-x2));
                double colorWeight = Weight(pf.colorRGB, color2);
                return static_cast<int>(colorWeight * std::exp(-spatialDist * ALPHA));
            };

            // Calculate weights for adjacent pixels
            if (y > 0) Matrix[y][x][0] = calcWeight(y-1, x);
            if (y < image.rows - 1) Matrix[y][x][1] = calcWeight(y+1, x);
            if (x > 0) Matrix[y][x][2] = calcWeight(y, x-1);
            if (x < image.cols - 1) Matrix[y][x][3] = calcWeight(y, x+1);
        }
    }
    return Matrix;
}

// Compress input image
CompressedImage compressImage(const Mat& original) {
    CompressedImage result;
    result.originalRows = original.rows;
    result.originalCols = original.cols;
    
    // Calculate compressed dimensions
    int newRows = (original.rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int newCols = (original.cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    result.compressed = Mat(newRows, newCols, original.type());
    result.blockColors.resize(original.rows, vector<Vec3b>(original.cols));
    
    // Process each block
    for (int y = 0; y < newRows; ++y) {
        for (int x = 0; x < newCols; ++x) {
            int count = 0;
            Vec3d sum(0, 0, 0);
            
            // Calculate average color for the block
            for (int by = y * BLOCK_SIZE; by < min((y + 1) * BLOCK_SIZE, original.rows); ++by) {
                for (int bx = x * BLOCK_SIZE; bx < min((x + 1) * BLOCK_SIZE, original.cols); ++bx) {
                    Vec3b pixel = original.at<Vec3b>(by, bx);
                    sum += Vec3d(pixel[0], pixel[1], pixel[2]);
                    result.blockColors[by][bx] = pixel;
                    count++;
                }
            }
            
            // Set compressed pixel value
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

// Restore segmentation result to original size
Mat restoreSegmentation(const CompressedImage& compressed, const vector<int>& S, int compressedCols) {
    Mat result = Mat(compressed.originalRows, compressed.originalCols, CV_8UC3);
    
    // Process each pixel in original size
    for (int y = 0; y < compressed.originalRows; ++y) {
        for (int x = 0; x < compressed.originalCols; ++x) {
            // Calculate block index
            int blockY = y / BLOCK_SIZE;
            int blockX = x / BLOCK_SIZE;
            int blockIndex = blockY * compressedCols + blockX;
            
            // Set pixel color based on segmentation
            if (find(S.begin(), S.end(), blockIndex) != S.end()) {
                result.at<Vec3b>(y, x) = compressed.blockColors[y][x];  // Foreground: use original color
            } else {
                result.at<Vec3b>(y, x) = Vec3b(0, 0, 0);  // Background: set to black
            }
        }
    }
    
    return result;
}

int main() {
    // Read input image
    Mat image = imread("Image-Foreground-Background-Segmentation-using-Network-Flow-main/Pictures/220*220.png");
    if (image.empty()) {
        cout << "Failed to read image!" << endl;
        return -1;
    }
    
    // 1. Compress image
    auto compressedImg = compressImage(image);
    cout << "Original size: " << image.rows << "x" << image.cols << endl;
    cout << "Compressed size: " << compressedImg.compressed.rows << "x" << compressedImg.compressed.cols << endl;
    
    // Save the compressed image before segmentation
    Mat resizedCompressed;
    resize(compressedImg.compressed, 
           resizedCompressed, 
           Size(compressedImg.originalCols, compressedImg.originalRows),
           0, 0, INTER_NEAREST);  // Use nearest neighbor to maintain block effect
    imwrite("compressed_original.png", resizedCompressed);
    
    // 2. Run Dinic's algorithm on compressed image
    int rows = compressedImg.compressed.rows;
    int cols = compressedImg.compressed.cols;
    auto start = chrono::high_resolution_clock::now();
    
    // Select source and sink points
    int source = (rows / 2) * cols + (cols / 2);
    int sink = rows * cols - 2;
    
    // Build graph
    auto Matrix = Build_Matrix(compressedImg.compressed);
    Graph graph(rows * cols);
    
    // Add edges
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
    
    // Run segmentation
    vector<int> S, T;
    graph.dinic(source, sink, S, T);
    
    // Calculate execution time
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;
    cout << "Time: " << duration.count() << " seconds" << endl;
    
    // 3. Restore to original size and save result
    Mat segmentationResult = restoreSegmentation(compressedImg, S, cols);
    
    // Mark source and sink points
    circle(segmentationResult, Point(source % cols * BLOCK_SIZE, source / cols * BLOCK_SIZE), 
           BLOCK_SIZE, Scalar(0, 255, 0), -1);  // Green for source
    circle(segmentationResult, Point(sink % cols * BLOCK_SIZE, sink / cols * BLOCK_SIZE), 
           BLOCK_SIZE, Scalar(0, 0, 255), -1);  // Red for sink
    
    // Save both results
    imwrite("compressed_segmentation_result.png", segmentationResult);
    
    cout << "Saved compressed original image as 'compressed_original.png'" << endl;
    cout << "Saved segmentation result as 'compressed_segmentation_result.png'" << endl;
    
    return 0;
}
