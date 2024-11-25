#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <queue>
#include <random>
#include <climits>

using namespace std;
using namespace cv;

// Enhanced structure for k-way segmentation
struct KSegmentCompressedImage {
    Mat compressed;
    int originalRows;
    int originalCols;
    vector<vector<Vec3b>> blockColors;
    vector<Point> seedPoints;  // Store k seed points
    vector<Vec3b> segmentColors;  // Store colors for k segments
};

struct PixelFeature {
    Vec3b colorRGB;
    Point position;
    int segment;  // Current segment assignment
    vector<double> probabilities;  // Probability distribution over segments
};

const int C = 100;
const double SIGMA_RGB = 10.0;
const double ALPHA = 0.1 ;
const int BLOCK_SIZE = 2;
const double CONVERGENCE_THRESHOLD = 0.01;

// Enhanced weight calculation for k-way segmentation
double WeightK(const Vec3b& color1, const Vec3b& color2, const Point& pos1, const Point& pos2) {
    double dx = color1[0] - color2[0];
    double dy = color1[1] - color2[1];
    double dz = color1[2] - color2[2];
    double rgbDistance = std::sqrt(dx * dx + dy * dy + dz * dz);
    
    double spatialDx = pos1.x - pos2.x;
    double spatialDy = pos1.y - pos2.y;
    double spatialDist = std::sqrt(spatialDx * spatialDx + spatialDy * spatialDy);
    
    return C * std::exp(-rgbDistance / SIGMA_RGB - spatialDist * ALPHA);
}

// Build probabilistic adjacency matrix for k segments
vector<vector<vector<double>>> BuildProbabilisticMatrix(const Mat& image, const vector<Point>& seedPoints) {
    int rows = image.rows;
    int cols = image.cols;
    int k = seedPoints.size();
    
    vector<vector<vector<double>>> probMatrix(rows, 
        vector<vector<double>>(cols, vector<double>(k, 0.0)));
    
    // Initialize with Gaussian mixture model
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            Vec3b pixelColor = image.at<Vec3b>(y, x);
            Point currentPoint(x, y);
            
            double sumWeights = 0.0;
            for (int s = 0; s < k; ++s) {
                Vec3b seedColor = image.at<Vec3b>(seedPoints[s].y, seedPoints[s].x);
                double weight = WeightK(pixelColor, seedColor, currentPoint, seedPoints[s]);
                probMatrix[y][x][s] = weight;
                sumWeights += weight;
            }
            
            // Normalize probabilities
            if (sumWeights > 0) {
                for (int s = 0; s < k; ++s) {
                    probMatrix[y][x][s] /= sumWeights;
                }
            }
        }
    }
    
    return probMatrix;
}

// Expectation-Maximization for k-way segmentation
void EMSegmentation(vector<vector<vector<double>>>& probMatrix, const Mat& image, 
                   vector<Point>& seedPoints, int maxIterations = 10) {
    int rows = image.rows;
    int cols = image.cols;
    int k = seedPoints.size();
    
    for (int iter = 0; iter < maxIterations; ++iter) {
        // E-step: Update segment assignments
        vector<Vec3d> newCenters(k, Vec3d(0, 0, 0));
        vector<double> counts(k, 0.0);
        
        for (int y = 0; y < rows; ++y) {
            for (int x = 0; x < cols; ++x) {
                Vec3b pixelColor = image.at<Vec3b>(y, x);
                for (int s = 0; s < k; ++s) {
                    double prob = probMatrix[y][x][s];
                    newCenters[s] += Vec3d(pixelColor) * prob;
                    counts[s] += prob;
                }
            }
        }
        
        // M-step: Update segment centers
        for (int s = 0; s < k; ++s) {
            if (counts[s] > 0) {
                newCenters[s] /= counts[s];
                // Find closest pixel to new center for seed point
                double minDist = DBL_MAX;
                Point bestPoint;
                
                for (int y = 0; y < rows; ++y) {
                    for (int x = 0; x < cols; ++x) {
                        Vec3b pixelColor = image.at<Vec3b>(y, x);
                        double dist = norm(Vec3d(pixelColor) - newCenters[s]);
                        if (dist < minDist) {
                            minDist = dist;
                            bestPoint = Point(x, y);
                        }
                    }
                }
                seedPoints[s] = bestPoint;
            }
        }
        
        // Update probabilities
        auto newProbMatrix = BuildProbabilisticMatrix(image, seedPoints);
        
        // Check convergence
        double maxDiff = 0.0;
        for (int y = 0; y < rows; ++y) {
            for (int x = 0; x < cols; ++x) {
                for (int s = 0; s < k; ++s) {
                    maxDiff = max(maxDiff, 
                        abs(newProbMatrix[y][x][s] - probMatrix[y][x][s]));
                }
            }
        }
        
        probMatrix = std::move(newProbMatrix);
        
        if (maxDiff < CONVERGENCE_THRESHOLD) {
            cout << "Converged after " << iter + 1 << " iterations" << endl;
            break;
        }
    }
}

Mat generateSegmentationResult(const Mat& original, 
                             const vector<vector<vector<double>>>& probMatrix,
                             const vector<Vec3b>& segmentColors) {
    Mat result(original.size(), original.type());
    
    for (int y = 0; y < original.rows; ++y) {
        for (int x = 0; x < original.cols; ++x) {
            // Find segment with highest probability
            int maxSegment = 0;
            double maxProb = probMatrix[y][x][0];
            
            for (int s = 1; s < segmentColors.size(); ++s) {
                if (probMatrix[y][x][s] > maxProb) {
                    maxProb = probMatrix[y][x][s];
                    maxSegment = s;
                }
            }
            
            result.at<Vec3b>(y, x) = segmentColors[maxSegment];
        }
    }
    
    return result;
}

// Modified main function to handle k segments
int main() {
    Mat image = imread("Pictures/500*400.png");
    if (image.empty()) {
        cout << "Failed to read image!" << endl;
        return -1;
    }
    
    const int k = 4; // Number of segments
    
    // Initialize seed points randomly
    vector<Point> seedPoints;
    RNG rng(time(nullptr));
    for (int i = 0; i < k; ++i) {
        Point seed(rng.uniform(0, image.cols), rng.uniform(0, image.rows));
        seedPoints.push_back(seed);
    }
    
    // Initialize segment colors
    vector<Vec3b> segmentColors = {
        Vec3b(255, 0, 0),   // Red
        Vec3b(0, 255, 0),   // Green
        Vec3b(0, 0, 255),   // Blue
        Vec3b(255, 255, 0)  // Yellow
    };
    
    auto start = chrono::high_resolution_clock::now();
    
    // Build initial probability matrix
    auto probMatrix = BuildProbabilisticMatrix(image, seedPoints);
    
    // Run EM algorithm
    EMSegmentation(probMatrix, image, seedPoints);
    
    // Generate final segmentation
    Mat result = generateSegmentationResult(image, probMatrix, segmentColors);
    
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;
    
    cout << "Time: " << duration.count() << " seconds" << endl;
    
    // Mark seed points
    for (int i = 0; i < k; ++i) {
        circle(result, seedPoints[i], 5, Scalar(255, 255, 255), -1);
    }
    
    imwrite("segmentation_result.png", result);
    
    return 0;
}