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

int determineBlockSize(int totalPixels) {
    if (totalPixels < 90000) {
        return 2;
    } else if (totalPixels < 250000) {
        return 4;
    } else if (totalPixels < 400000) {
        return 6;
    } 
    else if (totalPixels < 550000) {
        return 8;
    } 
    else if (totalPixels < 700000) {
        return 10;
    } 
    else {
        return 12;
    }
}

struct KSegmentCompressedImage {
    Mat compressed;
    int originalRows;
    int originalCols;
    vector<vector<Vec3b>> blockColors;
    vector<Point> seedPoints;  
    vector<Vec3b> segmentColors;  
};

struct PixelFeature {
    Vec3b colorRGB;
    Point position;
    int segment;  
    vector<double> probabilities;  
};

const int C = 100;
const double SIGMA_RGB = 10.0;
const double ALPHA = 0.01;
const double CONVERGENCE_THRESHOLD = 0.01;
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
vector<Point> initializeSeedPoints(const Mat& image, int k) {
    vector<Point> seedPoints;
    vector<double> minDistances(image.rows * image.cols, DBL_MAX);
    

    Point firstSeed(image.cols / 2, image.rows / 2);
    seedPoints.push_back(firstSeed);
    

    for (int i = 1; i < k; i++) {
        double totalDistance = 0.0;
        for (int y = 0; y < image.rows; y++) {
            for (int x = 0; x < image.cols; x++) {
                Vec3b currentColor = image.at<Vec3b>(y, x);
                double minDist = DBL_MAX;
                
                for (const Point& seed : seedPoints) {
                    Vec3b seedColor = image.at<Vec3b>(seed.y, seed.x);
                    double dist = norm(Vec3d(currentColor) - Vec3d(seedColor));
                    minDist = min(minDist, dist);
                }
                
                minDistances[y * image.cols + x] = minDist;
                totalDistance += minDist;
            }
        }
        
        double threshold = totalDistance / 2;
        double sum = 0.0;
        Point newSeed;
        for (int y = 0; y < image.rows; y++) {
            for (int x = 0; x < image.cols; x++) {
                sum += minDistances[y * image.cols + x];
                if (sum >= threshold) {
                    newSeed = Point(x, y);
                    break;
                }
            }
            if (sum >= threshold) break;
        }
        seedPoints.push_back(newSeed);
    }
    
    return seedPoints;
}
vector<vector<vector<double>>> BuildProbabilisticMatrix(const Mat& image, const vector<Point>& seedPoints) {
    int rows = image.rows;
    int cols = image.cols;
    int k = seedPoints.size();
    
    vector<vector<vector<double>>> probMatrix(rows, 
        vector<vector<double>>(cols, vector<double>(k, 0.0)));
    
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

vector<Vec3b> calculateSegmentColors(const Mat& image, const vector<vector<vector<double>>>& probMatrix,int k) {
    vector<Vec3d> colorSums(k, Vec3d(0, 0, 0));
    vector<double> counts(k, 0.0);
    
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            Vec3b pixelColor = image.at<Vec3b>(y, x);
            for (int s = 0; s < k; ++s) {
                double prob = probMatrix[y][x][s];
                colorSums[s] += Vec3d(pixelColor) * prob;
                counts[s] += prob;
            }
        }
    }
    
    vector<Vec3b> segmentColors(k);
    for (int s = 0; s < k; ++s) {
        if (counts[s] > 0) {
            segmentColors[s] = Vec3b(
                saturate_cast<uchar>(colorSums[s][0] / counts[s]),
                saturate_cast<uchar>(colorSums[s][1] / counts[s]),
                saturate_cast<uchar>(colorSums[s][2] / counts[s])
            );
        }
    }
    
    return segmentColors;
}

void EMSegmentation(vector<vector<vector<double>>>& probMatrix, const Mat& image, vector<Point>& seedPoints, vector<Vec3b>& segmentColors,int maxIterations = 10) {
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
        
        auto newProbMatrix = BuildProbabilisticMatrix(image, seedPoints);
        segmentColors = calculateSegmentColors(image, newProbMatrix, k);
        double maxDiff = 0.0;
        for (int y = 0; y < rows; ++y) {
            for (int x = 0; x < cols; ++x) {
                for (int s = 0; s < k; ++s) {
                    maxDiff = max(maxDiff, 
                        abs(newProbMatrix[y][x][s] - probMatrix[y][x][s]));
                }
            }
        }
        
        probMatrix =std:: move(newProbMatrix);
        
        // if (maxDiff < CONVERGENCE_THRESHOLD) {
        //     cout << "Converged after " << iter + 1 << " iterations" << endl;
        //     break;
        // }
    }
}
Mat generateSegmentationResult(const Mat& original, const vector<vector<vector<double>>>& probMatrix,const vector<Vec3b>& segmentColors) {
    Mat result(original.size(), original.type());
    vector<vector<int>> segmentAssignments(original.rows, vector<int>(original.cols, 0));
    
    // Assign each pixel to its most probable segment
    for (int y = 0; y < original.rows; ++y) {
        for (int x = 0; x < original.cols; ++x) {
            int maxSegment = 0;
            double maxProb = probMatrix[y][x][0];
            
            for (int s = 1; s < segmentColors.size(); ++s) {
                if (probMatrix[y][x][s] > maxProb) {
                    maxProb = probMatrix[y][x][s];
                    maxSegment = s;
                }
            }
            
            segmentAssignments[y][x] = maxSegment;
        }
    }

    vector<Vec3d> segmentColorSums(segmentColors.size(), Vec3d(0, 0, 0));
    vector<int> segmentPixelCounts(segmentColors.size(), 0);
    
    for (int y = 0; y < original.rows; ++y) {
        for (int x = 0; x < original.cols; ++x) {
            int segment = segmentAssignments[y][x];
            Vec3b originalColor = original.at<Vec3b>(y, x);
            segmentColorSums[segment] += Vec3d(originalColor);
            segmentPixelCounts[segment]++;
        }
    }
    
    vector<Vec3b> avgSegmentColors(segmentColors.size());
    for (size_t i = 0; i < segmentColors.size(); ++i) {
        if (segmentPixelCounts[i] > 0) {
            avgSegmentColors[i] = Vec3b(
                saturate_cast<uchar>(segmentColorSums[i][0] / segmentPixelCounts[i]),
                saturate_cast<uchar>(segmentColorSums[i][1] / segmentPixelCounts[i]),
                saturate_cast<uchar>(segmentColorSums[i][2] / segmentPixelCounts[i])
            );
        }
    }
    
    for (int y = 0; y < original.rows; ++y) {
        for (int x = 0; x < original.cols; ++x) {
            int segment = segmentAssignments[y][x];
            result.at<Vec3b>(y, x) = avgSegmentColors[segment];
        }
    }
    
    return result;
}

int main() {
    Mat image = imread("Pictures/800*500.png");
    if (image.empty()) {
        cout << "Failed to read image!" << endl;
        return -1;
    }
    int totalPixels = image.rows * image.cols;
    cout << "Image size: " << image.cols << "x" << image.rows << endl;
    cout << "Total pixels: " << totalPixels << endl;
    const int k = 3; 
    int BLOCK_SIZE = determineBlockSize(totalPixels);
    vector<Point> seedPoints = initializeSeedPoints(image, k);
    RNG rng(time(nullptr));
    for (int i = 0; i < k; ++i) {
        Point seed(rng.uniform(0, image.cols), rng.uniform(0, image.rows));
        seedPoints.push_back(seed);
    }
    
    auto start = chrono::high_resolution_clock::now();
    
    auto probMatrix = BuildProbabilisticMatrix(image, seedPoints);
    vector<Vec3b> segmentColors = calculateSegmentColors(image, probMatrix, k);
    EMSegmentation(probMatrix, image, seedPoints, segmentColors);
    Mat result = generateSegmentationResult(image, probMatrix, segmentColors);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;
    double processingTime = duration.count();
    double pixelsPerSecond = totalPixels / processingTime;
    cout << "Time: " << processingTime << " seconds" << endl;
    cout << "Speed: " << static_cast<int>(pixelsPerSecond) << " pixels/second" << endl;
    cout << "Number of segments: " << k << endl;
    
    for (int i = 0; i < k; ++i) {
        circle(result, seedPoints[i], 1, Scalar(0, 255, 0), -1);
        cout << "Seed point " << i + 1 << ": (" << seedPoints[i].x << ", " << seedPoints[i].y << ")" << endl;
    }
    
    string performanceText = format("Time: %.2fs, Speed: %d px/s @copyright Jsy&Whn", processingTime, static_cast<int>(pixelsPerSecond));
    putText(result, performanceText, 
            Point(10, result.rows - 10), 
            FONT_HERSHEY_SIMPLEX, 
            0.5,
            Scalar(0, 115, 255), 
            1); 
    imwrite("segmentation_result.png", result);
    
    return 0;
}
