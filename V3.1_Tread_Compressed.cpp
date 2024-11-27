#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <queue>
#include <random>
#include <climits>
#include <thread>
#include <mutex>
#include <future>

using namespace std;
using namespace cv;

const int C = 100;
const double SIGMA_RGB = 10.0;
const double ALPHA = 0.01;
const double CONVERGENCE_THRESHOLD = 0.01;
const int NUM_THREADS = thread::hardware_concurrency();

struct KSegmentCompressedImage {
    Mat compressed;
    int originalRows;
    int originalCols;
    vector<vector<Vec3b>> blockColors;
    vector<Point> seedPoints;
    vector<Vec3b> segmentColors;
};

int determineBlockSize(int totalPixels) {
    if (totalPixels < 90000) {
        return 1;
    } else if (totalPixels < 800000) {
        return 2;
    }  else {
        return 3;
    }
}

// 图像压缩函数
KSegmentCompressedImage compressImage(const Mat& originalImage, int blockSize) {
    KSegmentCompressedImage result;
    result.originalRows = originalImage.rows;
    result.originalCols = originalImage.cols;
    
    int compressedRows = (originalImage.rows + blockSize - 1) / blockSize;
    int compressedCols = (originalImage.cols + blockSize - 1) / blockSize;
    
    result.compressed = Mat(compressedRows, compressedCols, originalImage.type());
    result.blockColors.resize(compressedRows, vector<Vec3b>(compressedCols));
    
    vector<thread> threads;
    int rowsPerThread = compressedRows / (NUM_THREADS);
    
    auto processRows = [&](int startRow, int endRow) {
        for (int i = startRow; i < endRow; i++) {
            for (int j = 0; j < compressedCols; j++) {
                Vec3d avgColor(0, 0, 0);
                int count = 0;
                
                for (int bi = 0; bi < blockSize && i * blockSize + bi < originalImage.rows; bi++) {
                    for (int bj = 0; bj < blockSize && j * blockSize + bj < originalImage.cols; bj++) {
                        avgColor += Vec3d(originalImage.at<Vec3b>(i * blockSize + bi, j * blockSize + bj));
                        count++;
                    }
                }
                
                avgColor /= count;
                Vec3b finalColor(avgColor[0], avgColor[1], avgColor[2]);
                result.compressed.at<Vec3b>(i, j) = finalColor;
                result.blockColors[i][j] = finalColor;
            }
        }
    };
    
    for (int i = 0; i < NUM_THREADS; ++i) {
        int startRow = i * rowsPerThread;
        int endRow = (i == NUM_THREADS - 1) ? compressedRows : (i + 1) * rowsPerThread;
        threads.emplace_back(processRows, startRow, endRow);
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    return result;
}

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

void BuildProbabilisticMatrixChunk(const Mat& image,const vector<Point>& seedPoints,vector<vector<vector<double>>>& probMatrix,int startRow,int endRow) {
    int cols = image.cols;
    int k = seedPoints.size();
    
    for (int y = startRow; y < endRow; ++y) {
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
            
            if (sumWeights > 0) {
                for (int s = 0; s < k; ++s) {
                    probMatrix[y][x][s] /= sumWeights;
                }
            }
        }
    }
}

vector<vector<vector<double>>> BuildProbabilisticMatrixParallel(const Mat& image,const vector<Point>& seedPoints) {
    int rows = image.rows;
    int cols = image.cols;
    int k = seedPoints.size();
    
    vector<vector<vector<double>>> probMatrix(rows,
        vector<vector<double>>(cols, vector<double>(k, 0.0)));
    
    vector<thread> threads;
    int rowsPerThread = rows / NUM_THREADS;
    
    for (int i = 0; i < NUM_THREADS; ++i) {
        int startRow = i * rowsPerThread;
        int endRow = (i == NUM_THREADS - 1) ? rows : (i + 1) * rowsPerThread;
        
        threads.emplace_back(BuildProbabilisticMatrixChunk,ref(image),ref(seedPoints),ref(probMatrix),startRow,endRow);
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    return probMatrix;
}

// 并行计算段颜色
vector<Vec3b> calculateSegmentColorsParallel(const Mat& image,const vector<vector<vector<double>>>& probMatrix,int k) {
    vector<Vec3d> colorSums(k, Vec3d(0, 0, 0));
    vector<double> counts(k, 0.0);
    mutex mtx;
    vector<thread> threads;
    
    int rowsPerThread = image.rows / NUM_THREADS;
    
    auto processChunk = [&](int startRow, int endRow) {
        vector<Vec3d> localSums(k, Vec3d(0, 0, 0));
        vector<double> localCounts(k, 0.0);
        
        for (int y = startRow; y < endRow; ++y) {
            for (int x = 0; x < image.cols; ++x) {
                Vec3b pixelColor = image.at<Vec3b>(y, x);
                for (int s = 0; s < k; ++s) {
                    double prob = probMatrix[y][x][s];
                    localSums[s] += Vec3d(pixelColor) * prob;
                    localCounts[s] += prob;
                }
            }
        }
        
        lock_guard<mutex> lock(mtx);
        for (int s = 0; s < k; ++s) {
            colorSums[s] += localSums[s];
            counts[s] += localCounts[s];
        }
    };
    
    for (int i = 0; i < NUM_THREADS; ++i) {
        int startRow = i * rowsPerThread;
        int endRow = (i == NUM_THREADS - 1) ? image.rows : (i + 1) * rowsPerThread;
        threads.emplace_back(processChunk, startRow, endRow);
    }
    
    for (auto& thread : threads) {
        thread.join();
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

void EMSegmentationParallel(vector<vector<vector<double>>>& probMatrix,const Mat& image,
vector<Point>& seedPoints,vector<Vec3b>& segmentColors,int maxIterations = 10) {
    int rows = image.rows;
    int cols = image.cols;
    int k = seedPoints.size();
    
    for (int iter = 0; iter < maxIterations; ++iter) {
        vector<Vec3d> newCenters(k, Vec3d(0, 0, 0));
        vector<double> counts(k, 0.0);
        
        vector<future<pair<vector<Vec3d>, vector<double>>>> futures;
        int rowsPerThread = rows / NUM_THREADS;
        
        for (int i = 0; i < NUM_THREADS; ++i) {
            int startRow = i * rowsPerThread;
            int endRow = (i == NUM_THREADS - 1) ? rows : (i + 1) * rowsPerThread;
            
            futures.push_back(async(launch::async, [&](int start, int end) {
                vector<Vec3d> localCenters(k, Vec3d(0, 0, 0));
                vector<double> localCounts(k, 0.0);
                
                for (int y = start; y < end; ++y) {
                    for (int x = 0; x < cols; ++x) {
                        Vec3b pixelColor = image.at<Vec3b>(y, x);
                        for (int s = 0; s < k; ++s) {
                            double prob = probMatrix[y][x][s];
                            localCenters[s] += Vec3d(pixelColor) * prob;
                            localCounts[s] += prob;
                        }
                    }
                }
                
                return make_pair(localCenters, localCounts);
            }, startRow, endRow));
        }
        
        for (auto& future : futures) {
            auto result = future.get();
            for (int s = 0; s < k; ++s) {
                newCenters[s] += result.first[s];
                counts[s] += result.second[s];
            }
        }
        
        vector<Point> newSeedPoints(k);
        vector<future<Point>> seedFutures;
        
        for (int s = 0; s < k; ++s) {
            if (counts[s] > 0) {
                newCenters[s] /= counts[s];
                
                seedFutures.push_back(async(launch::async, [&](int segment) {
                    double minDist = DBL_MAX;
                    Point bestPoint;
                    Vec3d targetCenter = newCenters[segment];
                    
                    for (int y = 0; y < rows; ++y) {
                        for (int x = 0; x < cols; ++x) {
                            Vec3b pixelColor = image.at<Vec3b>(y, x);
                            double dist = norm(Vec3d(pixelColor) - targetCenter);
                            if (dist < minDist) {
                                minDist = dist;
                                bestPoint = Point(x, y);
                            }
                        }
                    }
                    return bestPoint;
                }, s));
            }
        }
        
        for (int s = 0; s < k; ++s) {
            if (counts[s] > 0) {
                seedPoints[s] = seedFutures[s].get();
            }
        }
        
        auto newProbMatrix = BuildProbabilisticMatrixParallel(image, seedPoints);
        segmentColors = calculateSegmentColorsParallel(image, newProbMatrix, k);
        
        probMatrix =std:: move(newProbMatrix);
    }
}

// 继续generateSegmentationResult函数
Mat generateSegmentationResult(const Mat& original,const vector<vector<vector<double>>>& probMatrix,const vector<Vec3b>& segmentColors) {
    Mat result(original.size(), original.type());
    vector<thread> threads;
    int rowsPerThread = original.rows / NUM_THREADS;
    
    auto processChunk = [&](int startRow, int endRow) {
        vector<vector<int>> localSegmentAssignments(endRow - startRow, 
            vector<int>(original.cols, 0));
            
        // 为每个像素分配最可能的段
        for (int y = startRow; y < endRow; ++y) {
            for (int x = 0; x < original.cols; ++x) {
                int maxSegment = 0;
                double maxProb = probMatrix[y][x][0];
                
                for (int s = 1; s < segmentColors.size(); ++s) {
                    if (probMatrix[y][x][s] > maxProb) {
                        maxProb = probMatrix[y][x][s];
                        maxSegment = s;
                    }
                }
                
                localSegmentAssignments[y - startRow][x] = maxSegment;
            }
        }
        
        // 计算每个段的平均颜色
        vector<Vec3d> localColorSums(segmentColors.size(), Vec3d(0, 0, 0));
        vector<int> localCounts(segmentColors.size(), 0);
        
        for (int y = startRow; y < endRow; ++y) {
            for (int x = 0; x < original.cols; ++x) {
                int segment = localSegmentAssignments[y - startRow][x];
                Vec3b originalColor = original.at<Vec3b>(y, x);
                localColorSums[segment] += Vec3d(originalColor);
                localCounts[segment]++;
            }
        }
        
        // 应用计算出的颜色
        for (int y = startRow; y < endRow; ++y) {
            for (int x = 0; x < original.cols; ++x) {
                int segment = localSegmentAssignments[y - startRow][x];
                if (localCounts[segment] > 0) {
                    Vec3d avgColor = localColorSums[segment] / localCounts[segment];
                    result.at<Vec3b>(y, x) = Vec3b(
                        saturate_cast<uchar>(avgColor[0]),
                        saturate_cast<uchar>(avgColor[1]),
                        saturate_cast<uchar>(avgColor[2])
                    );
                }
            }
        }
    };
    
    for (int i = 0; i < NUM_THREADS; ++i) {
        int startRow = i * rowsPerThread;
        int endRow = (i == NUM_THREADS - 1) ? original.rows : (i + 1) * rowsPerThread;
        threads.emplace_back(processChunk, startRow, endRow);
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    return result;
}

int main() {
    Mat image = imread("Pictures/1600*900.png");
    if (image.empty()) {
        cout << "Wrong Image Name, Please Reload!" << endl;
        return -1;
    }
    
    int totalPixels = image.rows * image.cols;
    cout << "Image size: " << image.cols << "x" << image.rows << endl;
    cout << "Total pixels: " << totalPixels << endl;
    int BLOCK_SIZE = determineBlockSize(totalPixels);
    
    auto start = chrono::high_resolution_clock::now();
    KSegmentCompressedImage compressedImage = compressImage(image, BLOCK_SIZE);
    
    
    const int k = 6;
    vector<Point> seedPoints = initializeSeedPoints(compressedImage.compressed, k);
    auto probMatrix = BuildProbabilisticMatrixParallel(compressedImage.compressed, seedPoints);
    vector<Vec3b> segmentColors = calculateSegmentColorsParallel(compressedImage.compressed, probMatrix, k);
    EMSegmentationParallel(probMatrix, compressedImage.compressed, seedPoints, segmentColors);
    
    Mat compressedResult = generateSegmentationResult(compressedImage.compressed,probMatrix, segmentColors);
    Mat result;
    resize(compressedResult, result, image.size(), 0, 0, INTER_NEAREST);
    
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;
    double processingTime = duration.count();
    double pixelsPerSecond = totalPixels / processingTime;
    
    cout << "Time: " << processingTime << " seconds" << endl;
    cout << "Speed: " << static_cast<int>(pixelsPerSecond) << " pixels/second" << endl;
    for (int i = 0; i < k; ++i) {
        Point adjustedSeed(
            seedPoints[i].x * image.cols / compressedImage.compressed.cols,
            seedPoints[i].y * image.rows / compressedImage.compressed.rows
        );
        circle(result, seedPoints[i], 1, Scalar(0, 255, 0), -1);
        // cout << "Seed point " << i + 1 << ": (" << adjustedSeed.x << ", " << adjustedSeed.y << ")" << endl;
    }
    
    string performanceText = format("Time:%.3fs, Speed:%d px/s, Image Size:%dx%d, @copyright Jsy&Whn", processingTime, static_cast<int>(pixelsPerSecond),image.cols,image.rows );
    putText(result, performanceText, 
            Point(10, result.rows - 10), 
            FONT_HERSHEY_SIMPLEX, 
            1.0,
            Scalar(0, 115, 255), 
            2);
            
    imwrite("segmentation_result.png", result);
    
    return 0;
}