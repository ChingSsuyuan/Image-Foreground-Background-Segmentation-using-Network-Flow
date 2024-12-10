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
const double SIGMA_RGB = 0.5;
const double ALPHA = 0.01;
const double CONVERGENCE_THRESHOLD = 0.1;

struct LKSPI {
    Mat compressed;
    int originalRows;
    int originalCols;
    vector<vector<Vec3b>> blockColors;
    vector<Point> seedPoints;  
    vector<Vec3b> segmentColors;  
};

int DBS(int totalPixels) {
    if (totalPixels < 90000) {
        return 1;
    } else if (totalPixels < 250000) {
        return 1;
    } else if (totalPixels < 400000) {
        return 2;
    } else if (totalPixels < 520000) {
        return 2;
    } else if (totalPixels < 650000) {
        return 3;
    } else {
        return 2;
    }
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

vector<Point> SPs(const Mat& image, int k) {
    vector<Point> seedPoints;
    vector<double> minDistances(image.rows * image.cols, DBL_MAX);
    
    // 选择第一个种子点（图像中心）
    Point firstSeed(image.cols / 2, image.rows / 2);
    seedPoints.push_back(firstSeed);
    
    // 为剩余的k-1个种子点使用K-means++策略
    for (int i = 1; i < k; i++) {
        // 计算每个像素到最近种子点的距离
        double totalDistance = 0.0;
        for (int y = 0; y < image.rows; y++) {
            for (int x = 0; x < image.cols; x++) {
                Vec3b currentColor = image.at<Vec3b>(y, x);
                double minDist = DBL_MAX;
                
                // 找到最近的种子点
                for (const Point& seed : seedPoints) {
                    Vec3b seedColor = image.at<Vec3b>(seed.y, seed.x);
                    double dist = norm(Vec3d(currentColor) - Vec3d(seedColor));
                    minDist = min(minDist, dist);
                }
                
                minDistances[y * image.cols + x] = minDist;
                totalDistance += minDist;
            }
        }
        
        // 选择新的种子点
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

LKSPI CI(const Mat& originalImage, int blockSize) {
    LKSPI result;
    result.originalRows = originalImage.rows;
    result.originalCols = originalImage.cols;
    
    int compressedRows = (originalImage.rows + blockSize - 1) / blockSize;
    int compressedCols = (originalImage.cols + blockSize - 1) / blockSize;
    
    result.compressed = Mat(compressedRows, compressedCols, originalImage.type());
    result.blockColors.resize(compressedRows, vector<Vec3b>(compressedCols));
    
    // 使用 async 并行处理每个块
    vector<future<void>> futures;
    
    for (int i = 0; i < compressedRows; i++) {
        futures.push_back(async(launch::async, [&](int row) {
            for (int j = 0; j < compressedCols; j++) {
                Vec3d avgColor(0, 0, 0);
                int count = 0;
                
                for (int bi = 0; bi < blockSize && row * blockSize + bi < originalImage.rows; bi++) {
                    for (int bj = 0; bj < blockSize && j * blockSize + bj < originalImage.cols; bj++) {
                        avgColor += Vec3d(originalImage.at<Vec3b>(row * blockSize + bi, j * blockSize + bj));
                        count++;
                    }
                }
                
                avgColor /= count;
                Vec3b finalColor(avgColor[0], avgColor[1], avgColor[2]);
                result.compressed.at<Vec3b>(row, j) = finalColor;
                result.blockColors[row][j] = finalColor;
            }
        }, i));
    }
    
    for (auto& future : futures) {
        future.get();
    }
    
    return result;
}

vector<vector<vector<double>>> Matrix(const Mat& image, const vector<Point>& seedPoints) {
    int rows = image.rows;
    int cols = image.cols;
    int k = seedPoints.size();
    
    vector<vector<vector<double>>> probMatrix(rows,
        vector<vector<double>>(cols, vector<double>(k, 0.0)));
    
    // 使用 async 并行处理每个像素
    vector<future<void>> futures;
    
    for (int y = 0; y < rows; ++y) {
        futures.push_back(async(launch::async, [&](int row) {
            for (int x = 0; x < cols; ++x) {
                Vec3b pixelColor = image.at<Vec3b>(row, x);
                Point currentPoint(x, row);
                
                double sumWeights = 0.0;
                vector<double> weights(k);
                
                for (int s = 0; s < k; ++s) {
                    Vec3b seedColor = image.at<Vec3b>(seedPoints[s].y, seedPoints[s].x);
                    weights[s] = WeightK(pixelColor, seedColor, currentPoint, seedPoints[s]);
                    sumWeights += weights[s];
                }
                
                if (sumWeights > 0) {
                    for (int s = 0; s < k; ++s) {
                        probMatrix[row][x][s] = weights[s] / sumWeights;
                    }
                }
            }
        }, y));
    }
    
    for (auto& future : futures) {
        future.get();
    }
    
    return probMatrix;
}
vector<Vec3b> CSCP(const Mat& image,const vector<vector<vector<double>>>& probMatrix,int k) {
    vector<Vec3d> colorSums(k, Vec3d(0, 0, 0));
    vector<double> counts(k, 0.0);
    
    // 使用 async 并行处理每个段
    vector<future<pair<Vec3d, double>>> futures;
    
    for (int s = 0; s < k; ++s) {
        futures.push_back(async(launch::async, [&](int segment) {
            Vec3d sum(0, 0, 0);
            double count = 0.0;
            
            for (int y = 0; y < image.rows; ++y) {
                for (int x = 0; x < image.cols; ++x) {
                    double prob = probMatrix[y][x][segment];
                    Vec3b pixelColor = image.at<Vec3b>(y, x);
                    sum += Vec3d(pixelColor) * prob;
                    count += prob;
                }
            }
            
            return make_pair(sum, count);
        }, s));
    }
    
    for (int s = 0; s < k; ++s) {
        auto result = futures[s].get();
        colorSums[s] = result.first;
        counts[s] = result.second;
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

void EMS(vector<vector<vector<double>>>& probMatrix,
const Mat& image,vector<Point>& seedPoints,
vector<Vec3b>& segmentColors,int maxIterations = 10) {
    int rows = image.rows;
    int cols = image.cols;
    int k = seedPoints.size();
    
    for (int iter = 0; iter < maxIterations; ++iter) {
        // 并行计算新的中心点
        vector<future<pair<Vec3d, double>>> centerFutures;
        for (int s = 0; s < k; ++s) {
            centerFutures.push_back(async(launch::async, [&](int segment) {
                Vec3d sum(0, 0, 0);
                double count = 0.0;
                
                for (int y = 0; y < rows; ++y) {
                    for (int x = 0; x < cols; ++x) {
                        double prob = probMatrix[y][x][segment];
                        Vec3b pixelColor = image.at<Vec3b>(y, x);
                        sum += Vec3d(pixelColor) * prob;
                        count += prob;
                    }
                }
                
                return make_pair(sum, count);
            }, s));
        }
        
        vector<Vec3d> newCenters(k);
        vector<double> counts(k);
        for (int s = 0; s < k; ++s) {
            auto result = centerFutures[s].get();
            if (result.second > 0) {
                newCenters[s] = result.first / result.second;
            }
        }
        
        // 并行更新种子点
        vector<future<Point>> seedFutures;
        for (int s = 0; s < k; ++s) {
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
        
        for (int s = 0; s < k; ++s) {
            seedPoints[s] = seedFutures[s].get();
        }
        
        // 并行更新概率矩阵和段颜色
        auto newProbMatrix = Matrix(image, seedPoints);
        segmentColors = CSCP(image, newProbMatrix, k);
        
        probMatrix =std:: move(newProbMatrix);
    }
}

Mat GSRP(const Mat& original,const vector<vector<vector<double>>>& probMatrix,const vector<Vec3b>& segmentColors) {
    Mat result(original.size(), original.type());
    int k = segmentColors.size();
    
    // 并行处理每一行
    vector<future<void>> futures;
    for (int y = 0; y < original.rows; ++y) {
        futures.push_back(async(launch::async, [&](int row) {
            for (int x = 0; x < original.cols; ++x) {
                int maxSegment = 0;
                double maxProb = probMatrix[row][x][0];
                
                for (int s = 1; s < k; ++s) {
                    if (probMatrix[row][x][s] > maxProb) {
                        maxProb = probMatrix[row][x][s];
                        maxSegment = s;
                    }
                }
                
                result.at<Vec3b>(row, x) = segmentColors[maxSegment];
            }
        }, y));
    }
    
    for (auto& future : futures) {
        future.get();
    }
    
    return result;
}

vector<Mat> LRs(const Mat& original,
                                 const vector<vector<vector<double>>>& probMatrix,
                                 const vector<Vec3b>& segmentColors) {
    int k = segmentColors.size();
    vector<Mat> layers(k);
    
    // 初始化K个图层，全部设为透明
    for (int i = 0; i < k; ++i) {
        layers[i] = Mat(original.size(), CV_8UC4, Scalar(0, 0, 0, 0));
    }
    
    // 对每个像素，将其分配到概率最大的图层
    for (int y = 0; y < original.rows; ++y) {
        for (int x = 0; x < original.cols; ++x) {
            // 找到最大概率的分割区域
            int maxSegment = 0;
            double maxProb = probMatrix[y][x][0];
            
            for (int s = 1; s < k; ++s) {
                if (probMatrix[y][x][s] > maxProb) {
                    maxProb = probMatrix[y][x][s];
                    maxSegment = s;
                }
            }
            
            // 获取原始像素颜色
            Vec3b originalColor = original.at<Vec3b>(y, x);
            
            // 将像素添加到对应图层，设置alpha通道为255（完全不透明）
            layers[maxSegment].at<Vec4b>(y, x) = Vec4b(originalColor[0], 
                                                      originalColor[1], 
                                                      originalColor[2], 
                                                      255);
        }
    }
    
    return layers;
}

int main() {
    Mat image = imread("Pictures/2000*2000.png");
    // Mat image = imread("Pictures/1920*1440.png");
    // Mat image = imread("Pictures/220*220.png");
    if (image.empty()) {
        cout << "Wrong Image Name, Please Reload!" << endl;
        return -1;
    }
    
    int totalPixels = image.rows * image.cols;
    cout << "Image size: " << image.cols << "x" << image.rows << endl;
    cout << "Total pixels: " << totalPixels << endl;
    int BLOCK_SIZE = DBS(totalPixels);
    cout << "Block size: " << BLOCK_SIZE << endl;
    
    auto start = chrono::high_resolution_clock::now();
    LKSPI compressedImage = CI(image, BLOCK_SIZE);
    cout << "Compressed image size: " << compressedImage.compressed.cols << "x" << compressedImage.compressed.rows << endl;
    
    const int k = 5;
    vector<Point> seedPoints = SPs(compressedImage.compressed, k);
    auto probMatrix = Matrix(compressedImage.compressed, seedPoints);
    vector<Vec3b> segmentColors = CSCP(compressedImage.compressed, probMatrix, k);
    EMS(probMatrix, compressedImage.compressed, seedPoints, segmentColors);
    vector<Mat> segmentLayers = LRs(compressedImage.compressed, probMatrix,segmentColors);
    Mat compressedResult = GSRP(compressedImage.compressed, probMatrix, segmentColors);
    Mat result;
    resize(compressedResult, result, image.size(), 0, 0, INTER_NEAREST);
    
    for (int i = 0; i < segmentLayers.size(); ++i) {
        Mat resizedLayer;
        resize(segmentLayers[i], resizedLayer, image.size(), 0, 0, INTER_NEAREST);
        Mat overlay;
        resizedLayer.copyTo(overlay);
        string filename = format("V4.0_Layer_%d.png", i + 1);
        vector<int> pngParams = {IMWRITE_PNG_COMPRESSION, 9};
        imwrite(filename, resizedLayer, pngParams);
    }
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;
    double processingTime = duration.count();
    double pixelsPerSecond = totalPixels / processingTime;
    
    cout << "Time: " << processingTime << " seconds" << endl;
    cout << "Speed: " << static_cast<int>(pixelsPerSecond) << " pixels/second" << endl;
    cout << "Number of segments: " << k << endl;
    for (int i = 0; i < k; ++i) {
        Point adjustedSeed(
            seedPoints[i].x * image.cols / compressedImage.compressed.cols,
            seedPoints[i].y * image.rows / compressedImage.compressed.rows
        );
        circle(result, adjustedSeed, 1, Scalar(0, 255, 0), -1);
    //     cout << "Seed point " << i + 1 << ": (" << adjustedSeed.x << ", " << adjustedSeed.y << ")" << endl;
    }
    string performanceText = format("Time:%.3fs, Speed:%d px/s, Image Size:%dx%d, @copyright Jsy&Whn", processingTime, static_cast<int>(pixelsPerSecond),image.cols,image.rows );
    putText(result, performanceText, 
            Point(10, result.rows - 10), 
            FONT_HERSHEY_SIMPLEX, 
            0.50,
            Scalar(0, 115, 255), 
            1);
            
    imwrite("V4.0_Final_Image.png", result);
    return 0;
}