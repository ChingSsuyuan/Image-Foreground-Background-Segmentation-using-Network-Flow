#include "GraphMatrix.h"
#include "FeatureExtractor.h"
// 计算像素间的相似权重
int calculateSimilarityWeight(const PixelFeature& feature1, const PixelFeature& feature2) {
    int dx = feature1.colorRGB[0] - feature2.colorRGB[0];
    int dy = feature1.colorRGB[1] - feature2.colorRGB[1];
    int dz = feature1.colorRGB[2] - feature2.colorRGB[2];
    double rgbDistance = std::sqrt(dx * dx + dy * dy + dz * dz);
    
    double gradDistance = std::abs(feature1.gradientMagnitude - feature2.gradientMagnitude);
    
    int weight = static_cast<int>(C * std::exp(-(ALPHA * (rgbDistance / SIGMA_RGB) + BETA * (gradDistance / SIGMA_GRAD))));
    return std::clamp(weight, 1, 100);
}

// 使用矩阵表示生成图
std::vector<std::vector<int>> generateGraphMatrix(const cv::Mat& image, const std::vector<std::vector<PixelFeature>>& features) {
    int totalPixels = image.rows * image.cols;
    std::vector<std::vector<int>> adjMatrix(totalPixels, std::vector<int>(totalPixels, 0));

    // 辅助函数将2D坐标转换为1D索引
    auto posToIndex = [&](int x, int y) -> int {
        return y * image.cols + x;
    };

    int dx[] = {0, 1, 0, -1};  // 右，下，左，上
    int dy[] = {-1, 0, 1, 0};

    // 遍历所有像素，填充邻接矩阵
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            int currentIndex = posToIndex(x, y);
            const PixelFeature& currentFeature = features[y][x];

            for (int i = 0; i < 4; ++i) {
                int nx = x + dx[i];
                int ny = y + dy[i];

                if (nx >= 0 && nx < image.cols && ny >= 0 && ny < image.rows) {
                    int neighborIndex = posToIndex(nx, ny);
                    const PixelFeature& neighborFeature = features[ny][nx];
                    
                    // 计算权重并填充到邻接矩阵
                    int weight = calculateSimilarityWeight(currentFeature, neighborFeature);
                    adjMatrix[currentIndex][neighborIndex] = weight;
                }
            }
        }
    }

    return adjMatrix;
}
