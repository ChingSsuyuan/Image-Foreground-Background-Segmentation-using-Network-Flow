#include <opencv2/opencv.hpp>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <memory>
#include "FeatureExtractor.h"
#include "Construct_Graph.h"
#include "EdmondsKarp.h"

using namespace cv;
using namespace std;

void printReachableNodes(const std::shared_ptr<ReachableNode>& head) {
    std::shared_ptr<ReachableNode> current = head;
    while (current) {
        std::cout << "(" << current->x << ", " << current->y << ")\n";
        current = current->next;
    }
}

int main() {
    
    string path = "Pictures/3x3_image.png";
    Mat image = imread(path, IMREAD_COLOR);

    if (!image.data)
    {
        cout << "This is en empty image"<<endl;
        return -1;
    }
    vector<vector<PixelFeature>> extraction = extractFeatures(image);
    for (vector<PixelFeature> i : extraction){
        for (PixelFeature j : i){
            cout << "position " << j.position << endl;
            cout << "colorRGB " <<j.colorRGB << endl;
            cout << "gradientMagnitude " << j.gradientMagnitude << endl;
        }
    }
    
    AdjacencyList adjList = generateGraph(image, extraction);
    for (const auto& [key, head] : adjList) {
        cout << "Node (" << key.first << ", " << key.second << "):";
        shared_ptr<ListNode> current = head;

        while (current) {
            cout << " -> (" << current->x << ", " << current->y << "), weight: " << current->weight;
            current = current->next;
        }
        cout << endl;
    }
    std::shared_ptr<ReachableNode> reachableNodes = nullptr;

    edmondsKarp(adjList, {image.cols/2, image.rows/2}, {0, 0}, reachableNodes);

    std::cout << "Reachable nodes from source (" << image.cols/2 << ", " << image.rows/2 << "):\n";
    printReachableNodes(reachableNodes);


}
