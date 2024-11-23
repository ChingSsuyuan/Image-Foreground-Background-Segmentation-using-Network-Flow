#include <FL/Fl_Window.H>
#include <FL/Fl_Box.H>
#include <FL/Fl_Button.H>
#include <FL/Fl_File_Chooser.H>
#include <FL/Fl_Image.H>
#include <FL/Fl_Text_Editor.H>
#include <FL/Fl_Text_Buffer.H>
#include <FL/Fl_Group.H>
#include <FL/Fl_Progress.H>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <sstream>

using namespace std;
using namespace cv;
struct PixelFeature { 
    Vec3b colorRGB;  
    Point position;   
};  

const int C = 100;             
const double SIGMA_RGB = 8.0;   // bigger more accurate but slower
const double ALPHA = 0.2;       // smaller more accurate but slower

// 添加辅助函数
int Weight(const Vec3b& color1, const Vec3b& color2) {
    int dx = color1[0] - color2[0];
    int dy = color1[1] - color2[1];
    int dz = color1[2] - color2[2];
    double rgbDistance = std::sqrt(dx * dx + dy * dy + dz * dz);
    return static_cast<int>(C * std::exp(-rgbDistance / SIGMA_RGB)); 
}

std::vector<std::vector<std::vector<int>>> Build_Matrix(const Mat& image) {
    std::vector<std::vector<PixelFeature>> features(image.rows, std::vector<PixelFeature>(image.cols));
    std::vector<std::vector<std::vector<int>>> Matrix(image.rows, 
        std::vector<std::vector<int>>(image.cols, std::vector<int>(4, 0)));

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

// 添加Graph类定义
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
        adj[v].push_back({u, cap});
        capacity[u][v] = cap;
        capacity[v][u] = cap;
    }

    void Kolmogorov(int source, int sink, vector<int>& S, vector<int>& T) {
        vector<int> level(V);
        vector<int> ptr(V);
        vector<int> flow_to(V);
        vector<bool> in_queue(V, false);
        vector<int> parent(V, -1);

        fill(flow[0].begin(), flow[0].end(), 0);
        queue<int> q;

        while (true) {
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

            if (level[sink] == -1) break;

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

        for (int i = 0; i < V; i++) {
            if (!visited[i]) {
                T.push_back(i);
            }
        }
    }
};
cv::Mat performSegmentation(const cv::Mat& inputImg) {
    // 确保输入图像是连续的内存布局
    cv::Mat image;
    if (!inputImg.isContinuous()) {
        image = inputImg.clone();
    } else {
        image = inputImg;
    }
    
    int rows = image.rows;
    int cols = image.cols;
    
    // 使用引用来避免复制
    const int source = (rows / 2) * cols + (cols / 2);  
    const int sink = rows * cols - 2;
    
    // 直接引用颜色值而不是复制
    const Vec3b& sourceColor = image.at<Vec3b>(source / cols, source % cols);
    const Vec3b& sinkColor = image.at<Vec3b>(sink / cols, sink % cols);
    
    // 预分配内存
    auto Matrix = Build_Matrix(image);
    Graph graph(rows * cols);
    
    // 优化图构建过程
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            const int u = y * cols + x;
            
            if (y > 0) {
                const int v = (y - 1) * cols + x;
                #pragma omp critical
                graph.addEdge(u, v, Matrix[y][x][0]);
            }
            if (y < rows - 1) {
                const int v = (y + 1) * cols + x;
                #pragma omp critical
                graph.addEdge(u, v, Matrix[y][x][1]);
            }
            if (x > 0) {
                const int v = y * cols + (x - 1);
                #pragma omp critical
                graph.addEdge(u, v, Matrix[y][x][2]);
            }
            if (x < cols - 1) {
                const int v = y * cols + (x + 1);
                #pragma omp critical
                graph.addEdge(u, v, Matrix[y][x][3]);
            }
        }
    }

    // 预分配结果向量
    vector<int> S, T;
    S.reserve(rows * cols / 2);
    T.reserve(rows * cols / 2);
    
    graph.Kolmogorov(source, sink, S, T);
    
    // 使用查找表优化
    vector<bool> isInS(rows * cols, false);
    for (int idx : S) {
        isInS[idx] = true;
    }
    
    // 优化结果图像生成
    Mat result = image.clone();
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            const int i = y * cols + x;
            result.at<Vec3b>(y, x) = isInS[i] ? sourceColor : sinkColor;
        }
    }
    
    // 标记源点和汇点
    circle(result, Point(source % cols, source / cols), 1, Scalar(0, 255, 0), -1);
    circle(result, Point(sink % cols, sink / cols), 1, Scalar(0, 0, 255), -1);
    
    return result;
}

// 自定义按钮样式
class CustomButton : public Fl_Button {
public:
    CustomButton(int x, int y, int w, int h, const char* l = 0) : Fl_Button(x, y, w, h, l) {
        box(FL_ROUNDED_BOX);
        color(fl_rgb_color(70, 130, 180));  // Steel Blue
        labelcolor(FL_WHITE);
        labelsize(14);
        labelfont(FL_HELVETICA_BOLD);
    }
    
    int handle(int event) override {
        if (event == FL_ENTER || event == FL_LEAVE) {
            color(event == FL_ENTER ? fl_rgb_color(100, 149, 237) : fl_rgb_color(70, 130, 180));
            redraw();
            return 1;
        }
        return Fl_Button::handle(event);
    }
};

class ImageWindow : public Fl_Window {
private:
    static constexpr int PADDING = 20;
    static constexpr int BUTTON_WIDTH = 120;
    static constexpr int BUTTON_HEIGHT = 35;
    
public:
    Fl_Box *inputImageBox, *outputImageBox;
    CustomButton *loadButton, *processButton;
    Fl_Text_Editor *infoEditor;
    Fl_Text_Buffer *infoBuffer;
    Fl_Group *topGroup, *bottomGroup;
    Fl_Progress *progressBar;
    
    unique_ptr<Fl_RGB_Image> inputImage;
    unique_ptr<Fl_RGB_Image> outputImage;
    unique_ptr<uchar[]> inputImageData;
    unique_ptr<uchar[]> outputImageData;
    
    Mat inputImg;

    ImageWindow(int w, int h, const char* title) : Fl_Window(w, h, title) {
        color(fl_rgb_color(240, 240, 240));
        
        setupTopGroup(w, h);
        setupBottomGroup(w, h);
        
        end();
    }

    ~ImageWindow() = default; // 使用智能指针，不需要手动释放内存

private:
    void setupTopGroup(int w, int h) {
        topGroup = new Fl_Group(0, 0, w, h - 180);
        topGroup->box(FL_FLAT_BOX);
        topGroup->color(FL_WHITE);
        
        int boxWidth = (w - 3 * PADDING) / 2;
        int boxHeight = h - 200;
        
        setupImageBox(inputImageBox, PADDING, PADDING, boxWidth, boxHeight, "Input Image");
        setupImageBox(outputImageBox, w/2 + PADDING, PADDING, boxWidth, boxHeight, "Processed Image");
        
        topGroup->end();
    }

    void setupBottomGroup(int w, int h) {
        bottomGroup = new Fl_Group(0, h - 180, w, 180);
        bottomGroup->box(FL_FLAT_BOX);
        bottomGroup->color(FL_WHITE);
        
        // 进度条
        progressBar = new Fl_Progress(PADDING, h - 170, w - 2 * PADDING, 20);
        progressBar->minimum(0);
        progressBar->maximum(100);
        progressBar->color(FL_BACKGROUND_COLOR);
        progressBar->selection_color(fl_rgb_color(70, 130, 180));
        progressBar->hide();
        
        // 信息显示区
        setupInfoEditor(w, h);
        
        // 按钮
        setupButtons(w, h);
        
        bottomGroup->end();
    }

    void setupImageBox(Fl_Box*& box, int x, int y, int w, int h, const char* label) {
        box = new Fl_Box(x, y, w, h);
        box->box(FL_BORDER_BOX);
        box->color(FL_WHITE);
        box->label(label);
        box->labeltype(FL_NORMAL_LABEL);
        box->labelsize(14);
        box->labelcolor(fl_rgb_color(70, 130, 180));
    }

    void setupInfoEditor(int w, int h) {
        infoEditor = new Fl_Text_Editor(PADDING, h - 140, w - 2 * PADDING, 60);
        infoEditor->box(FL_BORDER_BOX);
        infoEditor->textcolor(fl_rgb_color(50, 50, 50));
        infoEditor->textfont(FL_HELVETICA);
        infoEditor->textsize(12);
        infoBuffer = new Fl_Text_Buffer();
        infoEditor->buffer(infoBuffer);
        infoEditor->deactivate();
    }

    void setupButtons(int w, int h) {
        loadButton = new CustomButton(w/2 - BUTTON_WIDTH - 10, h - 60, 
                                    BUTTON_WIDTH, BUTTON_HEIGHT, "@fileopen Load Image");
        loadButton->callback(LoadImageCallback, this);
        
        processButton = new CustomButton(w/2 + 10, h - 60, BUTTON_WIDTH, BUTTON_HEIGHT, "@refresh Process");
        processButton->callback(ProcessImageCallback, this);
        processButton->deactivate();
    }

    void clearImages() {
        inputImage.reset();
        outputImage.reset();
        inputImageData.reset();
        outputImageData.reset();
        inputImageBox->image(nullptr);
        outputImageBox->image(nullptr);
    }

    static void calculateScaledSize(int originalWidth, int originalHeight, 
                                  int boxWidth, int boxHeight,
                                  int& newWidth, int& newHeight) {
        double scale = min(static_cast<double>(boxWidth) / originalWidth,
                         static_cast<double>(boxHeight) / originalHeight);
        newWidth = static_cast<int>(originalWidth * scale);
        newHeight = static_cast<int>(originalHeight * scale);
    }

    static void LoadImageCallback(Fl_Widget*, void* v) {
        auto* win = static_cast<ImageWindow*>(v);
        const char* filename = fl_file_chooser("Open Image", "Image Files (*.png, *.jpg, *.bmp)", "");
        if (!filename) return;

        win->clearImages();
        win->inputImg = imread(filename);
        
        if (win->inputImg.empty()) {
            fl_alert("Failed to load image!");
            return;
        }

        Mat displayImg = win->createDisplayImage(win->inputImg, 
            win->inputImageBox->w() - 10, win->inputImageBox->h() - 10);
        
        win->displayImage(win->inputImageBox, win->inputImage, 
            win->inputImageData, displayImg);

        stringstream ss;
        ss << "Image loaded successfully!\n"
           << "Resolution: " << win->inputImg.cols << "x" << win->inputImg.rows << " pixels\n";
        win->infoBuffer->text(ss.str().c_str());
        
        win->processButton->activate();
        win->redraw();
    }

    static void ProcessImageCallback(Fl_Widget*, void* v) {
        auto* win = static_cast<ImageWindow*>(v);
        
        if (win->inputImg.empty()) {
            fl_alert("Please load an image first!");
            return;
        }

        win->progressBar->show();
        win->progressBar->value(0);
        Fl::flush();

        // 更新进度到50%表示准备开始处理
        win->progressBar->value(50);
        Fl::check();
        
        // 只计算图像分割的时间
        auto start = chrono::high_resolution_clock::now();
        Mat processedImg = performSegmentation(win->inputImg);
        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration<double>(end - start).count();

        // 更新进度到80%表示处理完成，准备显示
        win->progressBar->value(80);
        Fl::check();
        
        // 显示处理后的图像
        Mat displayImg = win->createDisplayImage(processedImg, 
            win->outputImageBox->w() - 10, win->outputImageBox->h() - 10);
        
        win->displayImage(win->outputImageBox, win->outputImage, 
            win->outputImageData, displayImg);
            
        stringstream ss;
        ss << "Processing complete!\n"
           << "Segmentation time: " << fixed << setprecision(3) << duration << " seconds\n"
           << "Processed pixels: " << win->inputImg.total() << "\n";
        
        win->progressBar->hide();
        win->infoBuffer->text(ss.str().c_str());
        win->redraw();
    }

    Mat createDisplayImage(const Mat& img, int boxWidth, int boxHeight) {
        int newWidth, newHeight;
        calculateScaledSize(img.cols, img.rows, boxWidth, boxHeight, newWidth, newHeight);
        
        Mat canvas(boxHeight, boxWidth, img.type(), Scalar(255, 255, 255));
        Mat resized;
        cv::resize(img, resized, Size(newWidth, newHeight), 0, 0, INTER_AREA);
        
        Rect roi((boxWidth - newWidth) / 2, (boxHeight - newHeight) / 2, newWidth, newHeight);
        resized.copyTo(canvas(roi));
        
        return canvas;
    }
    void displayImage(Fl_Box* imageBox, unique_ptr<Fl_RGB_Image>& flImage, unique_ptr<uchar[]>& imageData, const Mat& img) {
        Mat rgbImg;
        cvtColor(img, rgbImg, COLOR_BGR2RGB);
        imageData = make_unique<uchar[]>(rgbImg.total() * rgbImg.channels());
        memcpy(imageData.get(), rgbImg.data, rgbImg.total() * rgbImg.channels());
        flImage = make_unique<Fl_RGB_Image>(imageData.get(), rgbImg.cols, rgbImg.rows, 3);
        imageBox->image(flImage.get());
    }
};

int main() {
    auto* window = new ImageWindow(1200, 900, "Image Segmentation");
    window->show();
    return Fl::run();
}
