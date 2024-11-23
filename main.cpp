#include <FL/Fl_Window.H>
#include <FL/Fl_Box.H>
#include <FL/Fl_Button.H>
#include <FL/Fl_File_Chooser.H>
#include <FL/Fl_Image.H>
#include <FL/Fl_Text_Editor.H>
#include <FL/Fl_Text_Buffer.H>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include "Segmentation.h"  // 包含分割函数头文件
#include <sstream> 

using namespace std;
using namespace cv;

class ImageWindow : public Fl_Window {
public:
    Fl_Box *inputImageBox, *outputImageBox;
    Fl_Button *loadButton, *processButton;
    Fl_Text_Editor *infoEditor;  // 新增文本框显示信息
    Fl_Text_Buffer *infoBuffer;  // 新增文本缓冲区

    // 用来保存图像对象，防止内存泄漏
    Fl_RGB_Image *inputImage = nullptr;
    Fl_RGB_Image *outputImage = nullptr;
    uchar *inputImageData = nullptr;
    uchar *outputImageData = nullptr;

    // 存储加载的 OpenCV 图像
    cv::Mat inputImg;

    ImageWindow(int w, int h, const char* title) : Fl_Window(w, h, title) {
        // 创建显示载入图片的框
        inputImageBox = new Fl_Box(20, 40, w / 2 - 40, h - 100);
        inputImageBox->label("Loaded Image");
        inputImageBox->labelcolor(FL_BLUE);

        // 创建显示输出图片的框
        outputImageBox = new Fl_Box(w / 2 + 20, 40, w / 2 - 40, h - 100);
        outputImageBox->label("Processed Image");
        outputImageBox->labelcolor(FL_BLUE);

        // 创建“Load Image”按钮
        loadButton = new Fl_Button(w / 2 - 110, h - 60, 100, 30, "Load Image");
        loadButton->callback(LoadImageCallback, this);

        // 创建“Process”按钮
        processButton = new Fl_Button(w / 2 + 10, h - 60, 100, 30, "Process");
        processButton->callback(ProcessImageCallback, this);

        // 创建文本框，用于显示计算时间和运算量
        infoEditor = new Fl_Text_Editor(w / 2 - 110, h - 150, w - 40, 80);
        infoEditor->box(FL_FLAT_BOX);
        infoEditor->color(FL_GRAY);
        infoEditor->textcolor(FL_BLACK);
        infoBuffer = new Fl_Text_Buffer();
        infoEditor->buffer(infoBuffer);

        end();
    }

    ~ImageWindow() {
        clearImages();
        delete inputImageBox;
        delete outputImageBox;
        delete loadButton;
        delete processButton;
        delete infoEditor;
    }

    void clearImages() {
        if (inputImage) {
            inputImageBox->image(nullptr);
            delete inputImage;
            inputImage = nullptr;
        }
        if (outputImage) {
            outputImageBox->image(nullptr);
            delete outputImage;
            outputImage = nullptr;
        }
        if (inputImageData) {
            delete[] inputImageData;
            inputImageData = nullptr;
        }
        if (outputImageData) {
            delete[] outputImageData;
            outputImageData = nullptr;
        }
    }

    // 计算等比例缩放后的尺寸
    static void calculateScaledSize(int originalWidth, int originalHeight, 
                                    int boxWidth, int boxHeight,
                                    int& newWidth, int& newHeight) {
        double scaleX = static_cast<double>(boxWidth) / originalWidth;
        double scaleY = static_cast<double>(boxHeight) / originalHeight;
        double scale = std::min(scaleX, scaleY);
        
        newWidth = static_cast<int>(originalWidth * scale);
        newHeight = static_cast<int>(originalHeight * scale);
    }

    // 图像缩放函数
    static cv::Mat resizeImage(const cv::Mat& img, int boxWidth, int boxHeight) {
        int newWidth, newHeight;
        calculateScaledSize(img.cols, img.rows, boxWidth, boxHeight, newWidth, newHeight);
        
        cv::Mat resized;
        cv::resize(img, resized, cv::Size(newWidth, newHeight), 0, 0, cv::INTER_AREA);
        
        // 创建一个与显示框大小相同的白色背景图像
        cv::Mat canvas = cv::Mat::zeros(boxHeight, boxWidth, img.type());
        canvas.setTo(cv::Scalar(255, 255, 255)); // 设置白色背景
        
        // 计算图像应该放置的位置（居中）
        int x = (boxWidth - newWidth) / 2;
        int y = (boxHeight - newHeight) / 2;
        
        // 将缩放后的图像复制到画布中心
        resized.copyTo(canvas(cv::Rect(x, y, newWidth, newHeight)));
        
        return canvas;
    }

    static void LoadImageCallback(Fl_Widget* widget, void* data) {
        ImageWindow* win = (ImageWindow*)data;
        const char* filename = fl_file_chooser("Open Image", "*.png|*.bmp|*.jpg|", "");

        if (filename) {
            // 清除之前的所有图像和数据
            win->clearImages();

            // 使用 OpenCV 读取原始图像
            win->inputImg = cv::imread(filename);
            if (win->inputImg.empty()) {
                fl_message("Failed to load image");
                return;
            }

            // 显示原始图像
            win->displayImage(win->inputImageBox, win->inputImage, win->inputImageData, win->inputImg);

            // 清空文本框
            win->infoBuffer->text("");

            // 更新信息框
            win->infoBuffer->append("Image loaded successfully.\n");

            // 重绘窗口
            win->redraw();
        }
    }

    static void ProcessImageCallback(Fl_Widget* widget, void* data) {
        ImageWindow* win = (ImageWindow*)data;

        if (win->inputImg.empty()) {
            fl_message("No image loaded! Please load an image first.");
            return;
        }

        // 记录开始时间
        auto start = chrono::high_resolution_clock::now();

        // 调用分割函数处理图像
        cv::Mat outputImg = performSegmentation(win->inputImg);

        // 记录结束时间
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> duration = end - start;

        // 显示分割后的图像
        win->displayImage(win->outputImageBox, win->outputImage, win->outputImageData, outputImg);

        // 更新信息框
        win->infoBuffer->text("");  // 清空当前文本

        // 使用字符串流来拼接字符串
        win->infoBuffer->append("Processing complete.\n");
        win->infoBuffer->append(("Time taken: " + std::to_string(duration.count()) + " seconds\n").c_str());
        win->infoBuffer->append(("Processed " + std::to_string(win->inputImg.rows * win->inputImg.cols) + " pixels.\n").c_str());

        // 重绘窗口
        win->redraw();
    }

    void displayImage(Fl_Box* imageBox, Fl_RGB_Image*& flImage, uchar*& imageData, const cv::Mat& img) {
        // 将 OpenCV 图像转换为 RGB 格式
        cv::Mat rgbImg;
        cv::cvtColor(img, rgbImg, cv::COLOR_BGR2RGB);

        // 为图像数据分配内存
        imageData = new uchar[rgbImg.total() * rgbImg.channels()];
        memcpy(imageData, rgbImg.data, rgbImg.total() * rgbImg.channels());

        // 创建新的 FLTK 图像
        flImage = new Fl_RGB_Image(imageData, rgbImg.cols, rgbImg.rows, 3);
        imageBox->image(flImage);
    }
};

int main() {
    int windowWidth = 1200;  // 设置窗口宽度
    int windowHeight = 800; // 设置窗口高度

    ImageWindow* window = new ImageWindow(windowWidth, windowHeight, "Image Segmentation");
    window->show();
    return Fl::run();
}