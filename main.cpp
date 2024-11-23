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
    Fl_Text_Editor *infoEditor;
    Fl_Text_Buffer *infoBuffer;

    Fl_RGB_Image *inputImage = nullptr;
    Fl_RGB_Image *outputImage = nullptr;
    uchar *inputImageData = nullptr;
    uchar *outputImageData = nullptr;

    cv::Mat inputImg;  // 保存原始尺寸的输入图片

    ImageWindow(int w, int h, const char* title) : Fl_Window(w, h, title) {
        inputImageBox = new Fl_Box(20, 40, w / 2 - 40, h - 100);
        inputImageBox->label("Loaded Image");
        inputImageBox->labelcolor(FL_BLUE);

        outputImageBox = new Fl_Box(w / 2 + 20, 40, w / 2 - 40, h - 100);
        outputImageBox->label("Processed Image");
        outputImageBox->labelcolor(FL_BLUE);

        loadButton = new Fl_Button(w / 2 - 110, h - 60, 100, 30, "Load Image");
        loadButton->callback(LoadImageCallback, this);

        processButton = new Fl_Button(w / 2 + 10, h - 60, 100, 30, "Process");
        processButton->callback(ProcessImageCallback, this);

        infoEditor = new Fl_Text_Editor(w / 2 - 110, h - 150, w - 40, 80);
        infoEditor->box(FL_NO_BOX);  // 设置背景为透明
        infoEditor->textcolor(FL_BLACK);  // 设置文字颜色
        infoEditor->textfont(FL_HELVETICA_BOLD);  // 加粗字体
        infoEditor->textsize(18);  // 设置文字大小
        infoBuffer = new Fl_Text_Buffer();
        infoEditor->buffer(infoBuffer);
        end();

        end();
    }

    ~ImageWindow() {
        clearImages();
        delete inputImageBox;
        delete outputImageBox;
        delete loadButton;
        delete processButton;
        delete infoEditor;
        delete infoBuffer;
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

    static void LoadImageCallback(Fl_Widget* widget, void* data) {
        ImageWindow* win = (ImageWindow*)data;
        const char* filename = fl_file_chooser("Open Image", "*.png|*.bmp|*.jpg|", "");

        if (filename) {
            win->clearImages();

            // 读取原始图像并保存
            win->inputImg = cv::imread(filename);
            if (win->inputImg.empty()) {
                fl_message("Failed to load image");
                return;
            }

            // 创建缩放后的显示图像
            cv::Mat displayImg = win->createDisplayImage(win->inputImg, 
                win->inputImageBox->w(), win->inputImageBox->h());

            // 显示缩放后的图像
            win->displayImage(win->inputImageBox, win->inputImage, 
                win->inputImageData, displayImg);

            win->infoBuffer->text("");
            win->infoBuffer->append("Image loaded successfully.\n");
            win->redraw();
        }
    }

    static void ProcessImageCallback(Fl_Widget* widget, void* data) {
        ImageWindow* win = (ImageWindow*)data;

        if (win->inputImg.empty()) {
            fl_message("No image loaded! Please load an image first.");
            return;
        }

        auto start = chrono::high_resolution_clock::now();

        // 使用原始尺寸的图像进行处理
        cv::Mat processedImg = performSegmentation(win->inputImg);

        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> duration = end - start;

        // 创建缩放后的显示图像
        cv::Mat displayImg = win->createDisplayImage(processedImg, 
            win->outputImageBox->w(), win->outputImageBox->h());

        // 显示缩放后的图像
        win->displayImage(win->outputImageBox, win->outputImage, 
            win->outputImageData, displayImg);

        win->infoBuffer->text("");
        win->infoBuffer->append("Processing complete.\n");
        win->infoBuffer->append(("Time taken: " + std::to_string(duration.count()) + " seconds\n").c_str());
        win->infoBuffer->append(("Processed " + std::to_string(win->inputImg.rows * win->inputImg.cols) + " pixels.\n").c_str());

        win->redraw();
    }

    // 创建用于显示的缩放图像
    cv::Mat createDisplayImage(const cv::Mat& img, int boxWidth, int boxHeight) {
        int newWidth, newHeight;
        calculateScaledSize(img.cols, img.rows, boxWidth, boxHeight, newWidth, newHeight);
        
        cv::Mat resized;
        cv::resize(img, resized, cv::Size(newWidth, newHeight), 0, 0, cv::INTER_AREA);
        
        // 创建白色背景的画布
        cv::Mat canvas = cv::Mat::zeros(boxHeight, boxWidth, img.type());
        canvas.setTo(cv::Scalar(255, 255, 255));
        
        // 将缩放后的图像放置在画布中心
        int x = (boxWidth - newWidth) / 2;
        int y = (boxHeight - newHeight) / 2;
        resized.copyTo(canvas(cv::Rect(x, y, newWidth, newHeight)));
        
        return canvas;
    }

    void displayImage(Fl_Box* imageBox, Fl_RGB_Image*& flImage, uchar*& imageData, const cv::Mat& img) {
        cv::Mat rgbImg;
        cv::cvtColor(img, rgbImg, cv::COLOR_BGR2RGB);

        imageData = new uchar[rgbImg.total() * rgbImg.channels()];
        memcpy(imageData, rgbImg.data, rgbImg.total() * rgbImg.channels());

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
