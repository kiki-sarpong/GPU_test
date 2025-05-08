#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>




using namespace std;
// using namespace cv;



/*
| Property                | Meaning                    |
| ----------------------- | -------------------------- |
| `CAP_PROP_FRAME_WIDTH`  | Width of frames            |
| `CAP_PROP_FRAME_HEIGHT` | Height of frames           |
| `CAP_PROP_FPS`          | Frames per second          |
| `CAP_PROP_POS_FRAMES`   | Index of the current frame |
| `CAP_PROP_POS_MSEC`     | Position in milliseconds   |
| `CAP_PROP_FRAME_COUNT`  | Total number of frames     |
| `CAP_PROP_FORMAT`       | Format (backend-specific)  |

*/

// double fps = cap.get(cv::CAP_PROP_FPS);
// cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
// cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

// a compile-time constant.
constexpr int ESC_KEY = 27;
constexpr int WAIT_TIME_MS = 30;

void cpuSpeedTest(cv::VideoCapture& cap, int width, int height){
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open webcam\n";
        return;
    }
    while (true){
        cv::Mat image;
        bool isSuccess = cap.read(image);
        cv::resize(image, image, cv::Size(width, height));

        if (image.empty()){
            std::cerr << "Could not load in image! \n";
            return ; 
        }

        auto start = cv::getTickCount();

        cv::Mat result;
        int d = 30;
        int sigmaColor = 100;
        int sigmaSpace = 100;

        /*
        cv::bilateralFilter is an OpenCV function used for edge-preserving smoothing —
         it reduces noise in an image without blurring edges, making it great for detail-sensitive 
         applications like face filtering or preprocessing before edge detection.
        */
        cv::bilateralFilter(image, result, d, sigmaColor, sigmaSpace);

        auto end = cv::getTickCount();

        // Calculate FPS
        auto total_time = (end - start)/ cv::getTickFrequency();
        auto fps = 1/total_time;

        std::cout << "FPS: " << fps << "\n";
        cv::putText(result, "FPS: " + std::to_string(int(fps)), cv::Point(50,50), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0,255,255));

        cv::imshow("CPU BILATERAL FILTER", result);

        if (cv::waitKey(WAIT_TIME_MS) == ESC_KEY){
            break;
        }


    }

}


void gpuSpeedTest(cv::VideoCapture& cap, int width, int height){
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open webcam\n";
        return;
    }
    while (true){
        cv::Mat image;

        bool isSuccess = cap.read(image);

        if (image.empty()){
            std::cerr << "Could not load in image! \n";
            return ; 
        }

        // Create GPU image and upload kikyss4312

        cv::cuda::GpuMat gpu_image, result_gpu;
        gpu_image.upload(image);

        cv::cuda::resize(gpu_image, gpu_image, cv::Size(width, height));

        auto start = cv::getTickCount();

        cv::Mat result;
        int d = 30;
        int sigmaColor = 100;
        int sigmaSpace = 100;

        /*
        cv::bilateralFilter is an OpenCV function used for edge-preserving smoothing —
         it reduces noise in an image without blurring edges, making it great for detail-sensitive 
         applications like face filtering or preprocessing before edge detection.
        */
        cv::cuda::bilateralFilter(gpu_image, result_gpu, d, sigmaColor, sigmaSpace);

        auto end = cv::getTickCount();

        // Calculate FPS
        auto total_time = (end - start)/ cv::getTickFrequency();
        auto fps = 1/total_time;

        /// Download image
        result_gpu.download(result);

        std::cout << "FPS: " << fps << "\n";
        cv::putText(result, "FPS: " + std::to_string(int(fps)), cv::Point(50,50), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0,255,255), 2);

        cv::imshow("GPU BILATERAL FILTER", result);

        if (cv::waitKey(WAIT_TIME_MS) == ESC_KEY){
            break;
        }

    }

}



int main(int argc, char* argv[]){
    int deviceCount = cv::cuda::getCudaEnabledDeviceCount();

    if (deviceCount == 0) {
        std::cerr << "No CUDA-enabled GPU found.\n";
        return -1;
    }
    
    std::cout << "CUDA-enabled GPU devices: " << deviceCount << "\n";

    int width = 360;
    int height = 240;
    cv::VideoCapture cap(0);

    // Reduce resolution for faster inference| Below wasn't working
    // cap.set(cv::CAP_PROP_FRAME_WIDTH, 720);
    // cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    if (argc > 1 && std::string(argv[1]) == "cpu") {
        cpuSpeedTest(cap, width, height);
    } else {
        gpuSpeedTest(cap, width, height);
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}