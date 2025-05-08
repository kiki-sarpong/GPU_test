#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>


using namespace std;
// using namespace cv;

int main(int argc, char* argv[]){

    std::cout << "Is CUDA ENABLED ? ->  ";
    std::cout << cv::cuda::getCudaEnabledDeviceCount() << std::endl;

    try{
        cv::Mat src_img, dst_img; // Regular matrix types

        src_img = cv::imread("../happy_people.png", 0);

        if (src_img.empty()){
            std::cerr << "Error with uploading image" << std::endl;
            return -1;
        }

        // Upload image to gpu
        cv::cuda::GpuMat d_img, d_blurred;    // Matrix types for gpu operations
        d_img.upload(src_img);    // Convert to the GPU mat type

        // Create kernel size
        cv::Size kernel_size(15,15);
        double sigma = 3.0;
        cv::cuda::bilateralFilter(d_img, d_blurred, 30, 100, 100);

        d_blurred.download(dst_img);    // Need to download from gpu format

        // cv::imshow("Blurred image", dst_img);
        cv::imwrite("blurred_img.png", dst_img);
        cv::waitKey(0);

    }
    catch (const cv::Exception& ex){
        cout << "Error: " << ex.what() << endl;
    }


    return  0;
}