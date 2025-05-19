#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>



using namespace cv;
using namespace std;

int main(int argc, char** argv){

    cuda::printCudaDeviceInfo(0);  // print device information


    Mat image, result_image;
    cuda::GpuMat gpu_image, gpu_conv_image;

    image = imread("../happy_people.png", 1);

    gpu_image.upload(image);


    //core operations
    cuda::cvtColor(gpu_image, gpu_image, COLOR_BGR2GRAY);
    cuda::transpose(gpu_image, gpu_image);

    gpu_image.download(result_image);

    imwrite("image_result.png", result_image);

    return 0;
}