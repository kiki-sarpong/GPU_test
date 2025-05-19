#include <iostream>
#include <fstream>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include <opencv2/opencv.hpp>
#include <vector>


using namespace std;

/*
Find more info about the models here -> https://github.com/isl-org/MiDaS?tab=readme-ov-file

https://www.youtube.com/watch?v=7fCheEYUpgU&list=PLkmvobsnE0GHmLeVETd6zbbJSDZJWa5Fw&index=8

https://github.com/niconielsen32/ComputerVision/blob/master/MonocularDepth/depthEstimationMono.cpp


This class loads a neural network for creating a depth map with a monocular camera
*/

class MonocularDepthMap{
    public:
        // string model_path = "/home/kiki-jetson/Desktop/projects/gpu_test/models/";
        // string active_model =  model_path + "model-f6b98070.onnx ";   // Must be ONNX
        string active_model = "/home/kiki-jetson/Desktop/projects/gpu_test/models/model-f6b98070.onnx";
        MonocularDepthMap(){}
        ~MonocularDepthMap(){
            cout << "Destroying object \n";
        }

        /*
        This method clips the value b/n an upper and lower bounds
        */
        int clip(int n, int lower, int upper){
            return max(lower, min(n, upper));
        }

        /*
        Default setup for getting the output names from the nueral network.
        */
        void getOutputNames(const cv::dnn::Net& net){
            if (output_names.empty()){
                vector<int> out_layers = net.getUnconnectedOutLayers();
                vector<string> layers_name = net.getLayerNames();
                output_names.resize(out_layers.size());
                for(int i = 0; i < out_layers.size(); ++i){
                    output_names[i] = layers_name[out_layers[i] - 1];
                }

            }

        }

        void runMonocularModel(cv::VideoCapture& cap, int width, int height){
            // Read neural network from file
            auto net = cv::dnn::readNetFromONNX(active_model);

            if (net.empty()){
                cerr << "Model didn't upload correctly..\n";
                return;
            }

            // Run on CPU than GPU
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

            while(cap.isOpened()){

                cv::Mat frame;
                cap.read(frame);

                if (frame.empty()){
                    cerr << "Frame is empty, can't process any data!\n";
                    return;
                }

                // int w = frame.rows;
                // int h = frame.cols;

                auto start = cv::getTickCount();  // Get start count

                // Create blob from image input
                // (scale :1/255, size: 384 x 384, mean subtraction: (123.675,116.28,103.53), channels order : RGB )
                cv::Mat blob = cv::dnn::blobFromImage(frame, 1/255.f, cv::Size(384, 384), cv::Scalar(123.675,116.28,103.53), true, false);

                // Set the blob to be input to the neural network
                net.setInput(blob);

                // Forward pass of the blob through the NN to get predictions
                getOutputNames(net);   // Update output names;
                cv::Mat output = net.forward(output_names[0]);
                // Convert size to 384x384 from 1x384x384
                const vector<int> size = {output.size[1], output.size[2]};
                output = cv::Mat(static_cast<int>(size.size()), &size[0], CV_32F, output.ptr<float>());

                // Resize output image to input image size
                cv::resize(output, output, cv::Size(width, height));
                // Visualize output image
                double mn, mx; // min, max
                cv::minMaxLoc(output, &mn, &mx);
                const double range = mx - mn;

                // Normalize (0-1)
                output.convertTo(output, CV_32F, 1.0/range, -(mn/range));
                // Scaling (0 - 255)
                output.convertTo(output, CV_8U, 255);

                // Calculate FPS
                auto end = cv::getTickCount();
                auto total_time = (end-start)/cv::getTickFrequency();
                auto fps = 1/total_time;


                cout << "FPS: " << fps << "\n";
                cv::putText(output, "FPS: " + std::to_string(int(fps)), cv::Point(50,50), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0,255,255), 2);

                cv::imshow("Monocular depth image", output);

                if (cv::waitKey(WAIT_TIME_MS) == ESC_KEY){
                    break;
                }

            }

        }


    private:
        // int width_;
        // int height_;
        // cv::VideoCapture& cap_;
        int WAIT_TIME_MS = 30;
        int ESC_KEY = 27;
        vector<string> output_names;
};




int main(int argc, char* argv[]){
    int deviceCount = cv::cuda::getCudaEnabledDeviceCount();

    if (deviceCount == 0) {
        std::cerr << "No CUDA-enabled GPU found.\n";
        return -1;
    }
    
    std::cout << "CUDA-enabled GPU devices: " << deviceCount << "\n";

    int width = 480;
    int height = 360;
    cv::VideoCapture cap(0);

    MonocularDepthMap mdm;
    mdm.runMonocularModel(cap, width, height);

    // Reduce resolution for faster inference| Below wasn't working
    // cap.set(cv::CAP_PROP_FRAME_WIDTH, 720);
    // cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
