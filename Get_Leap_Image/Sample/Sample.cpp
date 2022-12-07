#pragma once

#include <iostream>
#include <cstring>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include "Leap.h"
#include <cmath>

using namespace Leap;

unsigned char destination[320][120];
bool done = false;

float truncf(float x)
{
    return x >= 0.0f ? floorf(x) : ceilf(x);
}

void undistort(Image image)
{
    float destinationWidth = 320;
    float destinationHeight = 120;

    // define needed variables outside the inner loop
    float calibrationX, calibrationY;
    float weightX, weightY;
    float dX, dX1, dX2, dX3, dX4;
    float dY, dY1, dY2, dY3, dY4;
    int x1, x2, y1, y2;
    int denormalizedX, denormalizedY;
    int i, j;

    const unsigned char *raw = image.data();
    const float *distortion_buffer = image.distortion();

    // Local variables for values needed in loop
    const int distortionWidth = image.distortionWidth();
    const int width = image.width();
    const int height = image.height();

    for (i = 0; i < destinationWidth; i += 1)
    {
        for (j = 0; j < destinationHeight; j += 1)
        {
            // Calculate the position in the calibration map (still with a fractional part)
            calibrationX = 63 * i / destinationWidth;
            calibrationY = 62 * (1 - j / destinationHeight); // The y origin is at the bottom
            // Save the fractional part to use as the weight for interpolation
            weightX = calibrationX - truncf(calibrationX);
            weightY = calibrationY - truncf(calibrationY);

            // Get the x,y coordinates of the closest calibration map points to the target pixel
            x1 = calibrationX; // Note truncation to int
            y1 = calibrationY;
            x2 = x1 + 1;
            y2 = y1 + 1;

            // Look up the x and y values for the 4 calibration map points around the target
            dX1 = distortion_buffer[x1 * 2 + y1 * distortionWidth];
            dX2 = distortion_buffer[x2 * 2 + y1 * distortionWidth];
            dX3 = distortion_buffer[x1 * 2 + y2 * distortionWidth];
            dX4 = distortion_buffer[x2 * 2 + y2 * distortionWidth];
            dY1 = distortion_buffer[x1 * 2 + y1 * distortionWidth + 1];
            dY2 = distortion_buffer[x2 * 2 + y1 * distortionWidth + 1];
            dY3 = distortion_buffer[x1 * 2 + y2 * distortionWidth + 1];
            dY4 = distortion_buffer[x2 * 2 + y2 * distortionWidth + 1];

            // Bilinear interpolation of the looked-up values:
            //  X value
            dX = dX1 * (1 - weightX) * (1 - weightY) +
                 dX2 * weightX * (1 - weightY) +
                 dX3 * (1 - weightX) * weightY +
                 dX4 * weightX * weightY;

            // Y value
            dY = dY1 * (1 - weightX) * (1 - weightY) +
                 dY2 * weightX * (1 - weightY) +
                 dY3 * (1 - weightX) * weightY +
                 dY4 * weightX * weightY;

            // Reject points outside the range [0..1]
            if ((dX >= 0) && (dX <= 1) && (dY >= 0) && (dY <= 1))
            {
                // Denormalize from [0..1] to [0..width] or [0..height]
                denormalizedX = dX * width;
                denormalizedY = dY * height;

                // look up the brightness value for the target pixel
                destination[i][j] = raw[denormalizedX + denormalizedY * width];
            }
            else
            {
                destination[i][j] = -1;
            }
        }
    }
}

class SampleListener : public Listener
{
public:
    virtual void onInit(const Controller &);
    virtual void onConnect(const Controller &);
    virtual void onDisconnect(const Controller &);
    virtual void onExit(const Controller &);
    virtual void onFrame(const Controller &);
    virtual void onFocusGained(const Controller &);
    virtual void onFocusLost(const Controller &);
    virtual void onDeviceChange(const Controller &);
    virtual void onServiceConnect(const Controller &);
    virtual void onServiceDisconnect(const Controller &);

private:
};

void SampleListener::onInit(const Controller &controller)
{
    std::cout << "Initialized" << std::endl;
}

void SampleListener::onConnect(const Controller &controller)
{
    std::cout << "Connected" << std::endl;
}

void SampleListener::onDisconnect(const Controller &controller)
{
    // Note: not dispatched when running in a debugger.
    std::cout << "Disconnected" << std::endl;
}

void SampleListener::onExit(const Controller &controller)
{
    std::cout << "Exited" << std::endl;
}

void SampleListener::onFrame(const Controller &controller)
{
    if (!done)
    {
        // Get the most recent frame and report some basic information
        const Frame frame = controller.frame();
        const Image image = frame.images()[0];

        undistort(image);
        done = true;

        const unsigned char *data = image.data();

        cv::Mat img(100, 100, CV_8UC1);
        unsigned char currImg[100][100];

        //    cv::Mat imgCorrected(240, 640, CV_8UC1);
        for (int i = 0; i < 100; i++)
            for (int j = 0; j < 100; j++)
            {
                //   imgCorrected.at<unsigned char>(j, i) = destination[i][j];
                img.at<unsigned char>(j, i) = data[(270 + i) + ((70 + j) * image.width())];
                currImg[i][j] = data[(270 + i) + ((70 + j) * image.width())];
            }

        //  cv::imshow("ImgCorrected", imgCorrected);
        cv::Mat dst;
        cv::resize(img, dst, cv::Size(400, 400));
        cv::imshow("Img", dst);
        cv::waitKey();

        if (!frame.hands().isEmpty())
        {
            std::cout << "Nothing" << std::endl;
        }
    }
}

void SampleListener::onFocusGained(const Controller &controller)
{
    std::cout << "Focus Gained" << std::endl;
}

void SampleListener::onFocusLost(const Controller &controller)
{
    std::cout << "Focus Lost" << std::endl;
}

void SampleListener::onDeviceChange(const Controller &controller)
{
    std::cout << "Device Changed" << std::endl;
    const DeviceList devices = controller.devices();

    for (int i = 0; i < devices.count(); ++i)
    {
        std::cout << "id: " << devices[i].toString() << std::endl;
        std::cout << "  isStreaming: " << (devices[i].isStreaming() ? "true" : "false") << std::endl;
    }
}

void SampleListener::onServiceConnect(const Controller &controller)
{
    std::cout << "Service Connected" << std::endl;
}

void SampleListener::onServiceDisconnect(const Controller &controller)
{
    std::cout << "Service Disconnected" << std::endl;
}

int main(int argc, char **argv)
{
    // Create a sample listener and controller
    SampleListener listener;
    Controller controller;

    // Have the sample listener receive events from the controller
    controller.addListener(listener);

    controller.setPolicy(Leap::Controller::POLICY_IMAGES);

    // Keep this process running until Enter is pressed
    std::cout << "Press Enter to quit..." << std::endl;
    std::cin.get();

    // Remove the sample listener when done
    controller.removeListener(listener);

    return 0;
}

// code for loading in model in C++ for future endeavors

// #include <stdlib.h>
// #include <stdio.h>
// #include "tensorflow/c/c_api.h"

// void NoOpDeallocator(void *data, size_t a, void *b) {}

// int main()
// {
//     //********* Read model
//     TF_Graph *Graph = TF_NewGraph();
//     TF_Status *Status = TF_NewStatus();

//     TF_SessionOptions *SessionOpts = TF_NewSessionOptions();
//     TF_Buffer *RunOpts = NULL;

//     const char *saved_model_dir = "model/";
//     const char *tags = "serve"; // default model serving tag; can change in future
//     int ntags = 1;

//     TF_Session *Session = TF_LoadSessionFromSavedModel(SessionOpts, RunOpts, saved_model_dir, &tags, ntags, Graph, NULL, Status);
//     if (TF_GetCode(Status) == TF_OK)
//     {
//         printf("TF_LoadSessionFromSavedModel OK\n");
//     }
//     else
//     {
//         printf("%s", TF_Message(Status));
//     }

//     //****** Get input tensor
//     // TODO : need to use saved_model_cli to read saved_model arch
//     int NumInputs = 1;
//     TF_Output *Input = (TF_Output *)malloc(sizeof(TF_Output) * NumInputs);

//     TF_Output t0 = {TF_GraphOperationByName(Graph, "serving_default_input_1"), 0};
//     if (t0.oper == NULL)
//         printf("ERROR: Failed TF_GraphOperationByName serving_default_input_1\n");
//     else
//         printf("TF_GraphOperationByName serving_default_input_1 is OK\n");

//     Input[0] = t0;

//     //********* Get Output tensor
//     int NumOutputs = 1;
//     TF_Output *Output = (TF_Output *)malloc(sizeof(TF_Output) * NumOutputs);

//     TF_Output t2 = {TF_GraphOperationByName(Graph, "StatefulPartitionedCall"), 0};
//     if (t2.oper == NULL)
//         printf("ERROR: Failed TF_GraphOperationByName StatefulPartitionedCall\n");
//     else
//         printf("TF_GraphOperationByName StatefulPartitionedCall is OK\n");

//     Output[0] = t2;

//     //********* Allocate data for inputs & outputs
//     TF_Tensor **InputValues = (TF_Tensor **)malloc(sizeof(TF_Tensor *) * NumInputs);
//     TF_Tensor **OutputValues = (TF_Tensor **)malloc(sizeof(TF_Tensor *) * NumOutputs);

//     int ndims = 2;
//     int64_t dims[] = {1, 30};
//     float data[1 * 30]; //= {1,1,1,1,1,1,1,1,1,1};
//     for (int i = 0; i < (1 * 30); i++)
//     {
//         data[i] = 1.00;
//     }
//     int ndata = sizeof(float) * 1 * 30; // This is tricky, it number of bytes not number of element

//     TF_Tensor *int_tensor = TF_NewTensor(TF_FLOAT, dims, ndims, data, ndata, &NoOpDeallocator, 0);
//     if (int_tensor != NULL)
//     {
//         printf("TF_NewTensor is OK\n");
//     }
//     else
//         printf("ERROR: Failed TF_NewTensor\n");

//     InputValues[0] = int_tensor;

//     // //Run the Session
//     TF_SessionRun(Session, NULL, Input, InputValues, NumInputs, Output, OutputValues, NumOutputs, NULL, 0, NULL, Status);

//     if (TF_GetCode(Status) == TF_OK)
//     {
//         printf("Session is OK\n");
//     }
//     else
//     {
//         printf("%s", TF_Message(Status));
//     }

//     // //Free memory
//     TF_DeleteGraph(Graph);
//     TF_DeleteSession(Session, Status);
//     TF_DeleteSessionOptions(SessionOpts);
//     TF_DeleteStatus(Status);

//     void *buff = TF_TensorData(OutputValues[0]);
//     float *offsets = (float *)buff;
//     printf("Result Tensor :\n");
//     for (int i = 0; i < 10; i++)
//     {
//         printf("%f\n", offsets[i]);
//     }
// }