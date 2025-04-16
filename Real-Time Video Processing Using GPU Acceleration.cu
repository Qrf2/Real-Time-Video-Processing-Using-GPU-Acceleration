#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>

#define BLOCK_SIZE 16

// CUDA kernel for grayscale conversion
__global__ void rgbToGrayKernel(unsigned char* input, unsigned char* output, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * channels;
        unsigned char r = input[idx];
        unsigned char g = input[idx + 1];
        unsigned char b = input[idx + 2];
        output[y * width + x] = 0.299f * r + 0.587f * g + 0.114f * b;
    }
}

// CUDA kernel for object detection simulation (dummy logic)
__global__ void detectObjectsKernel(unsigned char* grayImage, int* detectionMap, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        detectionMap[idx] = (grayImage[idx] > 128) ? 1 : 0; // Dummy logic for detection
    }
}

// Host function to process video frames
void processVideo(const std::string& videoPath) {
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        std::cerr << "Error: Unable to open video file." << std::endl;
        return;
    }

    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int channels = 3;

    size_t imageSize = width * height * channels * sizeof(unsigned char);
    size_t grayImageSize = width * height * sizeof(unsigned char);
    size_t detectionMapSize = width * height * sizeof(int);

    // Allocate memory on the host and device
    unsigned char *h_inputImage = (unsigned char*)malloc(imageSize);
    unsigned char *d_inputImage, *d_grayImage;
    int *d_detectionMap;

    cudaMalloc((void**)&d_inputImage, imageSize);
    cudaMalloc((void**)&d_grayImage, grayImageSize);
    cudaMalloc((void**)&d_detectionMap, detectionMapSize);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cv::Mat frame;
    while (cap.read(frame)) {
        // Copy frame data to host memory
        memcpy(h_inputImage, frame.data, imageSize);

        // Copy data to device memory
        cudaMemcpy(d_inputImage, h_inputImage, imageSize, cudaMemcpyHostToDevice);

        // Launch kernels
        rgbToGrayKernel<<<gridSize, blockSize>>>(d_inputImage, d_grayImage, width, height, channels);
        detectObjectsKernel<<<gridSize, blockSize>>>(d_grayImage, d_detectionMap, width, height);

        // Copy detection map back to host (optional for visualization)
        int* h_detectionMap = (int*)malloc(detectionMapSize);
        cudaMemcpy(h_detectionMap, d_detectionMap, detectionMapSize, cudaMemcpyDeviceToHost);

        // Visualization (dummy logic)
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                if (h_detectionMap[y * width + x] == 1) {
                    cv::rectangle(frame, cv::Point(x, y), cv::Point(x + 2, y + 2), cv::Scalar(0, 255, 0), -1);
                }
            }
        }

        cv::imshow("Processed Frame", frame);
        if (cv::waitKey(30) >= 0) break;

        free(h_detectionMap);
    }

    // Free resources
    free(h_inputImage);
    cudaFree(d_inputImage);
    cudaFree(d_grayImage);
    cudaFree(d_detectionMap);
}

int main() {
    std::string videoPath = "video.mp4";
    processVideo(videoPath);
    return 0;
}
