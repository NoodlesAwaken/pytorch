#include <torch/script.h> // One-stop header.
#include "color.hpp"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <iostream>
#include <memory>
#include <time.h>

// NYU undistorted images have a white border which effects output
// use these parameters to make it black border
#define INPUT_BORDER_CROP_X 20
#define INPUT_BORDER_CROP_Y 20



int main(int argc, const char* argv[]) {
	if (argc != 3) {
		std::cerr << "usage: pytorch <path-to-exported-script-module> <path-to-input-image>\n";
	    	return -1;
	}

	// read image and normalize
	cv::Mat img = cv::imread(argv[2], cv::IMREAD_UNCHANGED);
	cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
	//cv::Size rsz = {640, 480};
	//cv::resize(img, img, rsz, 0, 0, cv::INTER_LINEAR);
	img.convertTo(img, CV_32FC3, 1/255.f);


	// for NYU dataset has white border
	//cv::Rect border(cv::Point(0, 0), img.size());
	//cv::rectangle(img, border, cv::Scalar(0, 0, 0), std::max(INPUT_BORDER_CROP_X, INPUT_BORDER_CROP_Y));


	auto options = torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).requires_grad(false).device(torch::kCPU);
	at::Tensor tensor_img = torch::from_blob(img.data, {1, img.rows, img.cols, 3}, options);

	tensor_img = tensor_img.permute({0, 3, 1, 2});
	tensor_img = tensor_img.to(torch::kCUDA);
	tensor_img[0][0] = tensor_img[0][0].sub(0.485).div(0.229);
	tensor_img[0][1] = tensor_img[0][1].sub(0.456).div(0.224);
	tensor_img[0][2] = tensor_img[0][2].sub(0.406).div(0.225);


	// prepare input
	std::vector<torch::jit::IValue> inputs;
	inputs.push_back(tensor_img);

	// show normalized image
	//cv::imshow("Prepared", img);
	//cv::waitKey(0);

	// load model from pytorch script
	torch::jit::script::Module module = torch::jit::load(argv[1]);
	module.to(at::kCUDA);

	std::cout << "module load ok\n";

	auto start = std::clock();

	auto outputs = module.forward(inputs).toTuple();

	std::cout << "processing time: " << (float)(std::clock() - start)/CLOCKS_PER_SEC << "\n";

	torch::Tensor segmentation = outputs->elements().at(0).toTensor().contiguous();
	torch::Tensor depth = outputs->elements().at(1).toTensor().contiguous();
	//torch::Tensor normals = outputs->elements().at(1).toTensor().contiguous();


	torch::Tensor segmentation_cpu = segmentation.argmax(1).to(torch::kCUDA);
	torch::Tensor depth_cpu = depth.to(torch::kCUDA);
	//torch::Tensor normals_cpu = normals.to(torch::kCPU);

	const int Hout = depth_cpu.size(2);
	const int Wout = depth_cpu.size(3);

	//cv::Mat normalMapPredFloatTmp(Hout, Wout, CV_32FC4, cv::Scalar(0, 0, 0, 0));
	cv::Mat depthMapPredTmp(Hout, Wout, CV_32FC1, cv::Scalar(0));
	cv::Mat segMapPredTmp(Hout, Wout, CV_8UC3, cv::Scalar(0, 0, 0));

	//float * normalMapPointer = (float *) normalMapPredFloatTmp.data;
	float * depthMapPointer = (float *) depthMapPredTmp.data;
	unsigned char * segMapPointer = (unsigned char*) segMapPredTmp.data;
	const int stepSize = Hout * Wout;

	for (auto i = 0; i < stepSize; i++) {
		//const int v = i / Wout;
		//const int u = i % Wout;

		//float Nx = normals_cpu.data_ptr<float>()[i];
		//float Ny = normals_cpu.data_ptr<float>()[i + stepSize];
		//float Nz = normals_cpu.data_ptr<float>()[i + (2 * stepSize)];
		//float denom = sqrtf(Nx * Nx + Ny * Ny + Nz * Nz);
		//Nx /= denom;
		//Ny /= denom;
		//Nz /= denom;

		float depth = depth_cpu.data_ptr<float>()[i];

		//normalMapPointer[4 * i] = (1.f - Nz) / 2.f;
		//normalMapPointer[4 * i + 1] = (1.f + Ny) / 2.f;
		//normalMapPointer[4 * i + 2] = (1.f + Nx) / 2.f;

		//if (Nz > 0)
		//	normalMapPointer[4 * i + 3] = 0.f;
		//else
		//	normalMapPointer[4 * i + 3] = 1.f;

		if (depth < 0)
			depth = 0.f;

		depthMapPointer[i] = depth;
		int segIdx = (int) segmentation_cpu.data_ptr<long>()[i];
		segMapPointer[3 * i] = (unsigned char) r_values[segIdx + 1];
		segMapPointer[3 * i + 1] = (unsigned char) g_values[segIdx + 1];
		segMapPointer[3 * i + 2] = (unsigned char) b_values[segIdx + 1];
		//segMapPointer[i] = segIdx + 1;
	}

	//cv::imshow("viewer", segMapPredTmp);
	//cv::waitKey(0);

	//cv::Mat normalMap(img.rows, img.cols, CV_32FC4);
	cv::Mat depthMap(img.rows, img.cols, CV_32FC1, cv::Scalar(0));
	cv::Mat semanticMap(img.rows, img.cols, CV_8UC3, cv::Scalar(0, 0, 0));

	//cv::resize(normalMapPredFloatTmp, normalMap, normalMap.size(), 0, 0, cv::INTER_NEAREST);
	cv::resize(depthMapPredTmp, depthMap, depthMap.size(), 0, 0, cv::INTER_LINEAR);
	cv::resize(segMapPredTmp, semanticMap, semanticMap.size(), 0, 0, cv::INTER_NEAREST);
	cv::imshow("NEAREST", semanticMap);

	cv::resize(segMapPredTmp, semanticMap, semanticMap.size(), 0, 0, cv::INTER_LINEAR);
	cv::imshow("LINEAR", semanticMap);

	cv::resize(segMapPredTmp, semanticMap, semanticMap.size(), 0, 0, cv::INTER_CUBIC);
	cv::imshow("CUBIC", semanticMap);

	cv::resize(segMapPredTmp, semanticMap, semanticMap.size(), 0, 0, cv::INTER_AREA);
	cv::imshow("AREA", semanticMap);


	cv::resize(segMapPredTmp, semanticMap, semanticMap.size(), 0, 0, cv::INTER_LANCZOS4);
	cv::imshow("LANCZOS4", semanticMap);

	cv::resize(segMapPredTmp, semanticMap, semanticMap.size(), 0, 0, cv::INTER_LINEAR_EXACT);
	//cv::imshow("LINEAR EXACT", semanticMap);
	//cv::waitKey(0);

	//cv::imshow("Normal Map", normalMap);
	
	double minv, maxv;
	cv::minMaxLoc(depthMap, &minv, &maxv);
	std::cerr << "Max D: " << maxv << ", Min D: " << minv << std::endl;
	cv::Mat depthMapTmpVis = (depthMap - minv) / (maxv - minv);
	//cv::imshow("Depth Map", depthMapTmpVis);

	return 0;
}

