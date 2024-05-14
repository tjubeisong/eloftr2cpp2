//ver 1.0 
//Copy right Blue Vision LLC. 2024
#include <torch/script.h>
#include <torch/torch.h>

#include <chrono>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <utility>
#include <opencv2/opencv.hpp>
#pragma once
#include <vector>

//structure to save matched data points
struct MatchedPoints
{
	//source image coordinates
	double x0, y0;

	//source image coodinates
	double x1, fY1;
};

//using LoFTR algorithm to match two images



class LoFTRMatching
{

public:
	cv::Mat sourceImage;
	cv::Mat targetImage;
	torch::jit::script::Module eLoFTRModel;
	torch::Dict<std::string, torch::Tensor> matchingRelation;
	torch::Device mdevice = torch::Device(torch::kCPU);
	
	at::Tensor mkpts0_f;
	at::Tensor mkpts1_f;
	at::Tensor mconf;
	
	
public:
	
	//contructor 
	LoFTRMatching();
	LoFTRMatching(const char* configFile);
	//Desctrucor
	~LoFTRMatching();

	torch::Tensor mat2tensor(cv::Mat image);
	
	//read configuration file and initialize object to set the project in function operational status.
	//the format of the configuaration file is TBD (To Be Determined), recommend using XML format and TinyXML parser
	// sConfgFile, input, configuaration file full path name
	// return, true if the initialization success
	bool Initialize(const char* sConfigFile);

	//Set source image from a image file into object and collect/generate keypoint and description information
	//sImageFile, input, image file full path name
	//return, true if setting success
	bool SetSourceImage(const char* sImageFile);

	//Set source image from a opencv image matrix into object and collect/generate keypoint and description information
	//oImage, input, opencv image 
	//return, true if the setting  success
	//bool SetSouceImage(const cv::Mat& oImage);    **** NOT implementation now

	//Set target image from a image file into object
	//sImageFile, input, image file full path name
	//return, true if setting success
	bool SetTargetImage(const char* sImageFile);

	//Set target image from a opencv image matrix into object
	//oImage, input, opencv image
	//return, true if the setting  success
	//bool SetTargetImage(const cv::Mat& oImage);    **** NOT implementation now

	//Update matching relation uing the known soure and target image information
	//retun, true if the Updating success. If success, matched data for source and target image will be available
	bool UpdateMatchingRelation();

	//Get matchend data
	//vData, output, the serial matched point coordinate
	//return, true if the matched data available, otherwise the vData is kept no touch in function call
	bool GetMatchedData(std::vector<float>& data);
	
	
	torch::Dict<std::string, torch::Tensor> toTensorDict(const torch::IValue &value) {
	    return c10::impl::toTypedDict<std::string, torch::Tensor>(value.toGenericDict());
	}




};

