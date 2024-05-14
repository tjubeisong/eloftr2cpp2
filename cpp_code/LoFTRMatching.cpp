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
#include "LoFTRMatching.h"
    

#include "viz.h"
#include "io.h"


LoFTRMatching::LoFTRMatching() { }  
LoFTRMatching::~LoFTRMatching() { }  

LoFTRMatching::LoFTRMatching(const char* configFile) {  
	// 初始化模型和其他资源  
	if (!Initialize(configFile)) {  
		// 初始化失败，可能需要抛出一个异常或设置错误状态  
		throw std::runtime_error("Failed to initialize LoFTRMatching");  
	}  
}  

torch::Tensor LoFTRMatching::mat2tensor(cv::Mat image){
    int target_width = image.cols;
    int target_height = image.rows;
    image.convertTo(image, CV_32F, 1.0f / 255.0f);
    torch::Tensor tensor = torch::from_blob(image.data, {1, 1, image.rows, image.cols},
	                                  torch::TensorOptions().dtype(torch::kFloat32));
    return tensor.clone();
}

bool LoFTRMatching::Initialize(const char* sConfigFile)  {  
    std::cout << "start loftr match..." << std::endl;  
    
    torch::manual_seed(1);  
    torch::autograd::GradMode::set_enabled(false);  
    
    if (torch::cuda::is_available()) {  
        std::cout << "CUDA is available! Training on GPU." << std::endl;  
        mdevice = torch::Device(torch::kCUDA);  
    }  
    // 假设executable_dir是包含模型文件的目录  
    std::string module_path = sConfigFile; 
  
    try {  
        // 尝试加载模型  
        eLoFTRModel = torch::jit::load(module_path);  
        // 确保模型已经加载到正确的设备上  
        eLoFTRModel.to(mdevice);  
   
        eLoFTRModel.eval();  
        // 初始化成功  
    } catch (const c10::Error& e) {  
        // 处理libtorch抛出的异常  
        std::cerr << "Error loading the model: " << e.what() << std::endl;  
        return false;  
    } catch (const std::exception& e) {  
        // 处理其他可能的异常  
        std::cerr << "An error occurred: " << e.what() << std::endl;  
        return false;  
    }  
    return true; 
}


bool LoFTRMatching::SetSourceImage(const char* sImageFile) {  
	// 使用OpenCV读取图像  
	sourceImage = cv::imread(sImageFile, cv::IMREAD_GRAYSCALE); // 假设我们处理灰度图像  
	if (sourceImage.empty()) {  
		// 读取失败  
		std::cerr << "Error: Unable to load image from file: " << sImageFile << std::endl;  
		return false;  
	}  

	// 可以在这里添加图像预处理步骤，如缩放、去噪等  

	// 设置成功  
	return true;  
} 

bool LoFTRMatching::SetTargetImage(const char* sImageFile) {
	// 使用OpenCV读取图像  
	targetImage = cv::imread(sImageFile, cv::IMREAD_GRAYSCALE); // 假设我们处理灰度图像  
	if (targetImage.empty()) {  
		// 读取失败  
		std::cerr << "Error: Unable to load image from file: " << sImageFile << std::endl;  
		return false;  
	}  

	// 可以在这里添加图像预处理步骤，如缩放、去噪等  

	// 设置成功  
	return true;
}

bool LoFTRMatching::UpdateMatchingRelation() {  
	// 确保sourceImage和targetImage是有效的cv::Mat对象  
	if (sourceImage.empty() || targetImage.empty()) {  
		// 处理空图像的情况  
		return false;  
	}  

        // 调整图像大小（如果需要）  
	cv::resize(sourceImage, sourceImage, cv::Size(800, 800));  
	cv::resize(targetImage, targetImage, cv::Size(800, 800));  
        
        // 将图像转换为tensor  
	torch::Tensor image0 = mat2tensor(sourceImage).to(mdevice);  
	torch::Tensor image1 = mat2tensor(targetImage).to(mdevice);  
        // 创建一个包含输入图像的列表  

        torch::Dict<std::string, torch::Tensor> input;

        input.insert("image0", image0);
        input.insert("image1", image1);
	// 调用LoFTR网络进行前向传播  
	matchingRelation = toTensorDict(eLoFTRModel.forward({input})); 

	// 提取返回的字典并转换为torch::Dict<std::string, torch::Tensor>  
	//matchingRelation = toTensorDict(output); // 假设matchResults是torch::Dict<std::string, torch::Tensor>类型的成员变量  

	return true;  
}  


bool LoFTRMatching::GetMatchedData(std::vector<float>& data) {  
    // 假设matchingRelation是torch::Dict<std::string, torch::Tensor>类型的成员变量  
    if (matchingRelation.empty()) {  
        // 如果matchingRelation为空（例如，尚未调用UpdateMatchingRelation），则返回false  
        return false;  
    }  
  
    // 从matchingRelation字典中提取tensor  
    mkpts0_f = matchingRelation.at("mkpts0_f"); // 假设是N*2的tensor，包含N个点的(x, y)坐标  
    mkpts1_f = matchingRelation.at("mkpts1_f"); // 假设是N*2的tensor，包含N个点的(x, y)坐标  
    mconf = matchingRelation.at("mconf"); // 假设是N*1的tensor，包含N个匹配的置信度  
  
    // 检查tensor的维度是否正确  
    if (mkpts0_f.dim() != 2 || mkpts0_f.size(1) != 2 ||  
        mkpts1_f.dim() != 2 || mkpts1_f.size(1) != 2 ||  
        mconf.dim() != 1) {  
        // 如果tensor的维度不正确，则返回false  
        return false;  
    }  
  
    // 获取tensor的大小（点数N）  
    int64_t N = mkpts0_f.size(0);  
  
    // 确保三个tensor的点数N相同  
    if (mkpts1_f.size(0) != N || mconf.size(0) != N) {  
        return false;  
    }  
  
    data.clear();
    data.push_back(N);
    // 遍历tensor并将数据添加到list
    for (int64_t i = 0; i < N; ++i) {  
        data.push_back(mkpts0_f[i][0].item<float>()); // x坐标  
        data.push_back(mkpts0_f[i][1].item<float>()); // y坐标  
    }  
    
    for (int64_t i = 0; i < N; ++i) {  
        data.push_back(mkpts1_f[i][0].item<float>()); // x坐标  
        data.push_back(mkpts1_f[i][1].item<float>()); // y坐标  
    }  
    
    for (int64_t i = 0; i < N; ++i) {  
        data.push_back(mconf[i].item<float>()); // 置信度  
    } 
  
    // 成功获取匹配数据，返回true  
    return true;  
}


int main(int argc, char** argv) {  
    
    // 假设的图片路径和模型路径（你可以直接给出这些值）  
    const char* img1_path = "../myimg1.jpg";  
    const char* img2_path = "../myimg2.jpg";  
    const char* model_path = "../traced_eloftr_model.zip";  
    
    // 实例化LoFTRMatching对象  
    LoFTRMatching loftrMatcher(model_path);  
  
    // 调用MatchImages方法并检查结果  
    
    if(loftrMatcher.SetSourceImage(img1_path)){
        std::cout<<"set source image succeed" << std::endl;
    }
    
    
    if(loftrMatcher.SetTargetImage(img2_path)){
        std::cout<<"set target image succeed" << std::endl;
    }
    
    if(loftrMatcher.UpdateMatchingRelation()){
    	std::cout<<"Update matching relation succeed" << std::endl;
    }  
    
    std::vector<float> data;
    data.clear();
    if(loftrMatcher.GetMatchedData(data)){
    	std::cout<<"get matched data succeed" << std::endl;
    	
    	int N = int(data[0]);
    	std::cout << "totally " << N << " matched points" << std::endl;
    }
    
    bool is_visualize = true;
    if(is_visualize)
    {
        //matching visualization
        cv::Mat plot =
          MakeMatchingPlotFast(loftrMatcher.sourceImage, loftrMatcher.targetImage, loftrMatcher.mkpts0_f, loftrMatcher.mkpts1_f, loftrMatcher.mconf, true, 10);
        cv::imwrite("../matches.png", plot);
        std::cout << "Done! Created matches.png for visualization." << std::endl;

        //test EpiLines
        DrawEpiLinesFast(loftrMatcher.sourceImage, loftrMatcher.targetImage, loftrMatcher.mkpts0_f, loftrMatcher.mkpts1_f, "../");
    }
    //save kpts to file
    save_kpts_2_txt(loftrMatcher.mkpts0_f, "kpts0", "../");
    save_kpts_2_txt(loftrMatcher.mkpts1_f, "kpts1", "../");
    return 0; 
}
