//ver 1.0 
//Copy right Blue Vision LLC. 2024

#include "LoFTRMatching.h" // 假设这是你的LoFTRMatching类的头文件  
#include <torch/script.h>
#include <torch/torch.h>

#include <chrono>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <utility>
#include <opencv2/opencv.hpp>
#include "LoFTRMatching.h"
    
LoFTRMatching::LoFTRMatching(const char* configFile) {  
	// 初始化模型和其他资源  
	if (!Initialize(configFile)) {  
		// 初始化失败，可能需要抛出一个异常或设置错误状态  
		throw std::runtime_error("Failed to initialize LoFTRMatching");  
	}  
}  


bool LoFTRMatching::Initialize(const char* sConfigFile)  {  
    std::cout << "start loftr match..." << std::endl;  
    torch::manual_seed(1);  
    torch::autograd::GradMode::set_enabled(false);  
    torch::Device device(torch::kCPU);  
    if (torch::cuda::is_available()) {  
        std::cout << "CUDA is available! Training on GPU." << std::endl;  
        device = torch::Device(torch::kCUDA);  
    }  
    // 假设executable_dir是包含模型文件的目录  
    std::string module_path = sConfigFile; 
  
    try {  
        // 尝试加载模型  
        eLoFTRModel = torch::jit::load(module_path);  
        // 确保模型已经加载到正确的设备上  
        eLoFTRModel.to(device);  
   
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

bool UpdateMatchingRelation() {  
	// 确保sourceImage和targetImage是有效的cv::Mat对象  
	if (sourceImage.empty() || targetImage.empty()) {  
		// 处理空图像的情况  
		return false;  
	}  

	// 调整图像大小（如果需要）  
	cv::resize(sourceImage, sourceImage, cv::Size(800, 800));  
	cv::resize(targetImage, targetImage, cv::Size(800, 800));  

	// 将图像转换为tensor  
	torch::Tensor image0 = mat2tensor(sourceImage).to(device);  
	torch::Tensor image1 = mat2tensor(targetImage).to(device);  

	// 创建一个包含输入图像的列表  
	std::vector<torch::jit::IValue> inputs;  
	inputs.push_back(image0);  
	inputs.push_back(image1);  

	// 调用LoFTR网络进行前向传播  
	torch::IValue output = eLoFTRModel.forward(inputs);  

	// 提取返回的字典并转换为torch::Dict<std::string, torch::Tensor>  
	matchResults = toTensorDict(output); // 假设matchResults是torch::Dict<std::string, torch::Tensor>类型的成员变量  

	return true;  
}  

bool LoFTRMatching::UpdateMatchingRelation(){
    cv::resize(img0, img0, cv::Size(800,800));
    cv::resize(img1, img1, cv::Size(800,800));
    
    //convert to tensor
    torch::Tensor image0 = mat2tensor(img0).to(device);
    torch::Tensor image1 = mat2tensor(img1).to(device);
    
    //dump to loftr network
    torch::Dict<std::string, Tensor> input;
    
    input.insert("image0", image0);
    input.insert("image1", image1);
    matchingRelation = toTensorDict(loftr.forward({input}));
    return true;
}
torch::Dict<std::string, Tensor> toTensorDict(const torch::IValue &value) {
  return c10::impl::toTypedDict<std::string, Tensor>(value.toGenericDict());
}
bool LoFTRMatching::GetMatchedData(std::vector<MatchedPoints>& vData) {  
    // 假设matchingRelation是torch::Dict<std::string, torch::Tensor>类型的成员变量  
    if (matchingRelation.empty()) {  
        // 如果matchingRelation为空（例如，尚未调用UpdateMatchingRelation），则返回false  
        return false;  
    }  
  
    // 从matchingRelation字典中提取tensor  
    at::Tensor mkpts0_f = matchingRelation.at("mkpts0_f"); // 假设是N*2的tensor，包含N个点的(x, y)坐标  
    at::Tensor mkpts1_f = matchingRelation.at("mkpts1_f"); // 假设是N*2的tensor，包含N个点的(x, y)坐标  
    at::Tensor mconf = matchingRelation.at("mconf"); // 假设是N*1的tensor，包含N个匹配的置信度  
  
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
  
    // 创建一个MatchedPoints实例来存储数据  
    MatchedPoints data;  
  
    // 遍历tensor并将数据添加到MatchedPoints实例中  
    for (int64_t i = 0; i < N; ++i) {  
        // 提取点坐标和置信度  
        data.points0.push_back(mkpts0_f[i][0].item<float>()); // x坐标  
        data.points0.push_back(mkpts0_f[i][1].item<float>()); // y坐标  
        data.points1.push_back(mkpts1_f[i][0].item<float>()); // x坐标  
        data.points1.push_back(mkpts1_f[i][1].item<float>()); // y坐标  
        data.confidences.push_back(mconf[i].item<float>()); // 置信度  
    }  
  
    // 将MatchedPoints实例添加到输出向量中  
    vData.push_back(data);  
  
    // 成功获取匹配数据，返回true  
    return true;  
}

int main(int argc, char** argv) {  
    
    // 假设的图片路径和模型路径（你可以直接给出这些值）  
    const char* img1_path = "./myimg1.jpg";  
    const char* img2_path = "./myimg2.jpg";  
    const char* model_path = "./traced_eloftr_model.zip";  
    
    std::cout << img1_path << img2_path << model_path << "\n";
  
    // 实例化LoFTRMatching对象  
    LoFTRMatching loftrMatcher(model_path);  
  
    // 调用MatchImages方法并检查结果  
    
    //if(loftrMatcher.SetSourceImage(img1_path)){
    //    std::cout<<"success" << std::endl;
    //}
    
    
    //bool success = loftrMatcher.MatchImages(img1_path, img2_path, model_path);  
  
  
    // 如果你的MatchImages方法返回匹配结果，你可以在这里添加更多的断言来验证结果  
    // ...  ();  
}
