#include <gflags/gflags.h>
#include <glog/logging.h>
#include <chrono>
#include <vector>
#include <numeric>
#include <functional>
#include <iostream>
#include <cstring>
#include "paddle/include/paddle_inference_api.h"

DEFINE_string(dirname, "../mobilenetv1", "Directory of the inference model.");
DEFINE_bool(use_gpu, true, "whether use gpu");
DEFINE_bool(use_mkldnn, true, "whether use mkldnn");
DEFINE_int32(batch_size, 1, "batch size of inference model");
DEFINE_int32(repeat_time, 100, "repeat time for inferencing.");

namespace paddle_infer{

    using Time = decltype(std::chrono::high_resolution_clock::now());
    Time time() {return std::chrono::high_resolution_clock::now();};

    double time_diff(Time t1, Time t2){
        typedef std::chrono::microseconds ms;
        auto diff = t2 - t1;
        ms counter = std::chrono::duration_cast<ms>(diff);
        return counter.count() / 1000.0;
    }

    void PrepareTRTConfig(Config *config){
        config->SetModel(FLAGS_dirname + "/model", FLAGS_dirname + "/params");
        if(FLAGS_use_gpu){
            config->EnableUseGpu(1000, 0); // gpu:0
        }else{
            config->DisableGpu();
            if(FLAGS_use_mkldnn) config->EnableMKLDNN();
        }
        
        config->SwitchUseFeedFetchOps(false);
        config->SwitchIrOptim(false);  // close all optimization
    }


    bool test_predictor_latency(){
        int batch_size = FLAGS_batch_size;
        int repeat = FLAGS_repeat_time;
        // 1. create Config
        Config config;
        PrepareTRTConfig(&config);

        // 2. create Predictor
        auto predictor = CreatePredictor(config);

        // 3. prepare input
        int channels = 3;
        int height = 224;
        int width = 224;
        int input_num = channels * height * width * batch_size;
        float input[input_num] = {0};

        // 4. feed data
        auto input_names = predictor->GetInputNames();
        auto input_t = predictor->GetInputHandle(input_names[0]);
        input_t->Reshape({batch_size, channels, height, width});
        input_t->CopyFromCpu(input);

        // 5. warmup
        int warmup = 10;
        for(int i=0;i < warmup; ++i){
            CHECK(predictor->Run());
        }

        // 6. test latency
        auto start_t = time();
        for(size_t i=0; i < repeat; ++i){
            predictor->Run();

            // fetch output
            std::vector<float> out_data;
            auto output_names = predictor->GetOutputNames();
            auto output_t = predictor->GetOutputHandle(output_names[0]);
            std::vector<int> output_shape = output_t->shape();
            int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());
            out_data.resize(out_num);
            output_t->CopyToCpu(out_data.data());
        }
        auto end_t = time();
        std::cout << "repeat time: " << FLAGS_repeat_time  << " , model: " << FLAGS_dirname << std::endl;
        std::cout << "batch: " << batch_size << " , predict cost: " << time_diff(start_t, end_t) / static_cast<float>(repeat) << " ms." << std::endl;
        
        return true;

    }
};

int main(int argc, char** argv){
    google::ParseCommandLineFlags(&argc, &argv, true);
    paddle::test_predictor_latency();
    return 0;
}