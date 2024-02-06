#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <omp.h>

namespace py = pybind11;

template<typename in_t, typename out_t> void _hwc_to_chw(in_t* input, size_t* input_shape, out_t* ret, int num_threads = -1){
    if (num_threads <= 0)
        num_threads = std::max(1, omp_get_num_procs() / 2);
    
    size_t hw = input_shape[0] * input_shape[1];
    size_t num_ch = input_shape[2];
    #pragma omp parallel for num_threads(num_threads)
    for(size_t stride = 0; stride < hw; stride++){
        for(size_t ch = 0; ch < num_ch; ch++){
            ret[hw * ch + stride] = (out_t)input[stride * num_ch + ch];
        }
    }
    return ;
}


template<typename in_t, typename out_t> void _hwc_to_chw_normalize(in_t* input, size_t* input_shape, out_t* mean, out_t* std, out_t* ret, int num_threads = -1){

    if (num_threads <= 0)
        num_threads = std::max(1, omp_get_num_procs() / 2);
    size_t hw = input_shape[0] * input_shape[1];
    size_t num_ch = input_shape[2];
    #pragma omp parallel for num_threads(num_threads)
    for(size_t stride = 0; stride < input_shape[0] * input_shape[1]; stride++){
        for(size_t ch = 0; ch < num_ch; ch++){
            ret[hw * ch + stride] = (out_t)((input[stride * num_ch + ch] - mean[ch]) / std[ch]);
        }
        // ret[hw * 0 + stride] = (out_t)((input[stride * 3 + 0] - mean[0]) / std[0]);
        // ret[hw * 1 + stride] = (out_t)((input[stride * 3 + 1] - mean[1]) / std[1]);
        // ret[hw * 2 + stride] = (out_t)((input[stride * 3 + 2] - mean[2]) / std[2]);
    }
    return ;
}

template<typename in_t, typename out_t> void _chw_channel_normalize(in_t* input, size_t* input_shape, out_t* mean, out_t* std, out_t* ret, int num_threads = -1){

    if (num_threads <= 0)
        num_threads = std::max(1, omp_get_num_procs() / 2);
    size_t hw = input_shape[1] * input_shape[2];
    size_t num_ch = input_shape[0];
    #pragma omp parallel for num_threads(num_threads)
    for(size_t stride = 0; stride < hw; stride++){
        for(size_t ch = 0; ch < num_ch; ch++){
            // ret[stride * 3 + 0] = (out_t)((input[stride * 3 + 0] - mean[0]) / std[0]);
            // ret[stride * 3 + 1] = (out_t)((input[stride * 3 + 1] - mean[1]) / std[1]);
            // ret[stride * 3 + 2] = (out_t)((input[stride * 3 + 2] - mean[2]) / std[2]);
            ret[hw * ch + stride] = (out_t)((input[hw * ch + stride] - mean[ch]) / std[ch]);
        }
    }
    return ;
}