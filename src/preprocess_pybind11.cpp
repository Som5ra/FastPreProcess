// #include <opencv2/opencv.hpp>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "preprocess.cpp"
namespace py = pybind11;

template<typename in_t, typename out_t> py::array_t<out_t> hwc_to_chw(py::array_t<in_t> input, bool flip_rb){
    auto info = input.request();
    in_t* input_array = (in_t *)info.ptr;
    size_t input_shape[] = {info.shape[0], info.shape[1], info.shape[2]};

    out_t *result = new out_t[info.shape[0] * info.shape[1] * info.shape[2]];
    _hwc_to_chw(input_array, input_shape, result, flip_rb);

    py::capsule free_when_done(result, [](void *f) {
        delete (out_t*)f;
    });
    return py::array_t<out_t>(
        {info.shape[2], info.shape[0], info.shape[1]},
        result,
        free_when_done
    );
}

template<typename in_t, typename out_t> py::array_t<out_t> hwc_to_chw_normalize(py::array_t<in_t> input, std::vector<out_t> mean, std::vector<out_t> std, bool flip_rb){
    auto info = input.request();
    in_t* input_array = (in_t *)info.ptr;
    size_t input_shape[] = {info.shape[0], info.shape[1], info.shape[2]};

    out_t *result = new out_t[info.shape[0] * info.shape[1] * info.shape[2]];
    _hwc_to_chw_normalize(input_array, input_shape, mean.data(), std.data(), result, flip_rb);

    py::capsule free_when_done(result, [](void *f) {
        delete (out_t*)f;
    });
    return py::array_t<out_t>(
        {info.shape[2], info.shape[0], info.shape[1]},
        result,
        free_when_done
    );
}

template<typename in_t, typename out_t> py::array_t<out_t> chw_channel_normalize(py::array_t<in_t> input, std::vector<out_t> mean, std::vector<out_t> std, bool flip_rb){
    auto info = input.request();
    in_t* input_array = (in_t *)info.ptr;
    size_t input_shape[] = {info.shape[0], info.shape[1], info.shape[2]};

    out_t *result = new out_t[info.shape[0] * info.shape[1] * info.shape[2]];
    _chw_channel_normalize(input_array, input_shape, mean.data(), std.data(), result);

    py::capsule free_when_done(result, [](void *f) {
        delete (out_t*)f;
    });
    return py::array_t<out_t>(
        {info.shape[0], info.shape[1], info.shape[2]},
        result,
        free_when_done
    );
}



PYBIND11_MODULE(fastpreprocess, m) {
    m.def("hwc_to_chw", &hwc_to_chw<uint8_t, float_t>, py::arg("input"), py::arg("flip_rb") = false, py::return_value_policy::take_ownership);
    m.def("hwc_to_chw_normalize", &hwc_to_chw_normalize<uint8_t, float_t>, py::arg("input"), py::arg("mean"), py::arg("std"), py::arg("flip_rb") = false, py::return_value_policy::take_ownership);
    m.def("chw_channel_normalize", &chw_channel_normalize<uint8_t, float_t>, py::arg("input"), py::arg("mean"), py::arg("std"), py::arg("flip_rb") = false, py::return_value_policy::take_ownership);
}