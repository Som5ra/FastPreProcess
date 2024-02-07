# FastPreProcess

- Implemented: 
    - hwc to chw (slower than numpy.transpose)
    - hwc to chw with normalization
    - chw channel normalization


 - Requirements:
    - pybind11
    - numpy
    - opencv

- Build:
```
mkdir -p build
cmake --build ./build --config Release --target all -j 34 --
```