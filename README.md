# FastPreProcess

### Test Envirnoment

- Ubuntu 22.04
- i9-13900K
- Python 3.9



### Implemented:

1. **hwc to chw (slower than numpy.transpose)**: array(768, 1024, 3) -> array(3, 768, 1024)
2. **chw channel normalization**: e.g. array(3, 768, 1024) -> (array(3, 768, 1024) - MEAN) / STD
3. **hwc to chw with normalization (the fastest)**: **e.g. array(768, 1024, 3) -> (array(3, 768, 1024) - MEAN) / STD**

### Requirements

1. pybind11
2. numpy
3. opencv



### Build

```
mkdir -p build
cmake --build ./build --config Release --target all -j 34 --
```