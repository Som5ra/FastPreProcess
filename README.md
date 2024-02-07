# FastPreProcess

### Test Envirnoment

- Ubuntu 22.04
- Intel i9-13900K
- Python 3.9



### Implemented

1. **hwc to chw (slower than numpy.transpose)**: array(768, 1024, 3) -> array(3, 768, 1024)
2. **chw channel normalization**: e.g. array(3, 768, 1024) -> (array(3, 768, 1024) - MEAN) / STD
3. **hwc to chw with normalization (the fastest)**: **e.g. array(768, 1024, 3) -> (array(3, 768, 1024) - MEAN) / STD**

### Performance

- Iteration / econd
- MIN / MAX / AVG

|      Batch Num     |        Python       |          Cpp           |      Cpp (batched)     | 
|--------------------|---------------------|------------------------|------------------------|
|         32         | 23.67 / 24.43 / 24.11 | 203.49 / 343.20 / 245.02 | 207.02 / 266.26 / 228.76 |


### Requirements

1. pybind11
2. numpy
3. opencv



### Build

```
mkdir -p build
cmake --build ./build --config Release --target all -j 34 --
```