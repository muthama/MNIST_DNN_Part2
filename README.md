# MNIST_DNN_Part2

## Repository Description
This repository contains the optimized implementation of a deep neural network (DNN) for handwritten digit recognition, developed from scratch in C++. It builds upon the vanilla implementation from Part 1 by applying various loop optimization techniques to enhance execution performance. 

The project demonstrates the use of:
- Loop blocking
- Loop unrolling
- Array access pattern optimization
- Performance profiling to target bottlenecks

These optimizations are applied to the feedforward neural network and training processes to reduce runtime while maintaining high accuracy on the MNIST dataset.

---

## Features
- Fully optimized DNN implementation for MNIST handwritten digit recognition.
- Application of loop optimization techniques to critical sections of the code.
- Significant reduction in execution time compared to the unoptimized version.
- Training and evaluation of the model on a subset of MNIST digits.

---

## File Structure
```
MNIST_DNN_Part2/
├── mnist_dnn_optimized.cpp         # Main program file
├── data/                # Directory for MNIST dataset files
│   ├── train-images-idx3-ubyte
│   ├── train-labels-idx1-ubyte
│   ├── t10k-images-idx3-ubyte
│   └── t10k-labels-idx1-ubyte
├── CMakeLists.txt       # Build configuration for the project
└── README.md            # Project documentation
```

---

## Getting Started

### Prerequisites
- **C++ Compiler:** GCC 7.5+, Clang, or MSVC
- **CMake:** Version 3.10+
- MNIST dataset files placed in the `data` directory

### Build and Run

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/MNIST_DNN_Part2.git
   cd MNIST_DNN_Part2
   ```

2. Create a build directory and configure the project:
   ```bash
   mkdir build
   cd build
   cmake ..
   make
   ```

3. Run the program:
   ```bash
   ./MNIST_DNN_Part2 data/train-images-idx3-ubyte data/train-labels-idx1-ubyte \
                     data/t10k-images-idx3-ubyte data/t10k-labels-idx1-ubyte
   ```

---

## Optimizations Applied
1. **Loop Blocking**
   - Improved cache locality for matrix operations by processing sub-blocks of data.
2. **Loop Unrolling**
   - Reduced loop overhead by manually unrolling loops in performance-critical sections.
3. **Array Access Pattern Optimization**
   - Reordered computations and array accesses to reduce cache misses.
4. **Profiling and Targeted Optimization**
   - Used profiling tools to identify performance bottlenecks and applied optimizations where they had the most impact.

---

## Dataset
The MNIST dataset can be downloaded from [DeepAI](https://deepai.org/dataset/mnist). Place the files in the `data/` directory as shown in the file structure above.

---

## License
This project is released under the MIT License.

---

## References
1. Dan Claudiu Ciresan, et al, “Deep Big Simple Neural Nets Excel on Hand-written Digit Recognition,” arXiv:1003.0358, Mar, 2010.
2. [MNIST Dataset on DeepAI](https://deepai.org/dataset/mnist)
3. Loop optimization techniques: [GeeksforGeeks](https://www.geeksforgeeks.org/loop-optimization-techniques/), [Intel Optimizations Guide](https://www.intel.com)
