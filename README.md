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
- Fully optimized DNN implementation for MNIST handwritten digit recognition
- Application of loop optimization techniques to critical sections of the code
- Significant reduction in execution time compared to the unoptimized version
- Training and evaluation of the model on a subset of MNIST digits

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
   - Improved cache locality for matrix operations by processing sub-blocks of data
2. **Loop Unrolling**
   - Reduced loop overhead by manually unrolling loops in performance-critical sections
3. **Array Access Pattern Optimization**
   - Reordered computations and array accesses to reduce cache misses
4. **Profiling and Targeted Optimization**
   - Used profiling tools to identify performance bottlenecks and applied optimizations where they had the most impact

---

## Training Results and Analysis

### Performance Metrics

The network was trained for 10 epochs, showing consistent improvement in both training and test accuracy:

```
Epoch 1 - Train Accuracy: 86.7133%, Loss: 7972
         Test Accuracy: 92.16%
Epoch 2 - Train Accuracy: 93.815%, Loss: 3711
         Test Accuracy: 94.43%
Epoch 3 - Train Accuracy: 95.5367%, Loss: 2678
         Test Accuracy: 95.38%
Epoch 4 - Train Accuracy: 96.4367%, Loss: 2138
         Test Accuracy: 96.1%
Epoch 5 - Train Accuracy: 97.055%, Loss: 1767
         Test Accuracy: 96.64%
Epoch 6 - Train Accuracy: 97.4767%, Loss: 1514
         Test Accuracy: 96.93%
Epoch 7 - Train Accuracy: 97.8667%, Loss: 1280
         Test Accuracy: 97.08%
Epoch 8 - Train Accuracy: 98.1333%, Loss: 1120
         Test Accuracy: 97.21%
Epoch 9 - Train Accuracy: 98.37%, Loss: 978
         Test Accuracy: 97.25%
Epoch 10 - Train Accuracy: 98.615%, Loss: 831
         Test Accuracy: 97.16%
```

### Result Analysis

The training results demonstrate several important characteristics of the network's learning process:

1. **Early Learning Phase (Epochs 1-3)**
   - The network shows rapid improvement in accuracy, jumping from 86.7% to 95.5% on the training set
   - Test accuracy similarly improves from 92.16% to 95.38%
   - This phase represents the network learning the primary features of digit recognition

2. **Refinement Phase (Epochs 4-7)**
   - Learning rate slows but continues steadily
   - Training accuracy improves from 96.4% to 97.8%
   - Test accuracy maintains close alignment with training accuracy
   - The network is fine-tuning its feature recognition capabilities

3. **Convergence Phase (Epochs 8-10)**
   - Training accuracy approaches 98.6%
   - Test accuracy stabilizes around 97.2%
   - Diminishing returns in improvement suggest the network is reaching its capacity

### Optimization Impact

An important observation is that these results match those from the unoptimized implementation (Part 1). This equivalence is by design, as the optimizations implemented in Part 2 affect computational efficiency rather than the mathematical process. The optimizations provide:

- Improved cache utilization through loop blocking
- Reduced computational overhead via loop unrolling
- Better memory access patterns
- Enhanced CPU instruction pipeline usage

While these optimizations significantly improve execution speed, they maintain mathematical equivalence with the original implementation, ensuring the same high accuracy while reducing computational time.

---

## Dataset
The MNIST dataset can be downloaded from [DeepAI](https://deepai.org/dataset/mnist). Place the files in the `data/` directory as shown in the file structure above.

---

## License
This project is released under the MIT License.

---

## References
1. Dan Claudiu Ciresan, et al, "Deep Big Simple Neural Nets Excel on Hand-written Digit Recognition," arXiv:1003.0358, Mar, 2010.
2. [MNIST Dataset on DeepAI](https://deepai.org/dataset/mnist)
3. Loop optimization techniques: [GeeksforGeeks](https://www.geeksforgeeks.org/loop-optimization-techniques/), [Intel Optimizations Guide](https://www.intel.com)
