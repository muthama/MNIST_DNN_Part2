/******************************************************************************
 * Compile example:
 *   g++ -O2 -std=c++26 mnist_dnn_optimized.cpp -o mnist_dnn_optimized
 *
 * Example run:
 *   ./mnist_dnn_optimized train-images-idx3-ubyte train-labels-idx1-ubyte \
 *                         t10k-images-idx3-ubyte t10k-labels-idx1-ubyte
 *
 * This code demonstrates loop-blocking, loop-unrolling, and (in some places)
 * loop-interchange for performance in the forward/backprop calculations.
 ******************************************************************************/

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <algorithm>

// ----------------------------- Configuration -------------------------------- //
static const int IMAGE_SIZE     = 28;   // MNIST images: 28x28
static const int INPUT_SIZE     = 784;  // 28*28
static const int NUM_CLASSES    = 10;
static const int HIDDEN1        = 128;
static const int HIDDEN2        = 64;

// Hyperparameters
static const float LEARNING_RATE = 0.001f;
static const int EPOCHS          = 10;      // Increase for higher accuracy
static const int TRAINING_SAMPLES= 60000;
static const int TEST_SAMPLES    = 10000;

// For loop blocking
static const int BLOCK_SIZE = 16;

// --------------------- MNIST Reading Utilities ------------------------------ //
int reverseInt(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

bool readMNISTImages(const std::string &filename,
                     std::vector<std::vector<float>> &images, int count) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Could not open file: " << filename << std::endl;
        return false;
    }
    int magicNumber = 0, numberOfImages = 0, rows = 0, cols = 0;
    file.read(reinterpret_cast<char*>(&magicNumber), 4);
    file.read(reinterpret_cast<char*>(&numberOfImages), 4);
    file.read(reinterpret_cast<char*>(&rows), 4);
    file.read(reinterpret_cast<char*>(&cols), 4);

    magicNumber    = reverseInt(magicNumber);
    numberOfImages = reverseInt(numberOfImages);
    rows           = reverseInt(rows);
    cols           = reverseInt(cols);

    if (numberOfImages < count) {
        std::cerr << "File has fewer images (" << numberOfImages
                  << ") than required (" << count << "). Using all available.\n";
        count = numberOfImages;
    }
    images.resize(count, std::vector<float>(rows * cols, 0.0f));

    for (int i = 0; i < count; ++i) {
        for (int r = 0; r < rows * cols; ++r) {
            unsigned char pixel = 0;
            file.read(reinterpret_cast<char*>(&pixel), 1);
            // normalize to [-1, +1]
            images[i][r] = (pixel / 255.0f) * 2.0f - 1.0f;
        }
    }
    file.close();
    return true;
}

bool readMNISTLabels(const std::string &filename,
                     std::vector<int> &labels, int count) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Could not open file: " << filename << std::endl;
        return false;
    }
    int magicNumber = 0, numberOfLabels = 0;
    file.read(reinterpret_cast<char*>(&magicNumber), 4);
    file.read(reinterpret_cast<char*>(&numberOfLabels), 4);
    magicNumber    = reverseInt(magicNumber);
    numberOfLabels = reverseInt(numberOfLabels);

    if (numberOfLabels < count) {
        std::cerr << "File has fewer labels (" << numberOfLabels
                  << ") than required (" << count << "). Using all available.\n";
        count = numberOfLabels;
    }

    labels.resize(count);
    for (int i = 0; i < count; ++i) {
        unsigned char label = 0;
        file.read(reinterpret_cast<char*>(&label), 1);
        labels[i] = static_cast<int>(label);
    }
    file.close();
    return true;
}

// --------------------- Activation (Scaled Tanh) ---------------------------- //
inline float scaledTanh(float x) {
    // y = 1.7159 * tanh( (2/3)*x )
    const float alpha = 1.7159f;
    const float beta  = 2.0f/3.0f;
    return alpha * std::tanh(beta * x);
}

inline float scaledTanhDerivative(float x) {
    // derivative: 1.7159 * (2/3) * sech^2( (2/3)*x )
    const float alpha = 1.7159f;
    const float beta  = 2.0f/3.0f;
    float th = std::tanh(beta * x);
    float sech2 = 1.0f - th*th; // since sech^2(z) = 1 - tanh^2(z)
    return alpha * beta * sech2;
}

// -------------------- Helper: Random Initialization ------------------------- //
inline float randWeight() {
    // small random init in [-0.05, 0.05]
    return ((std::rand() / static_cast<float>(RAND_MAX)) * 0.1f) - 0.05f;
}

// ------------------------- MLP Class (Optimized) ---------------------------- //
class MLP {
public:
    MLP() {
        // Allocate & random-init W/b for each layer
        W1.resize(INPUT_SIZE,  std::vector<float>(HIDDEN1));
        b1.resize(HIDDEN1);

        W2.resize(HIDDEN1, std::vector<float>(HIDDEN2));
        b2.resize(HIDDEN2);

        W3.resize(HIDDEN2, std::vector<float>(NUM_CLASSES));
        b3.resize(NUM_CLASSES);

        for (int i = 0; i < INPUT_SIZE; ++i) {
            for (int j = 0; j < HIDDEN1; ++j) {
                W1[i][j] = randWeight();
            }
        }
        for (int j = 0; j < HIDDEN1; ++j) {
            b1[j] = randWeight();
        }

        for (int i = 0; i < HIDDEN1; ++i) {
            for (int j = 0; j < HIDDEN2; ++j) {
                W2[i][j] = randWeight();
            }
        }
        for (int j = 0; j < HIDDEN2; ++j) {
            b2[j] = randWeight();
        }

        for (int i = 0; i < HIDDEN2; ++i) {
            for (int j = 0; j < NUM_CLASSES; ++j) {
                W3[i][j] = randWeight();
            }
        }
        for (int j = 0; j < NUM_CLASSES; ++j) {
            b3[j] = randWeight();
        }
    }

    // -------------------- Forward Pass (Optimized) ------------------------ //
    void forward(const std::vector<float> &x) {
        // ----- Input -> Hidden1 -----
        z1.resize(HIDDEN1, 0.0f);
        a1.resize(HIDDEN1);

        // Instead of naive double loop i in [0..INPUT_SIZE], j in [0..HIDDEN1]
        // we apply loop-blocking (and a mild loop interchange).
        // 1) Initialize partial sums to biases
        for (int j = 0; j < HIDDEN1; ++j) {
            z1[j] = b1[j];
        }

        // 2) Accumulate in blocks
        for (int iB = 0; iB < INPUT_SIZE; iB += BLOCK_SIZE) {
            int iMax = std::min(iB + BLOCK_SIZE, INPUT_SIZE);
            for (int jB = 0; jB < HIDDEN1; jB += BLOCK_SIZE) {
                int jMax = std::min(jB + BLOCK_SIZE, HIDDEN1);

                // For each block, do partial dot-product
                for (int i = iB; i < iMax; ++i) {
                    float xi = x[i];
                    // Example of loop unrolling by 4 in the 'j' loop
                    int j = jB;
                    for (; j + 3 < jMax; j += 4) {
                        z1[j+0] += xi * W1[i][j+0];
                        z1[j+1] += xi * W1[i][j+1];
                        z1[j+2] += xi * W1[i][j+2];
                        z1[j+3] += xi * W1[i][j+3];
                    }
                    // cleanup
                    for (; j < jMax; ++j) {
                        z1[j] += xi * W1[i][j];
                    }
                }
            }
        }

        // 3) Apply scaled Tanh
        for (int j = 0; j < HIDDEN1; ++j) {
            a1[j] = scaledTanh(z1[j]);
        }

        // ----- Hidden1 -> Hidden2 -----
        z2.resize(HIDDEN2, 0.0f);
        a2.resize(HIDDEN2);

        // bias init
        for (int j = 0; j < HIDDEN2; ++j) {
            z2[j] = b2[j];
        }

        // blocked accumulation
        for (int iB = 0; iB < HIDDEN1; iB += BLOCK_SIZE) {
            int iMax = std::min(iB + BLOCK_SIZE, HIDDEN1);
            for (int jB = 0; jB < HIDDEN2; jB += BLOCK_SIZE) {
                int jMax = std::min(jB + BLOCK_SIZE, HIDDEN2);
                for (int i = iB; i < iMax; ++i) {
                    float ai = a1[i];
                    // unroll by 4
                    int j = jB;
                    for (; j + 3 < jMax; j += 4) {
                        z2[j+0] += ai * W2[i][j+0];
                        z2[j+1] += ai * W2[i][j+1];
                        z2[j+2] += ai * W2[i][j+2];
                        z2[j+3] += ai * W2[i][j+3];
                    }
                    for (; j < jMax; ++j) {
                        z2[j] += ai * W2[i][j];
                    }
                }
            }
        }

        for (int j = 0; j < HIDDEN2; ++j) {
            a2[j] = scaledTanh(z2[j]);
        }

        // ----- Hidden2 -> Output -----
        z3.resize(NUM_CLASSES, 0.0f);
        a3.resize(NUM_CLASSES);

        // bias init
        for (int j = 0; j < NUM_CLASSES; ++j) {
            z3[j] = b3[j];
        }

        // blocked accumulation
        for (int iB = 0; iB < HIDDEN2; iB += BLOCK_SIZE) {
            int iMax = std::min(iB + BLOCK_SIZE, HIDDEN2);
            for (int jB = 0; jB < NUM_CLASSES; jB += BLOCK_SIZE) {
                int jMax = std::min(jB + BLOCK_SIZE, NUM_CLASSES);
                for (int i = iB; i < iMax; ++i) {
                    float ai = a2[i];
                    // unroll by 4
                    int j = jB;
                    for (; j + 3 < jMax; j += 4) {
                        z3[j+0] += ai * W3[i][j+0];
                        z3[j+1] += ai * W3[i][j+1];
                        z3[j+2] += ai * W3[i][j+2];
                        z3[j+3] += ai * W3[i][j+3];
                    }
                    for (; j < jMax; ++j) {
                        z3[j] += ai * W3[i][j];
                    }
                }
            }
        }

        // Softmax for final classification
        float maxLogit = *std::max_element(z3.begin(), z3.end());
        float sumExp = 0.0f;
        for (int j = 0; j < NUM_CLASSES; ++j) {
            float e = std::exp(z3[j] - maxLogit);
            a3[j] = e;
            sumExp += e;
        }
        for (int j = 0; j < NUM_CLASSES; ++j) {
            a3[j] /= sumExp;
        }
    }

    // -------------------- Backpropagation (Optimized) --------------------- //
    void backward(const std::vector<float> &x, int label) {
        // 1) Output delta (cross-entropy + softmax => delta = (p - y))
        std::vector<float> delta3(NUM_CLASSES);
        for (int j = 0; j < NUM_CLASSES; ++j) {
            float target = (j == label) ? 1.0f : 0.0f;
            delta3[j] = (a3[j] - target);
        }

        // 2) W3, b3 updates: Hidden2 -> Output
        for (int j = 0; j < NUM_CLASSES; ++j) {
            float d3j = delta3[j];
            // We can do blocking/unrolling on this loop as well
            for (int iB = 0; iB < HIDDEN2; iB += 4) {
                int iMax = std::min(iB + 4, HIDDEN2);
                for (int i = iB; i < iMax; ++i) {
                    W3[i][j] -= LEARNING_RATE * d3j * a2[i];
                }
            }
            b3[j] -= LEARNING_RATE * d3j;
        }

        // 3) delta2 = (delta3 * W3^T) * scaledTanh'(z2)
        std::vector<float> delta2(HIDDEN2, 0.0f);
        for (int i = 0; i < HIDDEN2; ++i) {
            float grad = 0.0f;
            // We can also block/unroll here
            for (int jB = 0; jB < NUM_CLASSES; jB += 4) {
                int jMax = std::min(jB + 4, NUM_CLASSES);
                for (int j = jB; j < jMax; ++j) {
                    grad += delta3[j] * W3[i][j];
                }
            }
            grad *= scaledTanhDerivative(z2[i]);
            delta2[i] = grad;
        }

        // 4) W2, b2 updates: Hidden1 -> Hidden2
        for (int j = 0; j < HIDDEN2; ++j) {
            float d2j = delta2[j];
            for (int iB = 0; iB < HIDDEN1; iB += 4) {
                int iMax = std::min(iB + 4, HIDDEN1);
                for (int i = iB; i < iMax; ++i) {
                    W2[i][j] -= LEARNING_RATE * d2j * a1[i];
                }
            }
            b2[j] -= LEARNING_RATE * d2j;
        }

        // 5) delta1 = (delta2 * W2^T) * scaledTanh'(z1)
        std::vector<float> delta1(HIDDEN1, 0.0f);
        for (int i = 0; i < HIDDEN1; ++i) {
            float grad = 0.0f;
            // block/unroll
            for (int jB = 0; jB < HIDDEN2; jB += 4) {
                int jMax = std::min(jB + 4, HIDDEN2);
                for (int j = jB; j < jMax; ++j) {
                    grad += delta2[j] * W2[i][j];
                }
            }
            grad *= scaledTanhDerivative(z1[i]);
            delta1[i] = grad;
        }

        // 6) W1, b1 updates: Input -> Hidden1
        for (int j = 0; j < HIDDEN1; ++j) {
            float d1j = delta1[j];
            // block/unroll
            for (int iB = 0; iB < INPUT_SIZE; iB += 4) {
                int iMax = std::min(iB + 4, INPUT_SIZE);
                for (int i = iB; i < iMax; ++i) {
                    W1[i][j] -= LEARNING_RATE * d1j * x[i];
                }
            }
            b1[j] -= LEARNING_RATE * d1j;
        }
    }

    // Predict label
    int predict(const std::vector<float> &x) {
        forward(x);
        int maxIdx = 0;
        float maxVal = a3[0];
        for (int j = 1; j < NUM_CLASSES; ++j) {
            if (a3[j] > maxVal) {
                maxVal = a3[j];
                maxIdx = j;
            }
        }
        return maxIdx;
    }

private:
    // Weights & biases
    std::vector<std::vector<float>> W1; // [INPUT_SIZE][HIDDEN1]
    std::vector<float> b1;             // [HIDDEN1]

    std::vector<std::vector<float>> W2; // [HIDDEN1][HIDDEN2]
    std::vector<float> b2;             // [HIDDEN2]

    std::vector<std::vector<float>> W3; // [HIDDEN2][NUM_CLASSES]
    std::vector<float> b3;             // [NUM_CLASSES]

    // Forward intermediate results
    std::vector<float> z1, a1;  // for Hidden1
    std::vector<float> z2, a2;  // for Hidden2
    std::vector<float> z3, a3;  // for Output
};

// --------------------------- Main Program ----------------------------------- //
int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <train_images> <train_labels> <test_images> <test_labels>\n";
        return 1;
    }

    std::string trainImagePath = argv[1];
    std::string trainLabelPath = argv[2];
    std::string testImagePath  = argv[3];
    std::string testLabelPath  = argv[4];

    // Seed random
    std::srand(123); // for reproducibility (optional)

    // Read training data
    std::vector<std::vector<float>> trainImages;
    std::vector<int> trainLabels;
    if (!readMNISTImages(trainImagePath, trainImages, TRAINING_SAMPLES) ||
        !readMNISTLabels(trainLabelPath, trainLabels, TRAINING_SAMPLES)) {
        std::cerr << "Error reading training data.\n";
        return 1;
    }

    // Read test data
    std::vector<std::vector<float>> testImages;
    std::vector<int> testLabels;
    if (!readMNISTImages(testImagePath, testImages, TEST_SAMPLES) ||
        !readMNISTLabels(testLabelPath, testLabels, TEST_SAMPLES)) {
        std::cerr << "Error reading test data.\n";
        return 1;
    }

    // Create MLP
    MLP dnn;

    // Training
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        int correctCount = 0;
        float epochLoss = 0.0f;

        // For better generalization, you might shuffle the data here.
        // We'll keep it simple.

        for (int i = 0; i < TRAINING_SAMPLES; ++i) {
            // Forward
            dnn.forward(trainImages[i]);
            // Predict
            int pred = dnn.predict(trainImages[i]);
            if (pred == trainLabels[i]) {
                correctCount++;
            }

            // A dummy "loss" measure (0 if correct, 1 if incorrect)
            epochLoss += (pred == trainLabels[i]) ? 0.0f : 1.0f;

            // Backprop
            dnn.backward(trainImages[i], trainLabels[i]);
        }

        float trainAccuracy = (100.0f * correctCount) / TRAINING_SAMPLES;
        std::cout << "Epoch " << (epoch + 1)
                  << " - Train Accuracy: " << trainAccuracy << "%, Loss: "
                  << epochLoss << std::endl;

        // Evaluate on test set
        int testCorrect = 0;
        for (int i = 0; i < TEST_SAMPLES; ++i) {
            int pred = dnn.predict(testImages[i]);
            if (pred == testLabels[i]) {
                testCorrect++;
            }
        }
        float testAccuracy = (100.0f * testCorrect) / TEST_SAMPLES;
        std::cout << "         Test Accuracy: " << testAccuracy << "%\n";
    }

    return 0;
}