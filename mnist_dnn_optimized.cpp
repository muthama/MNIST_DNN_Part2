/****************************************************************************
 * Deep Neural Network Implementation for MNIST Digit Recognition - Part 1
 *
 * This implementation showcases a neural network designed specifically for the MNIST
 * handwritten digit recognition task. The code demonstrates advanced optimization
 * techniques while maintaining readability and educational value.
 *
 * Key Optimization Techniques:
 * 1. Loop blocking (tiling): Breaks large matrix operations into cache-friendly chunks
 * 2. Loop unrolling: Reduces loop overhead by processing multiple elements per iteration
 * 3. Memory access optimization: Ensures efficient cache usage through strategic data access
 *
 * Network Architecture:
 * - Input Layer: 784 neurons (28x28 pixel images)
 * - Hidden Layer 1: 128 neurons
 * - Hidden Layer 2: 64 neurons
 * - Output Layer: 10 neurons (one per digit)
 ****************************************************************************/

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <algorithm>

// -------------------------- Network Configuration -------------------------- //
// The network's structure is defined by these constants, carefully chosen
// based on the MNIST dataset characteristics and empirical testing
static const int IMAGE_SIZE = 28; // MNIST images are 28x28 pixels
static const int INPUT_SIZE = 784; // Flattened input size (28*28)
static const int NUM_CLASSES = 10; // One class per digit (0-9)
static const int HIDDEN1 = 128; // First hidden layer size
static const int HIDDEN2 = 64; // Second hidden layer size

// Training hyperparameters - these values significantly impact learning
static const float LEARNING_RATE = 0.001f; // Controls weight update magnitude
// Small enough to prevent overshooting
// Large enough to learn effectively
static const int EPOCHS = 10; // Number of complete training passes
static const int TRAINING_SAMPLES = 60000; // Full MNIST training set size
static const int TEST_SAMPLES = 10000; // Full MNIST test set size

// Cache optimization parameter
// Chosen to match typical CPU cache line sizes (16 * 4 bytes = 64 bytes)
static const int BLOCK_SIZE = 16;

// ------------------------ MNIST File Reading Utilities -------------------- //
/**
 * Reverses byte order in a 32-bit integer
 *
 * MNIST files use big-endian format, while most modern processors use
 * little-endian. This function handles the necessary conversion.
 *
 * Process:
 * 1. Extract individual bytes using bitwise AND and shifts
 * 2. Reassemble bytes in reverse order
 */
int reverseInt(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255; // Extract least significant byte
    c2 = (i >> 8) & 255; // Extract second byte
    c3 = (i >> 16) & 255; // Extract third byte
    c4 = (i >> 24) & 255; // Extract most significant byte
    return ((int) c1 << 24) + ((int) c2 << 16) + ((int) c3 << 8) + c4;
}

/**
 * Reads and preprocesses MNIST image data
 *
 * The MNIST dataset uses a specific IDX file format:
 * - First 4 bytes: Magic number (always 2051 for images)
 * - Next 4 bytes: Number of images
 * - Next 4 bytes: Number of rows
 * - Next 4 bytes: Number of columns
 * - Remaining bytes: Actual image data, one byte per pixel
 *
 * This function:
 * 1. Opens the binary file
 * 2. Reads and validates header information
 * 3. Allocates memory for the images
 * 4. Reads pixel data and normalizes to [-1, 1] range
 */
bool readMNISTImages(const std::string &filename,
                     std::vector<std::vector<float> > &images, int count) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Could not open file: " << filename << std::endl;
        return false;
    }

    // Read and verify file header
    int magicNumber = 0, numberOfImages = 0, rows = 0, cols = 0;
    file.read(reinterpret_cast<char *>(&magicNumber), 4);
    file.read(reinterpret_cast<char *>(&numberOfImages), 4);
    file.read(reinterpret_cast<char *>(&rows), 4);
    file.read(reinterpret_cast<char *>(&cols), 4);

    // Convert from big-endian to host byte order
    magicNumber = reverseInt(magicNumber);
    numberOfImages = reverseInt(numberOfImages);
    rows = reverseInt(rows);
    cols = reverseInt(cols);

    // Handle case where requested count exceeds available images
    if (numberOfImages < count) {
        std::cerr << "File has fewer images (" << numberOfImages
                << ") than required (" << count << "). Using all available.\n";
        count = numberOfImages;
    }

    // Prepare storage for images
    images.resize(count, std::vector<float>(rows * cols, 0.0f));

    // Read and normalize pixel data
    // Normalization to [-1, 1] helps with training stability:
    // - Centers data around 0
    // - Gives equal weight to positive and negative values
    // - Helps prevent saturation of activation functions
    for (int i = 0; i < count; ++i) {
        for (int r = 0; r < rows * cols; ++r) {
            unsigned char pixel = 0;
            file.read(reinterpret_cast<char *>(&pixel), 1);
            images[i][r] = (pixel / 255.0f) * 2.0f - 1.0f;
        }
    }
    file.close();
    return true;
}

/**
 * Reads MNIST label data
 *
 * The label file format is similar but simpler than the image format:
 * - First 4 bytes: Magic number (always 2049 for labels)
 * - Next 4 bytes: Number of labels
 * - Remaining bytes: Label data, one byte per label
 *
 * Labels are single digits (0-9) stored as unsigned bytes
 */
bool readMNISTLabels(const std::string &filename,
                     std::vector<int> &labels, int count) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Could not open file: " << filename << std::endl;
        return false;
    }

    // Read header information
    int magicNumber = 0, numberOfLabels = 0;
    file.read(reinterpret_cast<char *>(&magicNumber), 4);
    file.read(reinterpret_cast<char *>(&numberOfLabels), 4);
    magicNumber = reverseInt(magicNumber);
    numberOfLabels = reverseInt(numberOfLabels);

    // Handle case where requested count exceeds available labels
    if (numberOfLabels < count) {
        std::cerr << "File has fewer labels (" << numberOfLabels
                << ") than required (" << count << "). Using all available.\n";
        count = numberOfLabels;
    }

    // Read label data
    labels.resize(count);
    for (int i = 0; i < count; ++i) {
        unsigned char label = 0;
        file.read(reinterpret_cast<char *>(&label), 1);
        labels[i] = static_cast<int>(label);
    }
    file.close();
    return true;
}

// ------------------------ Activation Functions ---------------------------- //
/**
 * Scaled Tanh Activation Function
 *
 * Formula: y = 1.7159 * tanh(2x/3)
 *
 * This scaling is not arbitrary - it's designed to have specific properties:
 * 1. The factor 1.7159 is chosen because tanh(1) ≈ 0.76, so:
 *    1.7159 * 0.76 ≈ 1.3, giving a good output range
 * 2. The 2/3 factor makes the derivative ≈ 1 at x=0, which helps with:
 *    - Training stability
 *    - Gradient flow
 *    - Avoiding vanishing/exploding gradients
 */
inline float scaledTanh(float x) {
    const float alpha = 1.7159f; // Output scaling factor
    const float beta = 2.0f / 3.0f; // Input scaling factor
    return alpha * std::tanh(beta * x);
}

/**
 * Derivative of Scaled Tanh
 *
 * Used during backpropagation to compute gradients.
 *
 * Mathematical derivation:
 * 1. y = 1.7159 * tanh(2x/3)
 * 2. dy/dx = 1.7159 * (2/3) * (1 - tanh²(2x/3))
 * 3. We use the identity sech²(x) = 1 - tanh²(x) for efficiency
 */
inline float scaledTanhDerivative(float x) {
    const float alpha = 1.7159f;
    const float beta = 2.0f / 3.0f;
    float th = std::tanh(beta * x);
    float sech2 = 1.0f - th * th; // More efficient than computing sech directly
    return alpha * beta * sech2;
}

// ---------------------- Weight Initialization ---------------------------- //
/**
 * Initialize weights with small random values
 *
 * We use small initial weights (-0.05 to 0.05) for several reasons:
 * 1. Prevents neuron saturation at the start of training
 * 2. Helps maintain reasonable gradient magnitudes
 * 3. Large enough to break symmetry between neurons
 * 4. Small enough to keep activations in the linear region initially
 */
inline float randWeight() {
    return ((std::rand() / static_cast<float>(RAND_MAX)) * 0.1f) - 0.05f;
}

// -------------------------- Neural Network Class ------------------------- //
class MLP {
public:
    /**
     * Constructor: Initializes the neural network
     *
     * Creates three layers with the following structure:
     * - Input -> Hidden1: W1 (784x128) and b1 (128)
     * - Hidden1 -> Hidden2: W2 (128x64) and b2 (64)
     * - Hidden2 -> Output: W3 (64x10) and b3 (10)
     *
     * Each weight matrix is initialized with small random values to:
     * 1. Break symmetry between neurons
     * 2. Ensure reasonable starting point for training
     */
    MLP() {
        // Allocate memory for weights and biases
        W1.resize(INPUT_SIZE, std::vector<float>(HIDDEN1));
        b1.resize(HIDDEN1);

        W2.resize(HIDDEN1, std::vector<float>(HIDDEN2));
        b2.resize(HIDDEN2);

        W3.resize(HIDDEN2, std::vector<float>(NUM_CLASSES));
        b3.resize(NUM_CLASSES);

        // Initialize with small random values
        // Layer 1: Input -> Hidden1
        for (int i = 0; i < INPUT_SIZE; ++i) {
            for (int j = 0; j < HIDDEN1; ++j) {
                W1[i][j] = randWeight();
            }
        }
        for (int j = 0; j < HIDDEN1; ++j) {
            b1[j] = randWeight();
        }

        // Layer 2: Hidden1 -> Hidden2
        for (int i = 0; i < HIDDEN1; ++i) {
            for (int j = 0; j < HIDDEN2; ++j) {
                W2[i][j] = randWeight();
            }
        }
        for (int j = 0; j < HIDDEN2; ++j) {
            b2[j] = randWeight();
        }

        // Layer 3: Hidden2 -> Output
        for (int i = 0; i < HIDDEN2; ++i) {
            for (int j = 0; j < NUM_CLASSES; ++j) {
                W3[i][j] = randWeight();
            }
        }
        for (int j = 0; j < NUM_CLASSES; ++j) {
            b3[j] = randWeight();
        }
    }

    /**
     * Forward Pass Implementation
     *
     * The forward pass computes the network's prediction for a given input.
     * For each layer, we:
     * 1. Compute weighted sum of inputs (z = Wx + b)
     * 2. Apply activation function (a = f(z))
     *
     * Optimization techniques used:
     * - Cache blocking: Process data in small chunks that fit in CPU cache
     * - Loop unrolling: Process multiple elements per iteration
     * - Register reuse: Cache frequently accessed values in CPU registers
     */
    void forward(const std::vector<float> &x) {
        // ------------ Layer 1: Input → Hidden1 ------------
        z1.resize(HIDDEN1, 0.0f); // Pre-activation values
        a1.resize(HIDDEN1); // Post-activation values

        // Initialize with biases (z = b initially)
        for (int j = 0; j < HIDDEN1; ++j) {
            z1[j] = b1[j];
        }

        // Compute z = Wx + b using blocked matrix multiplication
        for (int iB = 0; iB < INPUT_SIZE; iB += BLOCK_SIZE) {
            int iMax = std::min(iB + BLOCK_SIZE, INPUT_SIZE);
            for (int jB = 0; jB < HIDDEN1; jB += BLOCK_SIZE) {
                int jMax = std::min(jB + BLOCK_SIZE, HIDDEN1);

                // Process current block with loop unrolling
                for (int i = iB; i < iMax; ++i) {
                    float xi = x[i]; // Cache input value in register
                    // Unroll inner loop by 4 for better instruction pipelining
                    int j = jB;
                    for (; j + 3 < jMax; j += 4) {
                        z1[j + 0] += xi * W1[i][j + 0];
                        z1[j + 1] += xi * W1[i][j + 1];
                        z1[j + 2] += xi * W1[i][j + 2];
                        z1[j + 3] += xi * W1[i][j + 3];
                    }
                    // Handle remaining elements
                    for (; j < jMax; ++j) {
                        z1[j] += xi * W1[i][j];
                    }
                }
            }
        }

        // Apply activation function: a1 = f(z1)
        for (int j = 0; j < HIDDEN1; ++j) {
            a1[j] = scaledTanh(z1[j]);
        }

        // ------------ Layer 2: Hidden1 → Hidden2 ------------
        // Similar structure to Layer 1, but with different dimensions
        z2.resize(HIDDEN2, 0.0f);
        a2.resize(HIDDEN2);

        for (int j = 0; j < HIDDEN2; ++j) {
            z2[j] = b2[j];
        }

        // Blocked matrix multiplication
        for (int iB = 0; iB < HIDDEN1; iB += BLOCK_SIZE) {
            int iMax = std::min(iB + BLOCK_SIZE, HIDDEN1);
            for (int jB = 0; jB < HIDDEN2; jB += BLOCK_SIZE) {
                int jMax = std::min(jB + BLOCK_SIZE, HIDDEN2);
                for (int i = iB; i < iMax; ++i) {
                    float ai = a1[i]; // Cache activation from previous layer
                    int j = jB;
                    for (; j + 3 < jMax; j += 4) {
                        z2[j + 0] += ai * W2[i][j + 0];
                        z2[j + 1] += ai * W2[i][j + 1];
                        z2[j + 2] += ai * W2[i][j + 2];
                        z2[j + 3] += ai * W2[i][j + 3];
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

        // ------------ Layer 3: Hidden2 → Output ------------
        z3.resize(NUM_CLASSES, 0.0f);
        a3.resize(NUM_CLASSES);

        // Initialize with biases
        for (int j = 0; j < NUM_CLASSES; ++j) {
            z3[j] = b3[j];
        }

        // Final layer matrix multiplication
        for (int iB = 0; iB < HIDDEN2; iB += BLOCK_SIZE) {
            int iMax = std::min(iB + BLOCK_SIZE, HIDDEN2);
            for (int jB = 0; jB < NUM_CLASSES; jB += BLOCK_SIZE) {
                int jMax = std::min(jB + BLOCK_SIZE, NUM_CLASSES);
                for (int i = iB; i < iMax; ++i) {
                    float ai = a2[i];
                    int j = jB;
                    for (; j + 3 < jMax; j += 4) {
                        z3[j + 0] += ai * W3[i][j + 0];
                        z3[j + 1] += ai * W3[i][j + 1];
                        z3[j + 2] += ai * W3[i][j + 2];
                        z3[j + 3] += ai * W3[i][j + 3];
                    }
                    for (; j < jMax; ++j) {
                        z3[j] += ai * W3[i][j];
                    }
                }
            }
        }

        // Apply Softmax activation for output layer
        // Softmax(x)_i = exp(x_i) / Σ_j exp(x_j)
        // We subtract max(x) for numerical stability
        float maxLogit = *std::max_element(z3.begin(), z3.end());
        float sumExp = 0.0f;
        for (int j = 0; j < NUM_CLASSES; ++j) {
            float e = std::exp(z3[j] - maxLogit); // Subtract max for stability
            a3[j] = e;
            sumExp += e;
        }
        for (int j = 0; j < NUM_CLASSES; ++j) {
            a3[j] /= sumExp; // Normalize to get probabilities
        }
    }

    /**
     * Backpropagation Implementation
     *
     * Backpropagation computes gradients and updates weights using chain rule:
     * 1. Compute output layer error (δL = ∂Loss/∂z)
     * 2. Propagate error backward through network
     * 3. Update weights and biases using computed gradients
     *
     * For classification, we use cross-entropy loss with softmax output,
     * which simplifies the output layer gradient to (a - y) where:
     * - a is the softmax output
     * - y is the one-hot encoded true label
     */
    void backward(const std::vector<float> &x, int label) {
        // ------------ Output Layer Gradients ------------
        std::vector<float> delta3(NUM_CLASSES);
        // For cross-entropy loss with softmax, gradient is (a - y)
        for (int j = 0; j < NUM_CLASSES; ++j) {
            float target = (j == label) ? 1.0f : 0.0f;
            delta3[j] = (a3[j] - target);
        }

        // ------------ Update Output Layer (W3, b3) ------------
        for (int j = 0; j < NUM_CLASSES; ++j) {
            float d3j = delta3[j];
            // Update weights with gradient descent
            // ∂Loss/∂W = δ * a^T
            for (int iB = 0; iB < HIDDEN2; iB += 4) {
                int iMax = std::min(iB + 4, HIDDEN2);
                for (int i = iB; i < iMax; ++i) {
                    W3[i][j] -= LEARNING_RATE * d3j * a2[i];
                }
            }
            // Update bias (∂Loss/∂b = δ)
            b3[j] -= LEARNING_RATE * d3j;
        }

        // ------------ Hidden Layer 2 Gradients ------------
        // δ2 = (W3^T * δ3) ⊙ f'(z2)
        std::vector<float> delta2(HIDDEN2, 0.0f);
        for (int i = 0; i < HIDDEN2; ++i) {
            float grad = 0.0f;
            // Compute W3^T * δ3 with loop unrolling
            for (int jB = 0; jB < NUM_CLASSES; jB += 4) {
                int jMax = std::min(jB + 4, NUM_CLASSES);
                for (int j = jB; j < jMax; ++j) {
                    grad += delta3[j] * W3[i][j];
                }
            }
            // Multiply by derivative of activation function
            grad *= scaledTanhDerivative(z2[i]);
            delta2[i] = grad;
        }

        // ------------ Update Hidden Layer 2 (W2, b2) ------------
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

        // ------------ Hidden Layer 1 Gradients ------------
        // δ1 = (W2^T * δ2) ⊙ f'(z1)
        std::vector<float> delta1(HIDDEN1, 0.0f);
        for (int i = 0; i < HIDDEN1; ++i) {
            float grad = 0.0f;
            for (int jB = 0; jB < HIDDEN2; jB += 4) {
                int jMax = std::min(jB + 4, HIDDEN2);
                for (int j = jB; j < jMax; ++j) {
                    grad += delta2[j] * W2[i][j];
                }
            }
            grad *= scaledTanhDerivative(z1[i]);
            delta1[i] = grad;
        }

        // ------------ Update Input Layer (W1, b1) ------------
        for (int j = 0; j < HIDDEN1; ++j) {
            float d1j = delta1[j];
            for (int iB = 0; iB < INPUT_SIZE; iB += 4) {
                int iMax = std::min(iB + 4, INPUT_SIZE);
                for (int i = iB; i < iMax; ++i) {
                    W1[i][j] -= LEARNING_RATE * d1j * x[i];
                }
            }
            b1[j] -= LEARNING_RATE * d1j;
        }
    }

    /**
     * Prediction Function
     *
     * Returns the most likely digit (0-9) for the input image.
     * 1. Performs forward pass to get probabilities
     * 2. Returns index (digit) with highest probability
     */
    int predict(const std::vector<float> &x) {
        forward(x);
        return std::max_element(a3.begin(), a3.end()) - a3.begin();
    }

private:
    // Network weights and biases
    std::vector<std::vector<float> > W1; // Input → Hidden1
    std::vector<float> b1; // Hidden1 bias

    std::vector<std::vector<float> > W2; // Hidden1 → Hidden2
    std::vector<float> b2; // Hidden2 bias

    std::vector<std::vector<float> > W3; // Hidden2 → Output
    std::vector<float> b3; // Output bias

    // Intermediate values needed for backpropagation
    std::vector<float> z1, a1; // Hidden1 pre- and post-activation
    std::vector<float> z2, a2; // Hidden2 pre- and post-activation
    std::vector<float> z3, a3; // Output pre- and post-activation
};


/**
 * Main Program Implementation
 * 
 * This section implements the primary program flow for training and evaluating
 * the neural network on the MNIST dataset. The process follows these key steps:
 * 1. Load and preprocess the MNIST data
 * 2. Initialize the neural network
 * 3. Train the network through multiple epochs
 * 4. Evaluate performance on test data
 */
int main(int argc, char **argv) {
    // Verify correct command-line usage
    // The program expects paths to both training and test files
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0]
                << " <train_images> <train_labels> <test_images> <test_labels>\n";
        return 1;
    }

    // Extract file paths from command line arguments
    std::string trainImagePath = argv[1];
    std::string trainLabelPath = argv[2];
    std::string testImagePath = argv[3];
    std::string testLabelPath = argv[4];

    // Set random seed for reproducibility
    // This ensures weight initialization is consistent across runs
    // Important for debugging and comparing different optimizations
    std::srand(123);

    // ------------ Data Loading Phase ------------

    // Load training data
    // These vectors will hold the entire training dataset in memory
    std::vector<std::vector<float> > trainImages; // Each image is a vector of normalized pixel values
    std::vector<int> trainLabels; // Labels are integers 0-9

    // Attempt to read training data files
    if (!readMNISTImages(trainImagePath, trainImages, TRAINING_SAMPLES) ||
        !readMNISTLabels(trainLabelPath, trainLabels, TRAINING_SAMPLES)) {
        std::cerr << "Error reading training data.\n";
        return 1;
    }

    // Load test data
    // The test set is used to evaluate the network's generalization
    std::vector<std::vector<float> > testImages;
    std::vector<int> testLabels;

    // Attempt to read test data files
    if (!readMNISTImages(testImagePath, testImages, TEST_SAMPLES) ||
        !readMNISTLabels(testLabelPath, testLabels, TEST_SAMPLES)) {
        std::cerr << "Error reading test data.\n";
        return 1;
    }

    // ------------ Network Initialization ------------

    // Create neural network instance
    // This initializes all weights and biases with small random values
    MLP dnn;

    // ------------ Training Loop ------------

    // Iterate through multiple epochs
    // Each epoch processes the entire training set once
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        // Training metrics for this epoch
        int correctCount = 0; // Number of correct predictions
        float epochLoss = 0.0f; // Cumulative loss for this epoch

        // Note: For better generalization, you might want to shuffle
        // the training data here at the start of each epoch

        // Process each training example
        for (int i = 0; i < TRAINING_SAMPLES; ++i) {
            // Forward pass: compute network's prediction
            dnn.forward(trainImages[i]);

            // Get network's prediction for this image
            int pred = dnn.predict(trainImages[i]);

            // Update accuracy statistics
            if (pred == trainLabels[i]) {
                correctCount++;
            }

            // Update loss
            // Here we use a simple 0/1 loss for monitoring
            // 0 for correct predictions, 1 for incorrect
            epochLoss += (pred == trainLabels[i]) ? 0.0f : 1.0f;

            // Backward pass: update weights and biases
            // This is where the learning happens
            dnn.backward(trainImages[i], trainLabels[i]);
        }

        // ------------ Epoch Performance Reporting ------------

        // Calculate and display training accuracy for this epoch
        float trainAccuracy = (100.0f * correctCount) / TRAINING_SAMPLES;
        std::cout << "Epoch " << (epoch + 1)
                << " - Train Accuracy: " << trainAccuracy << "%, Loss: "
                << epochLoss << std::endl;

        // ------------ Evaluation on Test Set ------------

        // After each epoch, evaluate performance on test set
        // This helps monitor for overfitting
        int testCorrect = 0;
        for (int i = 0; i < TEST_SAMPLES; ++i) {
            int pred = dnn.predict(testImages[i]);
            if (pred == testLabels[i]) {
                testCorrect++;
            }
        }

        // Calculate and display test accuracy
        float testAccuracy = (100.0f * testCorrect) / TEST_SAMPLES;
        std::cout << "         Test Accuracy: " << testAccuracy << "%\n";

        // Note: In a more complete implementation, you might want to:
        // 1. Save the model if it achieves best test accuracy so far
        // 2. Implement early stopping if test accuracy plateaus
        // 3. Track additional metrics like confusion matrix
        // 4. Adjust learning rate based on performance
    }

    return 0;
}
