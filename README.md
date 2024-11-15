# RBF Neural Network: Custom Implementation for Function Approximation and Pattern Recognition

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.21%2B-green.svg)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4%2B-orange.svg)

## 📈 Overview

This **custom Radial Basis Function (RBF) Neural Network implementation** demonstrates a specialized approach for function approximation and pattern recognition tasks. By leveraging Gaussian basis functions and linear weights, the network is ideal for applications requiring nonlinear data modeling and generalization. Whether for regression, classification, or signal processing, this implementation provides a foundational template for exploring RBF networks' power in machine learning.

## 🚀 Key Features
### **Initialization**
- Centers of the radial basis functions are chosen using K-means clustering for optimal representation of the input data distribution.
- Weights are initialized randomly, and biases are set to zero, ensuring the network is ready for training.

### **Training**
- **Least Squares Method**: The network learns the optimal weights by minimizing the sum of squared errors.
- Supports supervised learning by computing the output as a linear combination of Gaussian basis functions.

### **Prediction**
- Once trained, the network predicts outputs for unseen data by applying the learned weights to the RBF outputs.

### **Customizable Parameters**
- Number of RBF nodes
- Gaussian kernel width (`sigma`)

## 🎨 Visual Examples

This section highlights the network's ability to approximate a nonlinear function and classify data.

### **1. Scatterplot of Training Data and Prototypes**
This scatterplot shows the training data points along with the hidden layer prototypes. The prototypes are the network's learned representations of the input data, which help the RBF network classify data effectively.

<img src="https://github.com/rubythedev/rbf_neural_network/blob/main/images/scatter_train_prototypes.png" width="400">

### **2. MNIST Dataset**
This image shows some sample images from the MNIST dataset, a collection of handwritten digits used for classification tasks. The RBF network is trained on this dataset to learn patterns and classify digits accurately.

<img src="https://github.com/rubythedev/rbf_neural_network/blob/main/images/mnist_dataset.png" width="400">

### **3. Hidden Layer Prototypes for Digit Classification**
This image visualizes the learned hidden layer prototypes after training the RBF network on the MNIST dataset. These prototypes represent the key features of each digit, and the network uses them to classify new data points. The network achieved an **80% classification accuracy** on the MNIST test data, demonstrating its effectiveness in digit recognition.

<img src="https://github.com/rubythedev/rbf_neural_network/blob/main/images/hidden_layer_prototypes.png" width="400">

## 🛠️ Technologies & Skills

- **Programming Languages:** 
  - [Python 3.x](https://www.python.org/) for general-purpose programming and data manipulation

- **Libraries & Frameworks:** 
  - [NumPy](https://numpy.org/) for efficient numerical computations and matrix operations
  - [Matplotlib](https://matplotlib.org/) for data visualization, including 2D plotting and charting
  - [palettable](https://github.com/jakevdp/palettable) for data visualization, specifically **CartoColors** for visually appealing color palettes

- **Machine Learning Techniques:** 
  - **Radial Basis Functions:** Using Gaussian kernels to model non-linear functions and decision boundaries
  - **Least Squares Training:** Minimizing error between predicted and actual outputs
  - **K-means Clustering:** Initializing RBF centers by clustering input data

- **Data Visualization:** 
  - Plotting regression curves and decision boundaries
  - Visualizing data distributions and RBF activations

- **Version Control:**
  - [Git](https://git-scm.com/) for version control and collaborative development
  - [GitHub](https://github.com/) for code hosting, collaboration, and project management

## 💡 Why RBF Neural Network?

RBF networks are **simple yet powerful** tools for modeling non-linear relationships in data. This custom implementation showcases how these networks can solve complex problems like function approximation and classification with minimal computational overhead. The modular design makes it easy to adapt and extend for specific use cases, providing a hands-on learning experience in neural networks and machine learning.

## 📚 Getting Started

Follow these steps to use the RBF Neural Network with your dataset:

1. **Load the Data**

    First, load your training and test data along with their labels. Ensure your data is in an appropriate format (e.g., NumPy arrays, CSV files, etc.). Here's an example of how you might load the data for any dataset:

    ```python
    # Replace these lines with actual data loading code
    x_train = np.load('path/to/your/train_data.npy')
    y_train = np.load('path/to/your/train_labels.npy')
    x_test = np.load('path/to/your/test_data.npy')
    y_test = np.load('path/to/your/test_labels.npy')
    ```

    Ensure that `x_train` and `x_test` are your feature arrays and `y_train` and `y_test` are the corresponding labels.

2. **Flatten the Data (if necessary)**

    If your data has more than 2 dimensions (like images), flatten it to a 2D array where each row is a flattened sample. This step is not necessary for all datasets, so make sure to adjust based on your data's structure.

    ```python
    def flatten_data(data):
        if len(data.shape) > 2:  # If the data is multi-dimensional, flatten it
            return data.reshape(data.shape[0], -1)
        return data

    x_train_flattened = flatten_data(x_train)
    x_test_flattened = flatten_data(x_test)
    ```

3. **Normalize the Data**

    Normalize the data for better performance in the network. Adjust the normalization to suit your dataset's range (e.g., scaling between 0 and 1, or standardization).

    ```python
    def normalize_data(data, data_min, data_max):
        return (data - data_min) / (data_max - data_min)

    x_train_normalized = normalize_data(x_train_flattened, np.min(x_train_flattened), np.max(x_train_flattened))
    x_test_normalized = normalize_data(x_test_flattened, np.min(x_train_flattened), np.max(x_train_flattened))
    ```

4. **Visualize the Training Data (Optional)**

    If your data is visual (e.g., images), you can plot a subset of the samples to visualize them. This is especially useful for datasets like MNIST or CIFAR.

    ```python
    import matplotlib.pyplot as plt

    # Example visualization for image datasets
    fig, axs = plt.subplots(5, 5, figsize=(12, 12))
    axs = axs.reshape(-1)

    for i in range(25):
        ax = axs[i]
        ax.imshow(x_train[i].reshape(28, 28), cmap='gray')  # Adjust the reshape size if needed
        ax.axis('off')

    plt.show()
    ```

    **Note:** This step is optional and depends on whether your dataset consists of images or other types of data.

5. **Prepare a Subset of Data for Faster Experimentation (Optional)**

    If you're working with a large dataset, you can train the network on a smaller subset of data to speed up experimentation.

    ```python
    subset = 1500  # Adjust this number based on your dataset
    x_train_subset = x_train_normalized[:subset]
    y_train_subset = y_train[:subset]
    ```

6. **Initialize the Network**

    Set up the RBF network with a predefined number of hidden units (RBF centers) and classes. Adjust the number of hidden units based on your dataset's complexity.

    ```python
    num_hidden_units = 25  # Adjust based on your dataset
    num_features = x_train_subset.shape[1]
    centroids = np.random.normal(size=(num_hidden_units, num_features))
    num_classes = np.unique(y_train_subset).size

    num_data_points = x_train_subset.shape[0]
    assignments = np.random.randint(low=0, high=num_hidden_units, size=(num_data_points,))

    mnist_net = RBF_Net(num_hidden_units, num_classes)
    ```

7. **Train the Network**

    Initialize the network, train it on the subset of data, and evaluate the accuracy on both the training and test sets.

    ```python
    mnist_net.initialize(x_train_subset)
    mnist_net.train(x_train_subset, y_train_subset)

    # Training accuracy
    y_train_pred = mnist_net.predict(x_train_subset)
    train_set_accuracy = mnist_net.accuracy(y_train_subset, y_train_pred)
    print("Training Set Accuracy:", train_set_accuracy)

    # Testing accuracy
    y_test_pred = mnist_net.predict(x_test_normalized)
    test_set_accuracy = mnist_net.accuracy(y_test, y_test_pred)
    print("Testing Set Accuracy:", test_set_accuracy)
    ```

8. **Visualize Hidden Layer Prototypes**

    After training, visualize the network's hidden layer prototypes as images (or any other representation, depending on your data type).

    ```python
    prototypes = mnist_net.get_prototypes()
    prototypes = np.reshape(prototypes, [prototypes.shape[0], 28, 28])  # Adjust reshape if needed

    cols = rows = 5
    fig, axes = plt.subplots(nrows=rows, ncols=cols)
    for i in range(rows):
        for j in range(cols):
            axes[i, j].imshow(prototypes[i*rows + j])
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

    plt.show()
    ```

By following these steps, you can apply the RBF Neural Network to any dataset. Make sure to replace the dataset loading code, adjust the dimensions where necessary, and fine-tune the network parameters based on your data's complexity and task requirements.

## 📈 Example Project: Handwritten Digit Classification with MNIST Data

This project demonstrates the application of a custom Radial Basis Function (RBF) Neural Network for classifying handwritten digits from the **MNIST dataset**. The network uses Gaussian kernels to map input data into a higher-dimensional space, and then a linear model is used for classification. Below are the steps to prepare and train the RBF network using the MNIST dataset.

### 1. **Data Loading**

The MNIST dataset is loaded using `numpy`. Ensure you have the training and test data stored as `.npy` files. If you're using a different dataset, simply update the data loading section.

```python
x_train_mnist = np.load('data/mnist_train_data.npy')
y_train_mnist = np.load('data/mnist_train_labels.npy')
x_test_mnist = np.load('data/mnist_test_data.npy')
y_test_mnist = np.load('data/mnist_test_labels.npy')
```

### 2. **Flatten and Normalize Data**

For image datasets like MNIST, flatten the 28x28 pixel images into 1D arrays to make them suitable for input into the RBF network. Normalization is also performed to scale the data between 0 and 1.

```python
def flatten_data(data):
    if len(data.shape) > 2:
        return data.reshape(data.shape[0], -1)
    return data

x_train_flattened = flatten_data(x_train_mnist)
x_test_flattened = flatten_data(x_test_mnist)

# Normalize the data
def normalize_data(data, data_min, data_max):
    return (data - data_min) / (data_max - data_min)

x_train_normalized = normalize_data(x_train_flattened, np.min(x_train_flattened), np.max(x_train_flattened))
x_test_normalized = normalize_data(x_test_flattened, np.min(x_train_flattened), np.max(x_train_flattened))

print("x_train_mnist.shape", x_train_normalized.shape)
print("y_train_mnist.shape", y_train_mnist.shape)
print("x_test_mnist.shape", x_test_normalized.shape)
print("y_test_mnist.shape", y_test_mnist.shape)
```

### 3. **Train the RBF Network**

In this step, the architecture of the RBF network is defined, initialized, and then trained using the training data.

```python
subset = 1500
x_train_subset = x_train_normalized[:subset]
y_train_subset = y_train_mnist[:subset]

num_hidden_units = 25
num_features = x_train_subset.shape[1]
centroids = np.random.normal(size=(num_hidden_units, num_features))
num_classes = np.unique(y_train_subset).size

# Randomly assign data points to centroids
num_data_points = x_train_subset.shape[0]
assignments = np.random.randint(low=0, high=num_hidden_units, size=(num_data_points,))

# Initialize the RBF network
mnist_net = RBF_Net(num_hidden_units, num_classes)

# Calculate cluster distances and initialize weights
clust_mean_dists = mnist_net.avg_cluster_dist(x_train_subset, centroids, assignments, kmeansObj)
mnist_net.initialize(x_train_subset)

# Train the network
mnist_net.train(x_train_subset, y_train_subset)

# Training accuracy
y_train_pred = mnist_net.predict(x_train_subset)
train_set_accuracy = mnist_net.accuracy(y_train_subset, y_train_pred)
print("Training Set Accuracy:", train_set_accuracy)
```

### 4. **Evaluate the Model**

Once the network is trained, its performance on the test set is evaluated to assess how well it generalizes to new, unseen data.

```python
# Test the network
y_test_pred = mnist_net.predict(x_test_normalized)
test_set_accuracy = mnist_net.accuracy(y_test_mnist, y_test_pred)

# Print testing accuracy
print("Testing Set Accuracy:", test_set_accuracy)
```

### 5. **Visualize the Network's Hidden Layer Prototypes**

The prototypes in the hidden layer represent the centroids that the network has learned to identify key features of the data. The prototypes are visualized to get an idea of the features the RBF network has learned.

```python
# Visualize network hidden layer prototypes
prototypes = mnist_net.get_prototypes()

# Reshape prototypes to match image dimensions (28x28)
prototypes = np.reshape(prototypes, [prototypes.shape[0], 28, 28])

# Create a 5x5 grid to display the prototypes
cols = rows = 5
fig, axes = plt.subplots(nrows=rows, ncols=cols)
for i in range(rows):
    for j in range(cols):
        axes[i, j].imshow(prototypes[i*rows + j], cmap='gray')
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])

# Show the plot
plt.show()
```

### 6. **Evaluate the Model's Performance**

After training the RBF network, the performance is evaluated on both the training and test sets. The accuracy metrics help determine how well the network has generalized to unseen data.

```python
# Evaluate on the training set
y_train_pred = mnist_net.predict(x_train_subset)
train_set_accuracy = mnist_net.accuracy(y_train_subset, y_train_pred)
print("Training Set Accuracy:", train_set_accuracy)

# Evaluate on the testing set
y_test_pred = mnist_net.predict(x_test_normalized)
test_set_accuracy = mnist_net.accuracy(y_test_mnist, y_test_pred)
print("Testing Set Accuracy:", test_set_accuracy)
```
