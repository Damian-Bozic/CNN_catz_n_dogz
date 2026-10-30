### 42 Rush - Neural Networks: Cats and Dogs Image Classifier

An end-to-end Machine Learning pipeline developed for the School 42 AI/ML Discovery Piscine. This project implements a Deep Convolutional Neural Network (CNN) in PyTorch to classify images into two categories: **Cat** or **Dog**. It includes a modern dependency management system via uv and a complete graphical Desktop User Interface (UI) using Tkinter to fulfill the bonus requirements. 

### 🚀 How to Install and Run

This project uses uv for lightning-fast and reliable Python environment management. 

### 1. Synchronize the Environment

Run this command in the project root directory to automatically create a virtual environment (.venv) and install all required dependencies (PyTorch, Torchvision, Scikit-Learn, Matplotlib, Datasets, Pillow): 

```bash

uv sync
```



### 2. Train the Model

To start the training process, execute the training script. This script will automatically download the dataset from Hugging Face into a local temporary cache directory (/tmp/ai-ml-piscine-rush), initialize the CNN layers, and train the model: 

```bash

uv run python3 train.py
```


*Note: After training is finished, the learned neural weights will be safely exported to /tmp/cat_dog_cnn.pth.* 

### 3. Evaluate the Model (CLI)

To run a bulk statistical test on the verification set, calculate scores, and render the graphical Confusion Matrix diagram plot file (confusion_matrix.png), run: 

```bash

uv run python3 evaulate.py
```



### 4. Test Single Images via Command Line

If you want to quickly test a single local image file from your terminal without opening the graphical window, use: 

```bash

uv run python3 predict.py path/to/your/image.jpg
```



### 5. Launch the Graphical Desktop User Interface (Bonus UI)

To launch the complete application containing interactive image browsing, single image manual loading, automated random dataset evaluation scoring displays, and the embedded live Confusion Matrix diagram window, run: 

```bash

uv run python3 gui.py
```


### 📊 Dataset & Data Pipeline

* **Source:** We utilize the official, high-quality open-source **microsoft/cats_vs_dogs** dataset hosted on **Hugging Face Datasets**.
* **Dataset Size:** The raw collection contains a total of **23,410 images** (~11,700 cats and ~11,700 dogs).
* **Data Partitioning (Train/Test Split):** 

  * **80% Training Set (~18,728 images):** Used strictly by the network during the training loop to study patterns.
  * **20% Testing Set (~4,682 images):** Kept completely hidden from the network during training. It serves as an unseen validation ground to gauge true real-world real performance.
* **Data Augmentation (In-Flight Transformation):** To prevent the network from memorizing the specific images (overfitting), our dataset.py pipeline applies real-time mutations to the training samples every single epoch. This includes random rotations up to 20 degrees, horizontal mirror-flips, image down-scaling to a standardized resolution of 64x64 pixels, and dynamic pixel normalization.

### 🧠 Neural Network Architecture (model.py)

Our model is a custom **Convolutional Neural Network (CNN)** designed specifically for computer vision. Standard dense neural networks fail at processing images because they treat pixels as isolated dots, completely losing spatial awareness. A CNN solves this by looking at images through structural localized filters. 

### How Data Flows Through Our Pipeline (The Forward Pass):

1. **Input Layer:** Receives a standardized digital tensor of size (3, 64, 64)—representing 3 color channels (RGB) and a resolution of 64x64 pixels.
2. **Convolutional Layers (conv1 to conv4):** These layers use matrices called kernels that slide across the image. 

  * Early layers (conv1, conv2) activate upon primitive spatial features like straight lines, edges, and simple contrast changes.
  * Deeper layers (conv3, conv4) assemble those basic lines into complex patterns like curves, textures, and biological shapes (ears, eyes, whiskers, noses).
3. **Batch Normalization (bn1 to bn4):** Stabilizes and accelerates the mathematical training distributions between layers.
4. **Max Pooling (pool):** Sub-samples the dimensional space by selecting only the highest pixel activation scores in a 2x2 window. This reduces the canvas size from 64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4, lowering computational weight while making the model translation-invariant (it can detect a cat regardless of where it stands in the frame).
5. **Flattening & Fully Connected Classification (fc1, fc2):** Converts the final 3D feature representation map (128 channels * 4 * 4 pixels = 2048 values) into a flat 1D array. A Dropout(0.5) layer randomly deactivates 50% of neurons to block structural co-dependency. The final linear projection yields exactly **2 raw scalar numbers (logits)** corresponding to the classification scores of the two target categories: [Cat, Dog].

### 🔄 The Core Training Process (train.py)

Training a neural network is an iterative optimization loop composed of distinct consecutive phases: 

1. **Forward Pass:** An image batch is fed into the network. The weights currently assigned to the neural channels perform matrix multiplications and output a raw prediction guess.
2. **Loss Calculation (The Cost):** We measure how incorrect the network's current guess is relative to the true target label using **Cross-Entropy Loss**. 

  * If the model is highly confident about a wrong classification answer, the system generates a massive **Loss penalty score**.
  * The core target of the training execution is to drive this loss scalar value down as close to zero as possible.
3. **Backward Pass (Backpropagation):** PyTorch automatically traces backward through the mathematical computation graph, calculating the exact derivatives (gradients) of the error relative to every single internal connection weight in the system.
4. **Weight Optimization (Adam Optimizer):** The optimization algorithm takes those gradient calculations and applies minor numerical adjustments to the millions of network wights, shifting them in the direction that decreases the loss score.
5. **Epoch Progression:** This entire chain of events repeats across the full training dataset batch-by-batch. One complete cycle through all 18,728 training photos is called **1 Epoch**.

### 📈 Clarifying Key Machine Learning Concepts

When defending this project during peer-evaluations, you must be ready to clearly distinguish between these fundamental diagnostic definitions: 

### 1. Accuracy vs. Precision

* **Accuracy (Overall Correctness):** 

  * *Question answered:* "Out of **all** the thousands of images in our test dataset, what percentage did the model classify correctly?"
  * *Formula:* (Correct Cats + Correct Dogs) / Total Test Images
* **Precision (Target Reliability / Exactness):** 

  * *Question answered:* "When the model points at an image and asserts **'This is a Dog'**, what is the mathematical probability that it actually is a dog?"
  * *Why it matters:* A model can have high overall accuracy but terrible precision if it is overly eager and constantly falsely alarms one specific class over another due to structural bias.

### 2. Understanding the Confusion Matrix Axis

The Confusion Matrix is a direct 2x2 grid tracking correct versus incorrect classifications across the unseen evaluation pool. 

* **Horizontal Rows (True Classes / Ground Truth):** Represent the actual reality of what the animal in the picture is. 

  * *Row 1:* Pictures that are definitively known to be Cats.
  * *Row 2:* Pictures that are definitively known to be Dogs.
* **Vertical Columns (Predicted Classes):** Represent the choices made by your network model. 

  * *Column 1:* Every single image your network decided to tag as a Cat.
  * *Column 2:* Every single image your network decided to tag as a Dog.

### Reading the Intersections:

* **Top-Left Cell (True Cats):** Actual Cats that your model correctly classified as Cats.
* **Bottom-Right Cell (True Dogs):** Actual Dogs that your model correctly classified as Dogs.
* **Top-Right Cell (False Dogs):** Cats that your model mistakenly flagged as Dogs.
* **Bottom-Left Cell (False Cats):** Dogs that your model mistakenly flagged as Cats.

*Defense Tip: Precision is calculated vertically down a column. Precision for Dogs is simply the value of the True Dogs cell divided by the total sum of everything in that whole column.*