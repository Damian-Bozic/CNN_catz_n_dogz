---
description: What is the subject of the project
applyTo: '**'
---
# General instructions
Unless explicitly specified, the following rules will apply to this project.
* This subject is the one and only trustable source. Don’t trust any rumor.
* Your assignments will be evaluated by your peers.
* You have a question? Ask your left neighbor. Otherwise, try your luck with your right neighbor.
* You must read the examples thoroughly. They can reveal requirements that are not obvious in the assignment’s description.
* Good luck and good work on this project!

# Common Instructions
Congratulations on making it this far!

For this finale module, you will work on a rush (a project in pairs, focusing on a more complex exercise).
You will need to automatically sort images of dogs and cats using a machine learning program. Building on your first week’s training, your task is to develop an image classification tool using a neural network.
This tool will enable users to upload images and automatically classify them into two categories: "cat" and "dog".
* Mandatory Part: You must complete the mandatory part to pass the project.
  * Step 1: Data Preparation: Download and prepare a dataset containing images of cats and dogs.
  * Step 2: Model Creation: Build a Convolutional Neural Network (CNN).
  * Step 3: Model Training: Train the model using the dataset images.
  * Step 4: Model Evaluation: Assess the model’s accuracy on a test dataset.
*  Bonus Part: You are not required to complete the bonus part.
  * User Interface: Develop a user interface that allows users to upload images and displays the predicted category.

# Exercise 00

| Exercise 00                |
|----------------------------|
| Exercise 00: The Model     |
| Allowed functions: **All** |

## Step 1: Data Preparation

You need to gather images of cats and dogs. You must organize them into two folders:
*  One folder with labeled images for training your model (training data)
* One folder with unlabeled images fot validating your model (testing data)

> [!INFO]
> Be resourceful, look for pre-existing datasets

> [!TIP]
> Preferably open-source datasets.

> [!WARNING]
> Ensure your data sets must be large enough to train your model.

## Step 2: Model Creation

Your task is to create your model.

> [!INFO]
> You can use frameworks such as `PyTorch`, `Keras`, or `TensorFlow` to create your model.

> [!TIP]
> While there are many examples available online, avoid simply copying and pasting. Ensure you fully understand the concepts and implementation

## Step 3: Model Training

Train your model using the labeled images from your training dataset.


## Step 4: Model Evaluation

Evaluate your model using the unlabeled images from your testing dataset.

Your model’s accuracy should exceed $50\%$, as $50\%$ represents random chance (e.g., a coin toss)

Ideally, your model should achieve an accuracy of at least $90\%$

# BONUS

| Exercise 00                |
|----------------------------|
| Bonus Part: User Interface |
| Allowed functions: **All** |


Congratulations on reaching this point!

Now you can develop a user interface (UI), a graphical interface that enables users to:
* Load and display images from your dataset.
* Show predictions from your model (including Accuracy and Precision).
* Display a confusion matrix.
* Create a random validation set to evaluate the model’s accuracy.
* Include any additional features you find relevant or useful.
