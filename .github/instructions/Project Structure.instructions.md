---
description: This file describes the existing CNN project structure and implementation details.
applyTo: '**/*'
---

# Existing CNN Reference

This document records the implementation inspected before adding the desktop
application. It is intentionally separate from the project's normal
documentation so the existing workflow remains unchanged.

## Project structure

- `model.py` defines the PyTorch `CatDogCNN` model.
- `dataset.py` downloads and prepares the Hugging Face
  `microsoft/cats_vs_dogs` dataset and returns PyTorch data loaders.
- `train.py` trains a new model and writes its state dictionary.
- `evaulate.py` (the filename is intentionally misspelled) evaluates the saved
  model, prints a classification report, and writes a confusion matrix.
- `predict.py` loads the saved model and predicts one local image.
- `testing_data/` contains checked-in JPEG images in class directories
  (`cats/` and `dogs/`).
- `testowe/` contains five checked-in PNG images used as an additional local
  image collection; it has no class subdirectories.
- `confusion_matrix.png` is the currently checked-in evaluation plot.
- `pyproject.toml` and `uv.lock` describe the Python environment.

The virtual environment and Python cache directories are local environment
artifacts and are not part of the application workflow.

## CNN architecture and initialization

`CatDogCNN` in `model.py` accepts RGB tensors sized `3 x 64 x 64`. It applies
four convolution/batch-normalization/ReLU/max-pooling blocks:

1. `3 -> 32` channels
2. `32 -> 64` channels
3. `64 -> 128` channels
4. `128 -> 128` channels

Each max-pooling layer halves the spatial dimensions, reducing `64 x 64` to
`4 x 4`. The flattened 2048 values are passed through a 256-unit linear
layer, ReLU, dropout (`p=0.5`), and a two-unit output layer. The model is
created with `CatDogCNN()` and moved to CUDA when available, otherwise CPU.

## Application execution flow

The existing command-line entry points are:

- `uv run python train.py`: invokes `train_model(epochs=10)` from the
  `__main__` block.
- `uv run python evaulate.py`: invokes `evaluate_model()`.
- `uv run python predict.py <image-path>`: invokes `predict_image()` for the
  supplied path.

The new desktop entry point is `python gui.py` (or
`uv run python gui.py` in the managed environment). It imports the existing
training, evaluation, and prediction modules rather than replacing them.

## Data loading and preprocessing

`get_data_loaders(batch_size=32)` in `dataset.py` calls
`datasets.load_dataset("microsoft/cats_vs_dogs", split="train",
cache_dir="/goinfre/krbartos")`. The complete split is deterministically split
80/20 using `torch.Generator().manual_seed(42)`.

The `HuggingFaceCatsDogs` wrapper reads the `image` and `labels` fields,
converts non-RGB images to RGB, and applies the supplied transform.

Training transforms resize to `64 x 64`, apply a random rotation up to 20
degrees and a random horizontal flip, then convert to tensors and normalize
with means `[0.485, 0.456, 0.406]` and standard deviations
`[0.229, 0.224, 0.225]`.

Test transforms resize to `64 x 64`, convert to tensors, and apply the same
normalization without augmentation. Training uses a shuffled data loader with
`num_workers=2`; testing uses an unshuffled loader with the same worker count.

## Training

`train_model(epochs=5, batch_size=32, lr=0.001)` selects the device, loads
both data loaders, creates a new `CatDogCNN`, and uses Adam with learning rate
`0.001` and `CrossEntropyLoss`. Each epoch trains over all training batches,
then evaluates accuracy over all test batches. It prints epoch loss, training
accuracy, and test accuracy.

After the final epoch it saves only the model state dictionary to
`/tmp/cat_dog_cnn.pth`. The script's command-line default is 10 epochs,
although the function default is 5. The function does not return metrics or
accept a progress callback.

## Evaluation

`evaluate_model()` loads the test loader with batch size 64, requires
`/tmp/cat_dog_cnn.pth`, and runs inference over the test split. It prints a
scikit-learn classification report using the labels `Kot` and `Pies`, then
creates and saves a confusion matrix as `confusion_matrix.png` relative to the
current working directory. It does not return the metrics and returns early
with a printed error when weights are missing.

## Single-image prediction

`predict_image(image_path)` checks that the path exists, opens and converts it
to RGB with Pillow, applies the same resize and normalization as test data,
loads `/tmp/cat_dog_cnn.pth`, and runs one inference pass. It maps output index
0/1 to `kot`/`pies` and prints the highest softmax probability and class.
Errors are printed and the function returns without a value; it does not
return a prediction object.

## Local dataset organization

The checked-in `testing_data/` directory is organized as:

```text
testing_data/
  cats/*.jpg
  dogs/*.jpg
```

The class directory name is the available label for browser display. The
`testowe/` directory contains standalone `*.png` files and therefore has no
label metadata. Neither local directory is consumed by `get_data_loaders()`;
the CNN training and evaluation data come from Hugging Face.

The GUI browser discovers image files lazily from these existing directories
and opens only the image currently being displayed, rather than loading the
dataset into memory. Because this repository has no local `training_data/`
directory and the requested training browser source was `testowe/`, the GUI
labels that tab as the training image browser for this local collection. The
files in `testowe/` remain unlabeled because they have no class metadata.

## Outputs and dependencies

- Model weights: `/tmp/cat_dog_cnn.pth` (written by training and read by
  evaluation/prediction).
- Evaluation image: `confusion_matrix.png` in the current working directory.
- Dataset cache: `/goinfre/krbartos` as configured in `dataset.py`.
- Required packages: Python `>=3.10`, PyTorch, torchvision, Pillow,
  Hugging Face `datasets`, scikit-learn, and matplotlib.
- The GUI uses Python's standard-library Tkinter and
  `concurrent.futures`; no additional GUI dependency is required. Pillow is
  already a project dependency and is used for previews.

## GUI integration considerations

The original operations are synchronous and print their progress rather than
returning it. The GUI therefore runs each operation in a worker thread so the
window remains responsive. Captured stdout is also written unchanged to the
terminal stream from which `gui.py` was launched, so original prints from the
existing Python files remain visible in the CLI.

`train_model()` and `evaluate_model()` retain their existing defaults and now
accept an optional progress callback. Training calls it after each completed
epoch; evaluation calls it after each completed test batch. The GUI uses these
callbacks to update determinate progress bars. Prediction has no intermediate
progress signal, so its bar completes when the existing prediction function
returns. Training, evaluation, and prediction buttons are locked while any
operation is running.

The GUI parses the existing scikit-learn classification report's `accuracy`
line and displays the percentage in the operation status area after
evaluation. Prediction can show the selected image and the existing printed
result; because `predict_image()` does not return a value, the GUI extracts
the existing result line for display without implementing another inference
pipeline.

The GUI does not alter model architecture, data loading, training,
evaluation, prediction, datasets, or weight files. A model must be trained
before evaluation or prediction can succeed. The existing hard-coded dataset
cache and `/tmp` weight path remain compatibility constraints.
