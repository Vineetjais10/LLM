## Report on Fine-Tuning a Multi-Modal Model on a Medical Imaging Dataset
The aim of this report is to outline the process of fine-tuning a multi-modal model, specifically LLaVA (Language Learning with Vision and Audio), on a medical imaging dataset. We selected the MIMIC-CXR dataset from Kaggle, which includes chest X-ray images and associated radiology reports.

Dataset Selection
Dataset: MIMIC-CXR
Source: Kaggle - MIMIC-CXR Dataset

Content:

Images: Over 350,000 chest X-ray images.
Text: Corresponding radiology reports providing textual descriptions of the findings.

# Code Description
Environment Setup:
The first step is to ensure that all the necessary libraries are installed. These include torch, torchvision, and transformers.

The provided code outlines the steps for fine-tuning a multi-modal model (specifically LLaVA) using the MIMIC-CXR dataset, which contains chest X-ray images and corresponding radiology reports. The process includes setting up the environment, loading and preprocessing the data, fine-tuning the model, and evaluating its performance.
# Data Loading and Preprocessing
Custom Dataset Class: MedicalDataset

Parameters:
image_dir: Directory containing images.
annotations_file: File with annotations (JSON format).
transform: Optional transformations to apply to the images.
Methods:
__len__(): Returns the number of items in the dataset.
__getitem__(idx): Loads and returns an image and its corresponding caption by index.
Image Transformations:

Using torchvision.transforms.Compose to resize images to 224x224 pixels and convert them to tensors.
DataLoader:

Loads the dataset and allows batching and shuffling of data.



# Model Fine-Tuning
Loading the Model:

VisionEncoderDecoderModel from Hugging Face, which integrates a vision encoder (ViT) and a text decoder (GPT-2).
ViTFeatureExtractor for processing images before feeding them into the model.
BertTokenizer for tokenizing text captions.
Fine-Tuning Loop:

Optimizer: AdamW optimizer for training.
Training the model for a specified number of epochs.
For each batch, images are processed using the feature extractor and captions using the tokenizer.
The model computes the loss, backpropagates, and updates the weights.
