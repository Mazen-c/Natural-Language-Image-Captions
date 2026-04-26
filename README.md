# Image Captioning with CNN-RNN Architecture

## Overview
This project implements an **Image Captioning system** that generates descriptive text captions for images using a combination of Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN). The system is trained on the MS COCO (Microsoft Common Objects in Context) dataset.

## Architecture

### Encoder (CNN) - `EncoderCNN`
- Uses a pre-trained **ResNet50** model for visual feature extraction
- Removes the final classification layer to extract intermediate features (2048-dimensional vectors)
- Projects features to an embedding space using a linear layer
- Includes batch normalization for training stability

### Decoder (RNN) - `DecoderRNN`
- **Step (b): Embedding Layer** - Converts word indices to dense vectors (embedding dimension: 256)
- **Step (c): RNN Decoder (LSTM)** - Processes sequences with LSTM cells (hidden size: 512)
  - Combines image features with word embeddings
  - Captures sequential dependencies in caption generation
- **Step (d): Final Dense Layer** - Projects LSTM outputs to vocabulary probabilities
  - Maps hidden states to vocabulary space for word prediction

## Dataset
- **Source**: MS COCO 2017 (train2017)
- **Size**: ~82.8K images with 5 captions per image
- **Paths**:
  - Captions: `/content/annotations/captions_train2017.json`
  - Images: `/content/train2017`

## Data Pipeline

### 1. **Data Loading**
   - Loads JSON annotations and builds image ID to filename mapping
   - Organizes captions by image file

### 2. **Text Preprocessing**
   - Converts to lowercase
   - Removes punctuation and special characters
   - Filters short words (< 2 characters)
   - Adds `<start>` and `<end>` tokens for sequence generation

### 3. **Vocabulary Construction**
   - Tokenizes captions using Keras Tokenizer
   - Applies frequency filtering (minimum 5 occurrences)
   - Creates word-to-index and index-to-word mappings
   - Special tokens: `<pad>` (index 0), `<start>`, `<end>`, `<unk>`

### 4. **Training Data Preparation**
   - Converts captions to token sequences
   - Creates input-output pairs for training
   - Pads sequences to maximum caption length
   - Stores image references for feature lookup

### 5. **Image Feature Extraction**
   - Resizes images to 224×224 (ImageNet standard)
   - Extracts 2048-dimensional feature vectors using ResNet50
   - Stores features as pickle files for efficient loading

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `embed_size` | 256 | Word embedding and image feature dimension |
| `hidden_size` | 512 | LSTM hidden state dimension |
| `vocab_size` | 10000 | Total vocabulary size (adjust based on filtered vocabulary) |
| `num_layers` | 1 | Number of LSTM layers |
| `dropout` | 0.5 | Dropout rate for regularization |
| `MIN_FREQUENCY` | 5 | Minimum word frequency threshold |

## Output Files

The notebook saves the following files:
- `tokenizer.pkl` - Fitted tokenizer for caption encoding
- `vocab_mappings.pkl` - Word-to-index and index-to-word mappings
- `coco_image_features.pkl` - Extracted image feature vectors

## Training Flow

```
Images (224×224)
    ↓
[EncoderCNN - ResNet50]
    ↓
Image Features (256-dim)
    ↓
[Combine with Word Embeddings]
    ↓
[DecoderRNN - LSTM]
    ↓
Vocabulary Probabilities
    ↓
Next Word Prediction
```

## Usage

1. **Run Data Loading** - Execute the first cell to load and explore MS COCO data
2. **Preprocess Text** - Clean captions and build vocabulary
3. **Extract Features** - Run CNN feature extraction (time-intensive for full dataset)
4. **Initialize Models** - Create encoder and decoder instances
5. **Train** - Use prepared data to train the caption generation model

## Requirements

- TensorFlow/Keras (for text preprocessing)
- PyTorch (for CNN and RNN models)
- torchvision (for ResNet50 and image transforms)
- NumPy, Pandas, tqdm (for data handling and progress tracking)
- PIL (for image loading)

## Notes

- The full MS COCO dataset is very large; consider using a subset for initial testing
- Feature extraction is computationally expensive; pre-computed features are saved to disk
- Vocabulary size should be adjusted based on frequency filtering results
- The decoder expects image features and caption sequences during training