# Next-Word Prediction using LSTM and GRU

## Overview
This project implements a Next-Word Prediction model using **LSTM (Long Short-Term Memory)** and **GRU (Gated Recurrent Unit)** networks. The model is trained on Shakespeare's *Hamlet* dataset to predict the next word in a given sequence of text.

## Features
- Uses **LSTM and GRU** for text prediction
- Trained on Shakespeare's *Hamlet* dataset
- Tokenization and text preprocessing with **TensorFlow/Keras**
- Model evaluation and comparison between LSTM and GRU
- Generates text based on given input

## Dataset
The dataset used is **Shakespeare's Hamlet** text. The text is tokenized and converted into sequences for training.

## Installation
### Prerequisites
Make sure you have Python installed (preferably 3.8+). Install the required libraries using:
```bash
pip install numpy pandas tensorflow keras nltk
```

### 1. Predict Next Word
To generate the next word prediction, run:
```bash
python predict.py "To be or not to"
```
Example Output:
```
Input: "To be or not to"
Predicted Next Word: "be"
```

## Model Architecture
Both LSTM and GRU models use:
- **Embedding layer** for word representation
- **Bidirectional LSTM/GRU layers** for sequential learning
- **Dense layer with Softmax activation** for word prediction


## Future Improvements
- Use **transformers** for improved text generation
- Experiment with different datasets
- Deploy the model as a web app

## Contributing
Feel free to fork this repository and improve the model. Pull requests are welcome!

## License
This project is open-source and available under the **MIT License**.


