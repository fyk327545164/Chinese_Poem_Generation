# Chinese_Poem_Generation

This is a base-line model to generate a Chinese poem with the input of a user-defined title.

## Requirement:

* Python 3.7

* TensorFlow 1.13

## Model Structure

#### Training
The training process could be treated as a multi-task learning.  
  
Several Encoder-Decoder are built to be trained:  
* title-encoder --> sent1-decoder:  
--- with attention to title-output  
--- Bi-LSTM applied
* sent1-encoder --> sent2-decoder:  
--- with attention to sent1-output  
--- multi-layer-LSTM applied
* sent1-encoder+sent2-encoder --> sent3-decoder  
--- with attention to title-output    
--- multi-layer-LSTM applied
  
During training, 3 batches of input are feed into the model.    
Each batch of data is to learn from the corresponding task(encoder-decoder).   
  
Perplexity is used as the eval metrics during training process.
#### Inference
The user inferences the model with a title as the input.  
  
Inference Structure:  
* title-encoder --> sent1-decoder  
--- input: title  
--- output: title_output, title_state, sent1_out  
* sent1-encoder --> sent2-decoder  
--- input: sent1_out  
--- output: sent1_output, sent1_state, sent2_out
* sent2-encoder  
--- input: sent2_out  
--- output: sent2_output, sent2_state
* sent3-decoder  
--- input: sent1_state, sent2_state, title_output  
--- output: sent3_out
* sent3-encoder(sent1-encoder) --> sent4-decoder(sent2-decoder)  
--- input: sent3_out  
--- output: sent3_output, sent3_state, sent4_out
* Final-Output: [sent1_out, sent2_out, sent3_out, sent4_out]

## Implementation Details:
* Segmentation & Embedding:  
Since characters in a poem sentence are relatively independent, there is no need to
do word segmentation and word embedding. Instead, character-level embedding are used here.
* Bi-directional & Multi-Layer LSTM:  
Initially, the model is built with only forward-LSTM structure. During the training process, 
however, the loss could be quickly decreased from 26 to 18 after hundreds of steps, but there
 is no update after that. Also, the inference result is very bad. Instead of the improper 
 learning-rate, the problem is the structure of the model. I apply the Bi-LSTM in the title 
 encoding, and multi-layer LSTM in the sentence encoding and decoding. The result gets much better 
 than before.
* Training on GPU:  
Training on GPU gets tens of time faster than on CPU.
* TensorBoard has been used during training time

## Advanced Model: Transformer
## Sources Being Used

Dataset used from: [This Repository](https://github.com/chinese-poetry/chinese-poetry)  
    
Conversion from traditional character to simplified character:  
* [langconv](https://raw.githubusercontent.com/skydark/nstools/master/zhtools/langconv.py)  
* [zh_wiki](https://raw.githubusercontent.com/skydark/nstools/master/zhtools/zh_wiki.py)
