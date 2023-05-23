# **Character Level Language Models**

Based on trained database of text the model will generate new text in the same style based on character level prediction. For example if we feed databes of names the model will generate new names in the same style. 

Files duplicate some functions and concepts and this is intended so they are independent of each other and can be read and run separately. Also this repo is purely educational and most of the code is done from scratch to learn the concepts.

Implementation follows a few key papers:

- Bigram (one character predicts the next one with a lookup table of counts)
- MLP, [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
- CNN, [WAVENET: A GENERATIVE MODEL FOR RAW AUDIO](https://arxiv.org/pdf/1609.03499.pdf)
- RNN, [Recurrent neural network based language model](https://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf)
- LSTM, [Generating Sequences With Recurrent Neural Networks](https://arxiv.org/pdf/1308.0850.pdf)
- GRU, [On the Properties of Neural Machine Translation: Encoder-Decoder Approaches](https://arxiv.org/pdf/1409.1259.pdf)
- Transformer, [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)

This is the code I created as part of learning neural networks during Andrej Karpathy lecture series "Neural Networks: Zero to Hero".

## Key topics learned during implementation:

- Bigram
- torch.tensor/zeros/ones
- torch.multinomial
- broadcasting semantics
- product of probabilities
- likelihood & log-likelihood
- model smoothing
- one-hot encoding
- matrix multiplication
- softmax
- weight regularization loss
- word embeddings
- torch.concat
- torch.unbind
- torch.randn
- tensor_name.view
- torch.nn.functional.cross_entropy
- minibatch
- choosing learning rate
- overfitting
- idea of training 80% / dev or validation 10% / test 10% splits of data
- training set is to optimize parameters like weights, biases, embeddings
- validation set is to choose hyperparameters like learning rate, regularization, layer size, number of layers etc.
- test set is to evaluate the final model

## Prerequisites

- `pip install torch`
- `pip install ipynb`

## License
MIT