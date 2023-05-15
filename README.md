# **Character Level Language Models**

Based on trained database of text the model will generate new text in the same style based on character level prediction. For example if we feed databes of names the model will generate new names in the same style.

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

## Prerequisites

- `pip install torch`
- `pip install ipynb`

## License
MIT