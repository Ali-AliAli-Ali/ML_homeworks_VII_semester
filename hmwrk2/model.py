import torch
from typing import Type
from torch import nn
from torch.distributions.categorical import Categorical
from dataset import TextDataset


class LanguageModel(nn.Module):
    def __init__(self, dataset: TextDataset, embed_size: int = 256, hidden_size: int = 256,
                 rnn_type: Type = nn.RNN, rnn_layers: int = 1):
        """
        Model for text generation
        :param dataset: text data dataset (to extract vocab_size and max_length)
        :param embed_size: dimensionality of embeddings
        :param hidden_size: dimensionality of hidden state
        :param rnn_type: type of RNN layer (nn.RNN or nn.LSTM)
        :param rnn_layers: number of layers in RNN (num_layers in class nn.RNN)
        """
        super(LanguageModel, self).__init__()
        self.dataset = dataset  # required for decoding during inference
        self.vocab_size = dataset.vocab_size
        self.max_length = dataset.max_length

        """
        YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        Create necessary layers
        """
        if (rnn_type not in [nn.RNN, nn.LSTM]):
            raise ValueError('Unknown type of layer. nn.RNN or nn.LSTM are available')

        self.embedding = nn.Embedding(
            self.vocab_size, 
            embed_size, 
            padding_idx=self.dataset.pad_id)
        self.rnn = rnn_type(
            embed_size, 
            hidden_size, 
            num_layers=rnn_layers, 
            batch_first=True)  
        self.linear = nn.Linear(
            hidden_size, 
            self.vocab_size, 
            bias=True)

    def forward(self, indices: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Compute forward pass through the model and
        return logits for the next token probabilities
        :param indices: LongTensor of encoded tokens of size (batch_size, length)
        :param lengths: LongTensor of lengths of size (batch_size, )
        :return: FloatTensor of logits of shape (batch_size, length, vocab_size)
        """

        """
        YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        Convert indices to embeddings, pass them through recurrent layers
        and apply output linear layer to obtain the logits
        """
        packed_input = nn.utils.rnn.pack_padded_sequence(
            self.embedding(indices.abs()),
            lengths,
            batch_first=True,
            enforce_sorted=False
        )
        packed_output, final_state = self.rnn(packed_input)
        output, lengths = nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True
        )
        logits = self.linear(output)
        return logits

    @torch.inference_mode()
    def inference(self, prefix: str = '', temp: float = 1.) -> str:
        """
        Generate new text with an optional prefix
        :param prefix: prefix to start generation
        :param temp: sampling temperature
        :return: generated text
        """
        self.eval()
        """
        YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        Encode the prefix (do not forget the BOS token!),
        pass it through the model to accumulate RNN hidden state and
        generate new tokens sequentially, sampling from categorical distribution,
        until EOS token or reaching self.max_length.
        Do not forget to divide predicted logits by temperature before sampling
        """
        
        indices = torch.tensor([[1]]) if (prefix == '') else \
                  torch.tensor([self.dataset.text2ids(prefix)], dtype=torch.int32)
        
        embeds = self.embedding(indices)
        output, final_state = self.rnn(embeds)
        logits = self.linear(output) / temp

        new_tokens = Categorical(logits=logits[:, -1:]).sample()
        tokens = torch.cat([indices, new_tokens], dim=1)
        while (tokens.shape[1] < self.max_length) and (new_tokens.item() != self.dataset.eos_id):
                embeds = self.embedding(new_tokens)
                output, final_state = self.rnn(embeds, final_state)
                logits = self.linear(output) / temp

                new_tokens = Categorical(logits=logits[:, -1:]).sample()
                tokens = torch.cat([tokens, new_tokens], dim=1)

        generated = self.dataset.ids2text(tokens.squeeze())
        return generated
