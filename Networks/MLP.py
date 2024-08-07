import torch
import torch.nn as nn
import pickle
import lzma

class MLP(nn.Module):
        def __init__(self, input_size: int, hidden_size: int, output_size: int, softmax: bool, nonneg_out: bool):
            super(MLP, self).__init__()
            self.input_size = input_size
            self.hidden_size  = hidden_size
            self.output_size = output_size
            self.softmax = softmax
            self.nonneg_out = nonneg_out
            self.network = [
                nn.LayerNorm(self.input_size),
                nn.Tanh(),
                nn.Linear(self.input_size, self.hidden_size),
                nn.Tanh(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.Tanh(),
                nn.Linear(self.hidden_size, self.output_size)
            ]

            self.network = nn.Sequential(*self.network)

            self.mean = 0.
            self.std = 1.

        def set_normalization(self, mean: float = 0, std: float = 1):
            self.mean = mean
            self.std = std

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = (self.network(x) * self.std) + self.mean
            # not used but maybe useful to have positive zeta_net outputs?
            if self.nonneg_out:
              return y #.exp()
            else:
              return y

        def save(self, filename: str) -> bool:
            try:
                with lzma.open(filename, 'wb') as filehandler:
                    pickle.dump({
                        'state_dict': self.state_dict(),
                        'mean': self.mean,
                        'std': self.std}, filehandler, protocol=pickle.HIGHEST_PROTOCOL)
                return True
            except:
                print(f'Could not create file {filename}')
                return False

        def load(self, filename: str) -> bool:
            try:
                with lzma.open(filename, 'rb') as filehandler:
                    data = pickle.load(filehandler)
                    self.mean = data['mean']
                    self.set_normalization(data['mean'], data['std'])
                    self.load_state_dict(data['state_dict'])
                return True
            except:
                print(f'Could not find file {filename}')
                return False