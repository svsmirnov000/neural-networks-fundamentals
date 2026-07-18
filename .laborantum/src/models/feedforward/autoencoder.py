import torch

class Autoencoder(torch.nn.Module):
    def __init__(
            self,
            channels,
            activation=torch.nn.ReLU):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(channels[0], channels[1]),
            activation(),
            torch.nn.Linear(channels[1], channels[2])
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(channels[2], channels[1]),
            activation(),
            torch.nn.Linear(channels[1], channels[0])
        )
        if not hasattr(self, 'encoder'):
            self.encoder = torch.nn.Identity()
        if not hasattr(self, 'decoder'):
            self.decoder = torch.nn.Identity()

    def __forward_kernel(self, signal):
        input_shape = signal.shape
        res = self.decoder(
            self.encoder(
                signal.flatten(start_dim=1)
            )
        )
        res = res.reshape(input_shape)
        return res

    def forward(self, batch):
        if 'signals' not in batch:
            batch['signals'] = {}

        batch['signals']['reconstruction'] = self.__forward_kernel(
            batch['data']['image']
        )
        return batch