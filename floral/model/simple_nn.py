import torch.nn as nn


class SimpleNN(nn.Module):
    def __init__(self, dim, hidden_dim_mult, dim_out,
                 init_func=nn.Identity, init_norm=nn.Identity):
        super().__init__()
        hidden_dim = hidden_dim_mult * dim
        self.featurizer = nn.Sequential(nn.Linear(dim, hidden_dim), init_func(), init_norm(hidden_dim))
        self.fc = nn.Linear(hidden_dim, dim_out)
        self.hidden_dim = hidden_dim

    def forward(self, x, return_features=False):
        features = self.featurizer(x)
        return features if return_features else self.fc(features)


class FCNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes, activation=nn.ReLU):
        super().__init__()
        assert len(hidden_dims) > 0, "just use nn.Linear"
        layers = [
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dims[0]),
            activation()
        ]
        for dim_in, dim_out in zip(hidden_dims, hidden_dims[1:]):
            layers += [
                nn.Linear(dim_in, dim_out),
                activation()
            ]
        layers += [nn.Linear(hidden_dims[-1], num_classes)]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class MnistNN(FCNN):
    def __init__(self):
        super().__init__(input_dim=28*28, hidden_dims=[200], num_classes=10, activation=nn.ReLU)


