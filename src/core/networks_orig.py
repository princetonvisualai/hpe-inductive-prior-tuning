import torch.nn as nn
import torch


class ParameterRegressor(nn.Module):
    def __init__(self, num_features, num_parts):
        super(ParameterRegressor, self).__init__()
        """
        convolutional encoder + linear layer at the end
        Args:
            num_features: list of ints containing number of features per layer
            num_parts: number of body parts for which we regress affine parameters
        Returns:
            torch.tensor (batch, num_parts, 2, 3), (2, 3) affine matrix for each body part
        """
        self.num_features = num_features
        self.num_parts = num_parts
        self.layers = self.define_network(num_features)

    def _add_conv_layer(self, in_ch, nf, dilation=1):
        return nn.Sequential(
            nn.Conv2d(in_ch, nf, 3, 1, 1, dilation=dilation),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(inplace=True)
        )

    def _add_down_layer(self, in_ch, nf):
        return nn.Sequential(
            nn.Conv2d(in_ch, nf, 3, 2, 1),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(inplace=True)
        )

    def define_network(self, num_features):
        layers = [self._add_conv_layer(in_ch=3+self.num_parts, nf=num_features[0])]
        for i in range(1, len(num_features)):
            if i == 1:
                layers.append(self._add_conv_layer(num_features[i-1], num_features[i-1], dilation=4))
            elif i == 2:
                layers.append(self._add_conv_layer(num_features[i-1], num_features[i-1], dilation=8))
            else:
                layers.append(self._add_conv_layer(num_features[i-1], num_features[i-1]))
            layers.append(self._add_down_layer(num_features[i-1], num_features[i]))
        
        n_f = 32

        param_branch = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_f * 8 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),
            # parameters for affine matrix
            nn.Linear(512, self.num_parts * 6)
        )

        layers.append(param_branch)

        return nn.Sequential(*layers)


    def forward(self, input, template):
        cat = torch.cat([input, template], dim=1)
        params = self.layers(cat)
        return params.view(-1, self.num_parts, 2, 3)

class Reconstructor(nn.Module):
    def __init__(self, num_features, num_parts):
        super(Reconstructor, self).__init__()
        """
        convolutional encoder, decoder
        """
        self.num_features = num_features
        self.num_parts = num_parts
        self.layers = self.define_network(self.num_features)

    def _add_conv_layer(self, in_ch, nf, dilation=1):
        return nn.Sequential(
            nn.Conv2d(in_ch, nf, 3, 1, 1, dilation=dilation),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(inplace=True)
        )

    def _add_down_layer(self, in_ch, nf):
        return nn.Sequential(
            nn.Conv2d(in_ch, nf, 3, 2, 1),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(inplace=True)
        )

    def _add_up_layer(self, in_ch, nf):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_ch, nf, 3, 1, 1),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(inplace=True)
        )

    def define_network(self, num_features):
        # encoder
        layers = [self._add_conv_layer(in_ch=3+self.num_parts, nf=num_features[0])]
        for i in range(1, len(num_features)):
            layers.append(self._add_conv_layer(num_features[i-1], num_features[i-1]))

            layers.append(self._add_down_layer(num_features[i - 1], num_features[i]))

        # decoder mirrors the encoder
        for i in range(len(num_features)-1, 0, -1):
            layers.append(self._add_conv_layer(num_features[i], num_features[i]))

            layers.append(self._add_up_layer(num_features[i], num_features[i-1]))

        layers.append(nn.Conv2d(num_features[i-1], 3, 3, 1, 1))
        return nn.Sequential(*layers)

    def forward(self, input, template):
        return self.layers(torch.cat([input, template], dim=1))