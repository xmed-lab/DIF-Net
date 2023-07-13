import torch
import torch.nn as nn
import torch.nn.functional as F



class SurfaceClassifier(nn.Module):
    def __init__(self, filter_channels, no_residual=True, last_op=None):
        super().__init__()

        self.filters = []
        self.no_residual = no_residual
        self.last_op = last_op

        if self.no_residual:
            for l in range(len(filter_channels) - 1):
                self.filters.append(nn.Conv1d(
                    filter_channels[l],
                    filter_channels[l + 1],
                    1))
                self.add_module("conv%d" % l, self.filters[l])
        else:
            for l in range(len(filter_channels) - 1):
                if 0 != l:
                    self.filters.append(
                        nn.Conv1d(
                            filter_channels[l] + filter_channels[0],
                            filter_channels[l + 1],
                            1))
                else:
                    self.filters.append(nn.Conv1d(
                        filter_channels[l],
                        filter_channels[l + 1],
                        1))

                self.add_module("conv%d" % l, self.filters[l])

    def forward(self, feature):
        '''
        :param feature: [B, C_in, N]
        :return: [B, C_out, N]
        '''
        y = feature
        tmpy = feature
        for i in range(len(self.filters)):
            if self.no_residual:
                y = self._modules['conv' + str(i)](y)
            else:
                y = self._modules['conv' + str(i)](
                    y if i == 0
                    else torch.cat([y, tmpy], 1)
                )
            
            if i != len(self.filters) - 1:
                y = F.leaky_relu(y)

        if self.last_op:
            y = self.last_op(y)

        return y
