import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion=1, downsample=None):
        super(Block, self).__init__()
        self.expansion = expansion
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels*self.expansion)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module): 
    def __init__(self, image_channels, block, num_layers, num_classes=4):
        super(ResNet, self).__init__()
        if num_layers == 18:
            layers = [2, 2, 2, 2]
        elif num_layers == 34:
            layers = [3, 4, 6, 3]
        else:
            layers = None
            
        self.expansion = 1
        self.in_channels = 64

        self.conv1 = nn.Conv2d(image_channels, self.in_channels, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*self.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride=1):
        downsample = None
        if stride != 1:
            # layer2 to layer4
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels*self.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels*self.expansion)
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, self.expansion, downsample))
        self.in_channels = out_channels * self.expansion

        for i in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels, expansion = self.expansion))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


# if __name__ == '__main__':
#     x = torch.rand([1, 3, 224, 224])
#     model = ResNet(image_channels=3, block=Block, num_layers=18, num_classes=4)
#     print(model)

#     # Total parameters and trainable parameters.
#     total_params = sum(p.numel() for p in model.parameters())
#     print(f"{total_params:,} total parameters.")
#     total_trainable_params = sum(
#         p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"{total_trainable_params:,} training parameters.")

#     output = model(x).to('cuda')
#     print(output.shape)
#     print(output)
    


        