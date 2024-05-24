import torch
import pytorch_lightning as pl

import torch.nn as nn
import torch.nn.functional as F
import torchmetrics


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, self.expansion * out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, in_channels=3):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

def resnet18(num_classes=1000, in_channels=3):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, in_channels=in_channels)


def resnet34(num_classes=1000, in_channels=3):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, in_channels=in_channels)


def resnet50(num_classes=1000, in_channels=3):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, in_channels=in_channels)


def resnet101(num_classes=1000, in_channels=3):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, in_channels=in_channels)


def resnet152(num_classes=1000, in_channels=3):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, in_channels=in_channels)

def get_resnet_model(model_name, num_classes=1000,in_channels=3):
        if model_name == 'resnet18':
            return resnet18(num_classes, in_channels)
        elif model_name == 'resnet34':
            return resnet34(num_classes, in_channels)
        elif model_name == 'resnet50':
            return resnet50(num_classes, in_channels)
        elif model_name == 'resnet101':
            return resnet101(num_classes, in_channels)
        elif model_name == 'resnet152':
            return resnet152(num_classes, in_channels)
        else:
            raise ValueError(f"Invalid model name: {model_name}")


class ResNetLightning(pl.LightningModule):
    def __init__(self, num_classes=1000, in_channels=3, resnet_name='resnet18', use_adam=False, use_scheduler=False):
        super(ResNetLightning, self).__init__()
        self.model = get_resnet_model(resnet_name, num_classes, in_channels)
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.use_adam = use_adam
        self.use_scheduler = use_scheduler
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.model(x)
        loss = self.loss_fn(output, y)
        preds = torch.argmax(output, dim=1)
        acc = self.train_acc(preds, y)
        self.log('train_loss', loss)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.model(x)
        loss = self.loss_fn(output, y)
        preds = torch.argmax(output, dim=1)
        acc = self.val_acc(preds, y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
    
    def on_validation_epoch_end(self):
        self.log('val_acc_epoch', self.val_acc.compute())
    
    def configure_optimizers(self):
        if self.use_adam:
            optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        else:
            optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        if self.use_scheduler:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
            return [optimizer], [scheduler]
        return optimizer
    
    def configure_callbacks(self):
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            filename='{epoch}-{val_loss:.2f}',
            save_top_k=1,
            monitor='val_loss',
            mode='min'
        )
        return [checkpoint_callback]
