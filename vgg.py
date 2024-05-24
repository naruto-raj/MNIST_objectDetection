import torch

import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
import numpy as np

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
class VGG(nn.Module):
    
    def __init__(self, vgg_name, in_channels, num_classes):
        super(VGG, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Sequential(
            nn.Linear(7*7*512,4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096,4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096,num_classes),
        )
        self._initialize_weight()
        
    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
    
    def _make_layers(self, cfg):
        layers = []
        in_channels = self.in_channels
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
    
    def _initialize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class VGG16Lightning(pl.LightningModule):
    def __init__(self, num_classes=10, in_channels=3, vgg_name='VGG16', use_adam=False, load_weights=False, weights_path=None, use_scheduler=False, lr_scheduler=None, lr_scheduler_params=None, **kwargs):
        super(VGG16Lightning, self).__init__()
        self.model = VGG(vgg_name, in_channels, num_classes)
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
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
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
