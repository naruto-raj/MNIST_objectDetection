import torch
import pytorch_lightning as pl
import torchvision

import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms
import torchmetrics

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_chanels, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_chanels, **kwargs)
        self.bn = nn.BatchNorm2d(out_chanels)
        
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class InceptionBlock(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_1x1,
        red_3x3,
        out_3x3,
        red_5x5,
        out_5x5,
        out_pool,
    ):
        super(InceptionBlock, self).__init__()
        self.branch1 = ConvBlock(in_channels, out_1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, red_3x3, kernel_size=1, padding=0),
            ConvBlock(red_3x3, out_3x3, kernel_size=3, padding=1),
        )
        self.branch3 = nn.Sequential(
            ConvBlock(in_channels, red_5x5, kernel_size=1),
            ConvBlock(red_5x5, out_5x5, kernel_size=5, padding=2),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, padding=1, stride=1),
            ConvBlock(in_channels, out_pool, kernel_size=1),
        )
    
    def forward(self, x):
        branches = (self.branch1, self.branch2, self.branch3, self.branch4)
        return torch.cat([branch(x) for branch in branches], 1)

class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.dropout = nn.Dropout(p=0.7)
        self.pool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = ConvBlock(in_channels, 128, kernel_size=1)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class InceptionV1(nn.Module):
    def __init__(self, aux_logits=True, num_classes=1000, in_channels=3,bbox=False):
        super(InceptionV1, self).__init__()
        self.aux_logits = aux_logits
        self.bbox=bbox
        self.conv1 = ConvBlock(
            in_channels=in_channels, 
            out_chanels=64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
        )
        self.conv2 = ConvBlock(64, 192, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)
        self.inception4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128)
        self.inception5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(1024, num_classes)
        
        if self.aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)
        else:
            self.aux1 = self.aux2 = None
        if self.bbox:
          self.fcbbox = nn.Linear(1024, 4) # minx, miny, maxx, maxy
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool(x)
        x = self.inception4a(x)
        
        if self.aux_logits and self.training:
            aux1 = self.aux1(x)
        
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        
        if self.aux_logits and self.training:
            aux2 = self.aux2(x)
        
        x = self.inception4e(x)
        x = self.maxpool(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        if self.bbox:
          class_pred = self.fc(x)
          bbox_pred = self.fcbbox(x)
        else:
          x = self.fc(x)
        
        if self.aux_logits and self.training:
            return aux1, aux2, x
        if self.bbox:
          return class_pred, bbox_pred
        return x


class InceptionV1Lightning(pl.LightningModule):
    def __init__(self, aux_logits=True, num_classes=10, in_channels=1, use_adam=False, use_scheduler=False):
        super(InceptionV1Lightning, self).__init__()
        # Initialize the InceptionV1 model specifically for MNIST
        self.model = InceptionV1(aux_logits=aux_logits, num_classes=num_classes, in_channels=in_channels)
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.use_adam = use_adam
        self.use_scheduler = use_scheduler

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        if self.model.aux_logits:
            aux1, aux2, output = self.model(x)
            loss1 = self.loss_fn(aux1, y)
            loss2 = self.loss_fn(aux2, y)
            loss = loss1*0.3 + loss2*0.3+ self.loss_fn(output, y)
        else:
            output = self.model(x)
            loss = self.loss_fn(output, y)

        # Compute accuracy
        preds = torch.argmax(output, dim=1)
        acc = self.train_acc(preds, y)
        self.log('train_loss', loss)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        # When validating, we often just care about the final output, not the auxiliary outputs
        output = self.model(x)
        loss = self.loss_fn(output, y)

        # Compute accuracy
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
