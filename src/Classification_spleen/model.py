import timm
import torch.nn as nn
from config import CFG


class TimmModelType2(nn.Module):
    def __init__(self, backbone, pretrained=False):
        super(TimmModelType2, self).__init__()

        self.encoder = timm.create_model(
            backbone,
            in_chans=CFG.in_chans,
            num_classes=CFG.out_dim,
            features_only=False,
            drop_rate=CFG.drop_rate,
            drop_path_rate=CFG.drop_path_rate,
            pretrained=pretrained
        )

        if 'efficient' in backbone:
            hdim = self.encoder.conv_head.out_channels
            self.encoder.classifier = nn.Identity()
        elif 'convnext' in backbone:
            hdim = self.encoder.head.fc.in_features
            self.encoder.head.fc = nn.Identity()
        elif 'resnet' in backbone:
            hdim = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()

        self.lstm = nn.LSTM(hdim, 256, num_layers=2, dropout=CFG.drop_rate, bidirectional=True, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(CFG.drop_rate_last),
            nn.LeakyReLU(0.1),
            nn.Linear(256, CFG.out_dim),
        )

    def forward(self, x):  # (bs, nc, ch, sz, sz)
        bs = x.shape[0]
        x = x.view(bs * CFG.n_slice_per_c, CFG.in_chans, CFG.image_size, CFG.image_size)
        feat = self.encoder(x)
        feat = feat.view(bs, CFG.n_slice_per_c, -1)
        feat, _ = self.lstm(feat)
        feat = feat.contiguous().view(bs * CFG.n_slice_per_c, -1)
        feat = self.head(feat)
        feat = feat.view(bs, CFG.n_slice_per_c, CFG.out_dim).contiguous()
        return feat
