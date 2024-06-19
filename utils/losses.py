import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.module import normalize_image

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, predict_parameters, target_parameters, predict_PET, predict_MRI, sample, target_key):
        pet_out = normalize_image(sample['PET_out'])
        # print(predict_PET.shape, pet_out.shape)
        mse = self.mse_loss(normalize_image(predict_PET).cuda(), pet_out.cuda())

        l1 = self.l1_loss(predict_parameters, target_parameters)

        if target_key == 'angles':
            weights = 5
        if target_key == 'Tdash':
            weights = 10   
        if target_key == 'reorientation':
            return (mse + l1 * 2) * 0.1, l1
        # print(mse)
        return mse + l1 * weights, l1

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, true):
        num_classes = logits.size(1)
        true = F.one_hot(true, num_classes).permute(0, 4, 1, 2, 3).float()
        probs = F.softmax(logits, dim=1)
        # print(probs.shape)
        intersection = torch.sum(probs * true, dim=(0, 2, 3, 4))
        union = torch.sum(probs, dim=(0, 2, 3, 4)) + torch.sum(true, dim=(0, 2, 3, 4))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class SegmentationLoss(nn.Module):
    def __init__(self):
        super(SegmentationLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()

    def forward(self, logits, targets):
        # print(logits.shape, targets.shape)
        ce = self.ce_loss(logits, targets.squeeze(1).long())
        dice = self.dice_loss(logits, targets.squeeze(1).long())
        return 0.5*ce + dice, dice

