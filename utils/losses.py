from utils.parser import get_parser_with_args
from utils.metrics import FocalLoss, dice_loss
import torch

parser, metadata = get_parser_with_args()
opt = parser.parse_args()

def hybrid_loss(predictions, target, dice_weight=0.5):
    """Calculating the loss"""
    loss = 0
    pos = 0

    # gamma=0, alpha=None --> CE
    focal = FocalLoss()
    loss_weight = torch.tensor([0.5, 0.5, 0.75, 0.75, 1.0], dtype=torch.float)
    for prediction in predictions:

        bce = focal(prediction, target)
        dice = dice_loss(prediction, target)
        loss += (bce + dice * dice_weight) * loss_weight[pos]
        pos += 1

    return loss

