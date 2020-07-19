import torch.utils.data
import torch.autograd as autograd

from utils.parser import get_parser_with_args
from utils.helpers import (get_test_loaders, get_criterion,
                           initialize_metrics, get_mean_metrics,
                           set_test_metrics)
from sklearn.metrics import precision_recall_fscore_support as prfs
from tqdm import tqdm


parser, metadata = get_parser_with_args()
opt = parser.parse_args()

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

test_loader = get_test_loaders(opt)

path = 'tmp/checkpoint_epoch_64.pt'
model = torch.load(path)
criterion = get_criterion(opt)

model.eval()

val_metrics = initialize_metrics()
with torch.no_grad():
    tbar = tqdm(test_loader)
    for batch_img1, batch_img2, labels in tbar:
        # Set variables for training
        batch_img1 = autograd.Variable(batch_img1).float().to(dev)
        batch_img2 = autograd.Variable(batch_img2).float().to(dev)
        labels = autograd.Variable(labels).long().to(dev)

        # Get predictions and calculate loss
        cd_preds = model(batch_img1, batch_img2)

        cd_preds = cd_preds[-1]
        _, cd_preds = torch.max(cd_preds, 1)

        # Calculate and log other batch metrics
        cd_corrects = (100 *
                       (cd_preds.squeeze().byte() == labels.squeeze().byte()).sum() /
                       (labels.size()[0] * (opt.patch_size**2)))

        cd_val_report = prfs(labels.data.cpu().numpy().flatten(),
                             cd_preds.data.cpu().numpy().flatten(),
                             average='binary',
                             pos_label=1)

        test_metrics = set_test_metrics(val_metrics,
                                  cd_corrects,
                                  cd_val_report)

        # log the batch mean metrics
        mean_test_metrics = get_mean_metrics(test_metrics)

        # clear batch variables from memory
        del batch_img1, batch_img2, labels

    print("EPOCH VALIDATION METRICS", mean_test_metrics)