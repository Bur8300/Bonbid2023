import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from model.unet import UNet
from utils.model_utils import test_arg_parser, set_seed
from utils.data_utils import MadisonStomach
from utils.viz_utils import visualize_predictions, plot_train_val_history, plot_metric
from utils.metric_utils import compute_dice_score

def test_model(model, args, save_path, test_loader):
    '''
    Tests the model on the test dataset and computes the average Dice score.
    
    Args:
    - model (torch.nn.Module): The PyTorch model to test.
    - args (argparse.Namespace): Parsed arguments for device, batch size, etc.
    - save_path (str): Directory where results (e.g., metrics plot) will be saved.
    
    Functionality:
    - Sets the model to evaluation mode and iterates over the test dataset.
    - Computes the Dice score for each batch and calculates the average.
    - Saves a plot of the Dice coefficient history.
    '''
    os.makedirs(os.path.join(save_path, args.exp_id, 'test'), exist_ok=True)
    model.eval()
    dice = 0
    dice_coef_history = []
    for batch, (images, masks) in enumerate(tqdm(test_loader, desc= f"Test", leave = False)):
        images, masks = images.to(args.device), masks.to(args.device)
        preds = model(images)
        if (batch % 25) == 0:
            visualize_predictions(images, masks, preds, os.path.join(save_path, args.exp_id) + "/test", batch_idx= batch, show = False, epoch= -1)
        dice += compute_dice_score(masks, (torch.nn.functional.sigmoid(preds) > 0.5).float())
        dice_coef_history.append(dice / (len(dice_coef_history) + 1))
    plot_metric(dice_coef_history, ylabel="dice coeff", plot_dir=save_path, args=args, metric='Test_dice_coeff', xlabel= "Batch Index")
    print(f"Test_Dice: {dice / len(test_loader)}") 

if __name__ == '__main__':

    args = test_arg_parser()
    save_path = "results"
    set_seed(42)

    #Define dataset
    dataset = MadisonStomach(data_path="madison-stomach", 
                            mode="test")

    test_dataloader = DataLoader(dataset, batch_size=args.bs)

    # Define and load your model
    model = UNet(1,1)
    model = torch.load(args.model_path, map_location=args.device)

    test_model(model=model,
                args=args,
                save_path=save_path,
                test_loader = test_dataloader)