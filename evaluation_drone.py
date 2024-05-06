import torch
import os
import numpy as np
import argparse
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_score, average_precision_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

from models.MobileNetV3 import get_model, MobileNetV3
from datasets.fsd50k import get_eval_set
from models.preprocess import AugmentMelSTFT
from helpers.init import worker_init_fn
from torch.utils.data import DataLoader

import numpy as np
import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import argparse
from sklearn import metrics
import torch.nn.functional as F

from datasets.fsd50k import get_eval_set, get_valid_set, get_training_set
from models.MobileNetV3 import get_model as get_mobilenet
from models.preprocess import AugmentMelSTFT
from helpers.init import worker_init_fn
from helpers.utils import NAME_TO_WIDTH, exp_warmup_linear_down, mixup


def visualize_confusion_matrix(true_labels, predicted_labels, class_names):
    """
    Visualizes the confusion matrix using a heatmap.

    Args:
        true_labels (list[int]): The true labels.
        predicted_labels (list[int]): The predicted labels.
        class_names (list[str]): List of class names to label the axes.
    """
    # Generate confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    # Plot using seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)

    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()

def evaluate(args):
    # Load pre-trained model
    model = get_model(
        width_mult=NAME_TO_WIDTH(args.model_name),
        pretrained_name=args.model_name,
        num_classes=args.num_classes
    )

    # Load model weights
    model_path = args.model_file
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    else:
        raise FileNotFoundError(f"Model file '{model_path}' not found.")
  
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.eval()

    # Create the preprocessing function
    mel = AugmentMelSTFT(
        n_mels=args.n_mels, sr=args.resample_rate,
        win_length=args.window_size, hopsize=args.hop_size,
        n_fft=args.n_fft, freqm=args.freqm,
        timem=args.timem, fmin=args.fmin,
        fmax=args.fmax, fmin_aug_range=args.fmin_aug_range,
        fmax_aug_range=args.fmax_aug_range
    )
    mel.to(device)

    # Evaluation DataLoader
    eval_dl = DataLoader(get_eval_set(resample_rate=args.resample_rate, variable_eval=args.variable_eval_length),
                         worker_init_fn=worker_init_fn, num_workers=args.num_workers,
                         batch_size=1 if args.variable_eval_length else args.batch_size)

    # Collect predictions and true labels
    targets = []
    outputs = []
    for batch in tqdm(eval_dl):
        x, _, y = batch
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            x = _mel_forward(x, mel)
            y_hat, _ = model(x)
        targets.append(y.cpu().numpy())
        outputs.append(y_hat.float().cpu().numpy())

    # Combine and convert to binary labels
    targets = np.concatenate(targets)
    outputs = np.concatenate(outputs)
    outputs_binary = (outputs > 0.7).astype(int)

    # Extract true and predicted labels for visualization
    true_labels = targets.argmax(axis=1).tolist()
    predicted_labels = outputs_binary.argmax(axis=1).tolist()
    print(predicted_labels)
    # List of class names
    class_names = ["Airplane", "Background", "Drone", "Explosion", "Helicopter"]

    print(f"Targets:  {targets}")
    print(f"Outputs:  {outputs}")
    print(f"Binary_outputs:  {outputs_binary}")
    # Confusion Matrix
    conf_matrix = metrics.confusion_matrix(targets.argmax(axis=1), outputs_binary.argmax(axis=1))
    print(f"confusion: {conf_matrix}")
    # Class-wise Precision
    precision_scores = metrics.precision_score(targets, outputs_binary, average=None)
    print(f"precision_scores: {precision_scores}")
    # Visualize confusion matrix
    visualize_confusion_matrix(true_labels, predicted_labels, class_names)

    # Additional evaluation metrics
    mAP = average_precision_score(targets, outputs, average=None)
    ROC = roc_auc_score(targets, outputs, average=None)

    print(f"Results on dataset for model '{args.model_name}':")
    print(f"  mAP: {mAP.mean():.3f}")
    print(f"  ROC: {ROC.mean():.3f}")

def _mel_forward(x, mel):
    old_shape = x.size()
    x = x.reshape(-1, old_shape[2])
    x = mel(x)
    x = x.reshape(old_shape[0], old_shape[1], x.shape[1], x.shape[2])
    return x

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Evaluation Script')

    # General arguments
    parser.add_argument('--model_file', type=str, required=True, help="Path to the model file (.pt)")
    parser.add_argument('--model_name', type=str, required=True, help="Name of the model to load")
    parser.add_argument('--num_classes', type=int, default=5, help="Number of classes")
    parser.add_argument('--cuda', action='store_true', help="Use CUDA if available")

    # Preprocessing
    parser.add_argument('--resample_rate', type=int, default=32000, help="Sampling rate for audio")
    parser.add_argument('--window_size', type=int, default=800, help="Window size for STFT")
    parser.add_argument('--hop_size', type=int, default=320, help="Hop size for STFT")
    parser.add_argument('--n_fft', type=int, default=1024, help="Number of FFT points")
    parser.add_argument('--n_mels', type=int, default=128, help="Number of Mel bands")
    parser.add_argument('--freqm', type=int, default=0, help="Frequency mask")
    parser.add_argument('--timem', type=int, default=0, help="Time mask")
    parser.add_argument('--fmin', type=int, default=0, help="Minimum frequency for Mel")
    parser.add_argument('--fmax', type=int, default=None, help="Maximum frequency for Mel")
    parser.add_argument('--fmin_aug_range', type=int, default=10, help="Augmentation range for minimum frequency")
    parser.add_argument('--fmax_aug_range', type=int, default=2000, help="Augmentation range for maximum frequency")

    # Evaluation specific
    parser.add_argument('--variable_eval_length', action='store_true', help="Variable length for evaluation dataset")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for evaluation")
    parser.add_argument('--num_workers', type=int, default=12, help="Number of workers for dataloading")

    args = parser.parse_args()
    evaluate(args)
