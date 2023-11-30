import matplotlib.pyplot as plt
import numpy as np

def plot_metrics(train_arr, val_arr, output_folder, logger, type='None'):
    plt.figure(figsize=(10, 5))
    plt.plot(train_arr, label=f'Training {type}')
    plt.plot(val_arr, label=f'Validation {type}')
    plt.title(f'Training and Validation {type} Per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel(f'{type}')
    plt.legend()
    plt.savefig(f'{output_folder}/plot_train_val_{type}.png')
    logger.info(f'Saved images to: {output_folder}/plot_train_val_{type}.png')

def denormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)  # Multiply by std and then add the mean
    return tensor

def plot_pred(inputs, labels, preds, output_folder, logger):
    num_images = len(inputs)
    cols = int(np.sqrt(num_images))
    rows = cols if cols**2 == num_images else cols + 1

    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)  # Adjust the space between images

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for i, ax in enumerate(axes.flatten()):
        if i < num_images:
            input = inputs[i]
            denormalized_input = denormalize(input.clone(), mean, std)
            img = denormalized_input.numpy().transpose((1, 2, 0))
            plt.imshow(img, cmap='gray')
            actual_label = 'Positive' if labels[i].item() == 1 else 'Negative'
            predicted_label = 'Positive' if preds[i].item() == 1 else 'Negative'
            ax.set_title(f'Actual: {actual_label}\nPredicted: {predicted_label}', fontsize=10, backgroundcolor='white')
            ax.axis('off')  # Hide the axis
        else:
            ax.axis('off')  # Hide axis if no image

    plt.tight_layout()
    plt.savefig(f'{output_folder}/img_chest_pred.png')
    logger.info(f'Saved images to: {output_folder}/img_chest_pred.png')
    logger.info('Done training')