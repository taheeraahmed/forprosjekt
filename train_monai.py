import os
import shutil
import tempfile
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import torch
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import glob
import pandas as pd 

from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import decollate_batch, DataLoader
from monai.metrics import ROCAUCMetric
from monai.networks.nets import DenseNet121
from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    Compose,
    LoadImage,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
)
from monai.utils import set_determinism
from utils.set_up import set_up

# https://github.com/Project-MONAI/tutorials/blob/main/2d_classification/mednist_tutorial.ipynb'
def train():
    logger = set_up()

    data_path = '/cluster/home/taheeraa/datasets/chestxray-14'
    image_files_list = []
    for i in range(1, 13):
        folder_name = f'{data_path}/images_{i:03}'
        files_in_subfolder = glob.glob(f'{folder_name}/images/*')
        image_files_list.extend(files_in_subfolder)
    logger.info(f"Image paths: {len(image_files_list)}")

    df = pd.read_csv(f'{data_path}/Data_Entry_2017.csv')
    # Add the image paths to the dataframe
    df['Image Files List'] = image_files_list
    # Create a new dataframe to store the image paths, labels, and patient IDs
    df = df[['Image Files List', 'Finding Labels', 'Patient ID']]

    # Make a list of all the labels
    diseases = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema',
                'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening',
                'Pneumonia', 'Pneumothorax']

    # For each label, make a new column and
    # assign 1 if the disease is present and 0 if the disease is absent
    for disease in diseases:
        df[disease] = df['Finding Labels'].apply(lambda x: 1 if disease in x else 0)
        
    disease_mapping = {
        'Atelectasis': 0, 'Cardiomegaly': 1, 'Consolidation': 2, 'Edema': 3,
        'Effusion': 4, 'Emphysema': 5, 'Fibrosis': 6, 'Hernia': 7,
        'Infiltration': 8, 'Mass': 9, 'Nodule': 10, 'Pleural_Thickening': 11,
        'Pneumonia': 12, 'Pneumothorax': 13
    }

    # Function to get the first disease and map it
    def map_disease_to_number(label):
        first_disease = label.split('|')[0]
        return disease_mapping.get(first_disease, -1)  # Returns -1 if disease not found

    # Apply the function to create the new column
    df['image_class'] = df['Finding Labels'].apply(map_disease_to_number)


    image_class = df['image_class'].tolist()
    class_names = diseases
    image_width, image_height = PIL.Image.open(image_files_list[0]).size
    num_total = len(image_class)
    num_class = len(class_names)

    logger.info(f"Total image count: {num_total}")
    logger.info(f"Image dimensions: {image_width} x {image_height}")
    logger.info(f"Label names: {class_names}")


    plt.subplots(3, 3, figsize=(8, 8))
    for i, k in enumerate(np.random.randint(num_total, size=9)):
        im = PIL.Image.open(image_files_list[k])
        arr = np.array(im)
        plt.subplot(3, 3, i + 1)
        plt.xlabel(class_names[image_class[k]])
        plt.imshow(arr, cmap="gray", vmin=0, vmax=255)
    plt.tight_layout()
    plt.savefig('images/monai/random-chest-xray-and-label.jpg')

    val_frac = 0.1
    test_frac = 0.1
    length = len(image_files_list)
    indices = np.arange(length)
    np.random.shuffle(indices)

    test_split = int(test_frac * length)
    val_split = int(val_frac * length) + test_split
    test_indices = indices[:test_split]
    val_indices = indices[test_split:val_split]
    train_indices = indices[val_split:]

    train_x = [image_files_list[i] for i in train_indices]
    train_y = [image_class[i] for i in train_indices]
    val_x = [image_files_list[i] for i in val_indices]
    val_y = [image_class[i] for i in val_indices]
    test_x = [image_files_list[i] for i in test_indices]
    test_y = [image_class[i] for i in test_indices]

    logger.info(f"Training count: {len(train_x)}, Validation count: " f"{len(val_x)}, Test count: {len(test_x)}")

    train_transforms = Compose(
        [
            ScaleIntensity(),
            RandRotate(range_x=np.pi / 12, prob=0.5, keep_size=True),
            RandFlip(spatial_axis=0, prob=0.5),
            RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
        ]
    )

    val_transforms = Compose([ScaleIntensity()])

    y_pred_trans = Compose([Activations(softmax=True)])
    y_trans = Compose([AsDiscrete(to_onehot=num_class)])


    class ChestXRay14(torch.utils.data.Dataset):
        def __init__(self, image_files, labels, transforms):
            self.image_files = image_files
            self.labels = labels
            self.transforms = transforms

        def __len__(self):
            return len(self.image_files)

        def __getitem__(self, index):
            # Load the image from the file path
            image_path = self.image_files[index]
            image = Image.open(image_path).convert('L')  # Convert to grayscale
            image = np.array(image)

            # Check if the image has only two dimensions (H, W)
            if len(image.shape) == 2:
                # Add a channel dimension (C, H, W)
                image = np.expand_dims(image, axis=0)

            # Apply the transforms
            image = self.transforms(image)

            return image, self.labels[index]


    train_ds = ChestXRay14(train_x, train_y, train_transforms)
    train_loader = DataLoader(train_ds, batch_size=300, shuffle=True, num_workers=10)

    val_ds = ChestXRay14(val_x, val_y, val_transforms)
    val_loader = DataLoader(val_ds, batch_size=300, num_workers=10)

    test_ds = ChestXRay14(test_x, test_y, val_transforms)
    test_loader = DataLoader(test_ds, batch_size=300, num_workers=10)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenseNet121(spatial_dims=2, in_channels=1, out_channels=num_class).to(device)

    # Counting the number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Total number of parameters: {total_params}")
    logger.info(f"Total number of trainable parameters: {trainable_params}")

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 1e-5)
    max_epochs = 4
    val_interval = 1
    auc_metric = ROCAUCMetric()

    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []

    for epoch in range(max_epochs):
        logger.info("-" * 10)
        logger.info(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            logger.info(f"{step}/{len(train_ds) // train_loader.batch_size}, " f"train_loss: {loss.item():.4f}")
            epoch_len = len(train_ds) // train_loader.batch_size
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        logger.info(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                y_pred = torch.tensor([], dtype=torch.float32, device=device)
                y = torch.tensor([], dtype=torch.long, device=device)
                for val_data in val_loader:
                    val_images, val_labels = (
                        val_data[0].to(device),
                        val_data[1].to(device),
                    )
                    y_pred = torch.cat([y_pred, model(val_images)], dim=0)
                    y = torch.cat([y, val_labels], dim=0)
                y_onehot = [y_trans(i) for i in decollate_batch(y, detach=False)]
                y_pred_act = [y_pred_trans(i) for i in decollate_batch(y_pred)]
                auc_metric(y_pred_act, y_onehot)
                result = auc_metric.aggregate()
                auc_metric.reset()
                del y_pred_act, y_onehot
                metric_values.append(result)
                acc_value = torch.eq(y_pred.argmax(dim=1), y)
                acc_metric = acc_value.sum().item() / len(acc_value)
                if result > best_metric:
                    best_metric = result
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(root_dir, "best_metric_model.pth"))
                    logger.info("saved new best metric model")
                logger.info(
                    f"current epoch: {epoch + 1} current AUC: {result:.4f}"
                    f" current accuracy: {acc_metric:.4f}"
                    f" best AUC: {best_metric:.4f}"
                    f" at epoch: {best_metric_epoch}"
                )

    logger.info(f"Train completed, best_metric: {best_metric:.4f} " f"at epoch: {best_metric_epoch}")


    plt.figure("train", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Epoch Average Loss")
    x = [i + 1 for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.xlabel("epoch")
    plt.plot(x, y)
    plt.subplot(1, 2, 2)
    plt.title("Val AUC")
    x = [val_interval * (i + 1) for i in range(len(metric_values))]
    y = metric_values
    plt.xlabel("epoch")
    plt.plot(x, y)
    plt.savefig('images/monai/loss-metric.png')

    model.load_state_dict(torch.load(os.path.join(root_dir, "runs/best_metric_model.pth")))
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for test_data in test_loader:
            test_images, test_labels = (
                test_data[0].to(device),
                test_data[1].to(device),
            )
            pred = model(test_images).argmax(dim=1)
            for i in range(len(pred)):
                y_true.append(test_labels[i].item())
                y_pred.append(pred[i].item())

    logger.info(classification_report(y_true, y_pred, target_names=class_names, digits=4))

if __name__ == "__main__":
    train()
