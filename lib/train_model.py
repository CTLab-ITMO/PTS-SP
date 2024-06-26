import argparse
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn

from .models.fcnn_models import InceptModel, CLIPModel
from .models.custom_datasets import CustomDataset
from .utils import (
    evaluate,
    get_incept_image_embedding,
    get_clip_image_embedding,
    split_and_process_data,
)


def main(
    model_name,
    max_combinations_train,
    max_combinations_test,
    split_ratio,
    num_epochs,
    lr,
):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if model_name == "inception":
        model = InceptModel().to(device)
        get_image_embedding = get_incept_image_embedding
    elif model_name == "clip":
        model = CLIPModel().to(device)
        get_image_embedding = get_clip_image_embedding
    else:
        raise ValueError("Invalid model name. Choose either 'inception' or 'clip'.")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Resize images if needed
            transforms.ToTensor(),  # Convert PIL images to tensors
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    data_dict_train, data_dict_test, data_dict_values_train, data_dict_values_test = (
        split_and_process_data(split_ratio)
    )

    train_dataset = CustomDataset(
        data_dict=data_dict_train,
        data_dict_values=data_dict_values_train,
        transform=transform,
        max_combinations=max_combinations_train,
        split="train",
    )
    test_dataset = CustomDataset(
        data_dict=data_dict_test,
        data_dict_values=data_dict_values_test,
        transform=transform,
        max_combinations=max_combinations_test,
        split="test",
    )

    train_loader = DataLoader(train_dataset, batch_size=1)
    test_loader = DataLoader(test_dataset, batch_size=1)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels, targets, _, _ in tqdm(
            train_loader
        ):  # Ignoring empirical mAP and sizes here
            inputs = inputs.to(device)
            labels = labels.to(device)
            targets = targets.to(device)  # Parameters targets

            optimizer.zero_grad()

            outputs = model(
                torch.cat(
                    (get_image_embedding(inputs[0]), get_image_embedding(labels[0])),
                    dim=1,
                )
            )

            loss = criterion(outputs, targets.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Evaluation
        avg_smape_params = evaluate(model, test_loader, device)
        torch.save(model.state_dict(), f"model_{model_name}_v3_fc.pt")
        print(f"Epoch {epoch+1}, Training Loss: {running_loss/len(train_loader)}")
        for i, smape_value in enumerate(avg_smape_params):
            print(f"Validation SMAPE for parameter {i+1}: {smape_value}%")

    print("Training finished!")
    avg_smape_params = evaluate(model, test_loader, device)
    for i, smape_value in enumerate(avg_smape_params, start=1):
        print(f"Test SMAPE for parameter {i}: {smape_value}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a model with either Inception or CLIP"
    )
    parser.add_argument(
        "model",
        type=str,
        choices=["clip", "inception"],
        help="The model type to use (clip or inception)",
    )
    parser.add_argument(
        "--max_combinations_train",
        type=int,
        default=2000,
        help="Maximum combinations for training data",
    )
    parser.add_argument(
        "--max_combinations_test",
        type=int,
        default=500,
        help="Maximum combinations for test data",
    )
    parser.add_argument(
        "--split_ratio", type=float, default=0.8, help="Train/test split ratio"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate for the optimizer"
    )

    args = parser.parse_args()

    main(
        args.model,
        args.max_combinations_train,
        args.max_combinations_test,
        args.split_ratio,
        args.num_epochs,
        args.lr,
    )
