from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torch
from data_loading import ImageDataset
from two_heads_model import TwoHeadModel
import os
import kagglehub


# download data
data_dir = kagglehub.dataset_download("andrewmvd/dog-and-cat-detection")
print("Path to dataset files:", data_dir)

# data transformations
annotation_dir = os.path.join(data_dir, 'annotations')
image_dir = os.path.join(data_dir, 'images')

data_transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# train test split
dataset = ImageDataset(image_dir, annotation_dir, data_transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
generator = torch.Generator().manual_seed(42)
train_set, val_set = random_split(dataset=dataset, lengths=[train_size, val_size], generator=generator)

# data loader
train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=2)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=2)

# create model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TwoHeadModel()
model.to(device)

# loss and optimizer
classifier_criterion = torch.nn.CrossEntropyLoss()
regressor_criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# train loop
def train_model(model):
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        current_items = 0
        total_class_loss = 0
        total_box_loss = 0
        for images, labels, bboxes in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            bboxes = bboxes.to(device)

            # forward pass
            logits, pred_coord = model(images)
            class_loss = classifier_criterion(logits, labels)
            box_loss = regressor_criterion(bboxes, pred_coord)
            loss = box_loss + class_loss  # combine losses

            # update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print loss
            current_items += labels.size(0)
            total_class_loss += class_loss.item() * labels.size(0)
            total_box_loss += box_loss.item() * labels.size(0)
            print(f'Epoch {epoch+1}/{num_epochs}, item {current_items}/{len(train_set)}: class_loss: {class_loss}, box_loss: {box_loss}')
        print(f'avg_class_loss: {total_class_loss/len(train_set)}, avg_box_loss: {total_box_loss/len(train_set)}')

        # evaluate 
        print('Evaluating...')
        model.eval()
        with torch.no_grad():
            total_class_loss = 0.0
            total_box_loss = 0.0
            correct = 0
            for images, labels, bboxes in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                bboxes = bboxes.to(device)

                # forward
                logits, pred_coord = model(images)
                total_class_loss += classifier_criterion(logits, labels).item() * labels.size(0)
                total_box_loss += regressor_criterion(bboxes, pred_coord).item() * labels.size(0)
                _, class_pred = torch.max(logits, 1)
                correct += (class_pred == labels).sum()

        print(f'Epoch {epoch+1}/{num_epochs}: avg_class_loss: {total_class_loss/len(val_set)}, avg_box_loss: {total_box_loss/len(val_set)}, accuracy: {100*correct/len(val_set):.2f}')

    # save model
    torch.save(model.state_dict(), '2_head_model.ckpt')

if __name__ == '__main__':
    train_model(model)