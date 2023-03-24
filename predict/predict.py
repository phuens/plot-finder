import torch
import torchvision
import os
import pandas as pd

from train.training import Classification
from train.dataset import PrepareDataset

class Predict(Classification): 
    def __init__(self, config: object) -> None:
        super().__init__(config)

    def setup_testing(self): 
        get_data =  PrepareDataset(self.config)
        self.test_data = get_data.prepare_dataset(category='validation')

        model_path = os.path.join(self.config["test"]["model_path"], self.config["test"]["name"])
        self.model.load_state_dict(torch.load(model_path))

    def predict(self): 
        self.model.to(self.device)
        self.model.eval()

        predicted, target, image_name = [],[],[]

        with torch.no_grad(): 
            for (images, label, _, img_name) in self.test_data: 
                images = images.to(self.device)
                label = label.to(self.device)
                outputs = self.model(images)
                
                _, preds = torch.max(outputs, 1)

                predicted.append(preds.cpu().numpy())
                target.append(label.cpu().numpy())
                image_name.append(img_name)


        accuracy, f1, precision, recall = self.calculate_metrics(predicted=predicted, targets=target)
        
        print(f"Acc: {accuracy} f1:{f1}, val: {precision}, recall: {recall}")


        df = pd.DataFrame([image_name, predicted, target], columns=['name', 'predicted', 'target'])
        df.to_csv("predictions.csv", index=False)