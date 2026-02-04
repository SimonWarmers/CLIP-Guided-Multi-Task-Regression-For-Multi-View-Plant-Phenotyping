"""
In this file the Dataset classes are defined for further use
"""
from torch.utils.data import Dataset
import pandas as pd
import torch
import re
import os


class LevelEncodingDataset(Dataset):
    def __init__(self, root_dir, csv_files, selected_files=[0, 0, 0, 0], max_age=100, max_leaf_count=200):
        if len(selected_files) != len(csv_files) and sum(selected_files) < 1:
            raise ValueError("selected_files must contain one entry per class and at least one selected file.")

        self.max_age = max_age
        self.max_leaf_count = max_leaf_count

        self.data = pd.DataFrame()
        for i, csv_file in enumerate(csv_files):
            if selected_files[i] == 1:
                df = pd.read_csv(os.path.join(root_dir, csv_file))
                df['class'] = i
                self.data = pd.concat([self.data, df], ignore_index=True)

        self.day_sequences = []

        consecutive_ids = []
        self.plant_ids = []
        plant_id_previous = -1
        day_previous = -1
        class_label_previous = -1
        level_previous = -1

        for i in self.data.iterrows():
            plant_id = int(i[1]['filename'].split('_')[1].replace('p', ''))
            day = int(i[1]['filename'].split('_')[2].replace('d', '').replace('D', ''))
            level = int(i[1]['filename'].split('_')[3].replace('L', ''))
            class_label = i[1]['class']

            if plant_id != plant_id_previous and plant_id_previous != -1:
                if len(self.plant_ids) == 0:
                    self.plant_ids.append([0, consecutive_ids[-1], plant_id_previous])
                else:
                    self.plant_ids.append([self.plant_ids[-1][1] + 1, consecutive_ids[-1] + 1, plant_id_previous])


            if plant_id != plant_id_previous or day != day_previous or class_label != class_label_previous or level != level_previous:
                if len(consecutive_ids) == 24:
                    self.day_sequences.append(consecutive_ids)
                else:
                    print(f"New sequence detected: plant_id: {plant_id_previous}, day: {day_previous}, class_label: {class_label_previous}, level: {level_previous}", f"with {len(consecutive_ids)} items")
                consecutive_ids = []

            consecutive_ids.append(i[0])

            plant_id_previous = plant_id
            day_previous = day
            class_label_previous = class_label
            level_previous = level

        self.plant_ids.append([self.plant_ids[-1][1] + 1, consecutive_ids[-1] + 1, plant_id_previous])

        self.text_features = pd.read_csv(os.path.join(root_dir, "text_features.csv")).to_numpy()
        self.leaf_count_range = 100
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    def __len__(self):
        return len(self.day_sequences)

    def __getitem__(self, idx):
        embedding_list = []
        filename_list = []
        text_embedding_list = []
        consecutive_ids = self.day_sequences[idx]

        for id in consecutive_ids:
            item = self.data.iloc[id]

            leaf_count = torch.tensor(int(item['leaf_count']), dtype=torch.float32)
            leaf_count = self.normaliz_leaf_count(leaf_count)

            age = torch.tensor(int(item['Age']), dtype=torch.float32)
            age = self.normaliz_age(age)
            class_label = int(item['class'])
            filename_list.append(item['filename'])
            clip_embedding = item['clip_embedding']

            clip_embedding_str = re.sub(r", device='[^']+'", "", clip_embedding)
            clip_embedding_tensor = eval(clip_embedding_str, {"tensor": torch.tensor, "float32": torch.float32})
            clip_embedding = clip_embedding_tensor
            if not isinstance(clip_embedding, torch.Tensor):
                clip_embedding = torch.tensor(clip_embedding_tensor, dtype=torch.float32).to(self.device)

            clip_embedding = clip_embedding.to(dtype=torch.float32).to(self.device)
            embedding_list.append(clip_embedding)

            text_embedding_list.append(self.get_text_embedding(int(item['leaf_count']), int(item['Age'])))

        paired_list = list(zip(filename_list, embedding_list))

        paired_list.sort(key=lambda pair: (
            int(pair[0].split('_')[3].replace('L', '').replace('D', '')),
            int(pair[0].split('_')[4].split('.')[0])
        ))

        filename_list, embedding_list = zip(*paired_list)
        filename_list = list(filename_list)
        embedding_list = list(embedding_list)

        return embedding_list, leaf_count, age, class_label, filename_list, text_embedding_list
    
    def normaliz_leaf_count(self, value):
        return value / self.max_leaf_count if self.max_leaf_count > 0 else 0

    def reverse_normaliz_leaf_count(self, value):
        return round(value * self.max_leaf_count) if self.max_leaf_count > 0 else 0

    def normaliz_age(self, value):
        return value / self.max_age if self.max_age > 0 else 0

    def reverse_normaliz_age(self, value):
        return round(value * self.max_age) if self.max_age > 0 else 0

    def get_text_embedding(self, leaf_count, age):
        age_step = age * self.leaf_count_range
        leaf_count_step = leaf_count
        return self.text_features[age_step + leaf_count_step]

    def get_text_embeddings_from_normalized(self, leaf_counts, ages):
        leaf_counts = (leaf_counts * self.max_leaf_count).astype(int)
        ages = (ages * self.max_age).astype(int)
        embeddings = []
        for leaf_count, age in zip(leaf_counts, ages):
            embeddings.append(self.get_text_embedding(leaf_count, age))
        return embeddings


###################################################################################
#  dataset for level detection
###################################################################################

class AnglesEncodingDataset(Dataset):
    def __init__(self, root_dir, csv_files, selected_files=[0, 0, 0, 0], max_age=100, max_leaf_count=200):
        if len(selected_files) != len(csv_files) and sum(selected_files) < 1:
            raise ValueError("selected_files must contain one entry per class and at least one selected file.")

        self.max_age = max_age
        self.max_leaf_count = max_leaf_count
        self.plant_ids = []
        plant_id_previous = -1
        self.data = pd.DataFrame()
        for i, csv_file in enumerate(csv_files):
            if selected_files[i] == 1:
                df = pd.read_csv(os.path.join(root_dir, csv_file))
                df['class'] = i
                self.data = pd.concat([self.data, df], ignore_index=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        for i in self.data.iterrows():
            plant_id = int(i[1]['filename'].split('_')[1].replace('p', ''))
            if plant_id != plant_id_previous and plant_id_previous != -1:
                self.plant_ids.append([0, i[0] - 1, plant_id_previous])
            plant_id_previous = plant_id
        self.plant_ids.append([self.plant_ids[-1][1] + 1, i[0], plant_id_previous])


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]

        class_label = int(item['class'])
        filename = item['filename']

        level = torch.tensor(int(filename.split('_')[3].replace('L', '')), dtype=torch.float32).to(self.device)
        angle = int(filename.split('_')[4].split('.')[0])
        angle = torch.tensor(angle, dtype=torch.float32).to(self.device)

        clip_embedding = item['clip_embedding']
        clip_embedding_str = re.sub(r", device='[^']+'", "", clip_embedding)
        clip_embedding_tensor = eval(clip_embedding_str, {"tensor": torch.tensor, "float32": torch.float32})
        clip_embedding = clip_embedding_tensor
        if not isinstance(clip_embedding, torch.Tensor):
            clip_embedding = torch.tensor(clip_embedding_tensor, dtype=torch.float32)

        clip_embedding = clip_embedding.to(self.device)

        return clip_embedding, level, angle, class_label, filename
    