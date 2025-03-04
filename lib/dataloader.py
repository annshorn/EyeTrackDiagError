import os
import torch
from .utils import load_pickle_file, save_dicts
from torch.utils.data import Dataset, DataLoader
import pandas as pd

def create_dict_dataset(df, data_folder, feature_folder='gru', feature_name_csv='gaze_features.csv'):
    """Reads gaze feature CSV files from a structured dataset and creates a dictionary 
    of DataFrames grouped by 'index_row'.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing metadata.
        data_folder (str): Base path to the dataset.
        feature_folder (str, optional): Subfolder where features are stored. Default: 'gru'.
        feature_name_csv (str, optional): Feature filename. Default: 'gaze_features.csv'.
    
    Returns:
        dict: Dictionary of DataFrames grouped by 'index_row'.
    """
    index_row = 0
    final_df = pd.DataFrame()
    for index, row in df.iterrows():
        path_to_features = os.path.join(data_folder, str(int(row['case'])), 
                                        row['gaze_path'], feature_folder, feature_name_csv)
        if os.path.exists(path_to_features):
            current_feature_df = pd.read_csv(path_to_features)
            if 'Unnamed: 0' in current_feature_df.columns:
                current_feature_df = current_feature_df.drop(['Unnamed: 0'], axis=1)
            
            current_feature_df['radiologist'] = row.get('radiologist', None)

            current_feature_df['label'] = row['label']
            current_feature_df['index_row'] = index_row
            index_row+=1
            final_df = pd.concat([final_df, current_feature_df], ignore_index=True)
    final_df = dict(list(final_df.groupby(['index_row'])))
    return final_df

def prepare_longitudinal_dataset(data_folder):
    """Prepares and processes longitudinal datasets.

    Args:
        data_folder (str): Path to the folder containing the dataset CSV files.

    Returns:
        tuple: A tuple containing three dictionaries:
            - train (dict): Processed training dataset.
            - test (dict): Processed test dataset.
            - val (dict): Processed validation dataset.
    """
    splitted_train_df = pd.read_csv(os.path.join(data_folder, "train.csv"))
    splitted_test_df = pd.read_csv(os.path.join(data_folder, "test.csv"))
    splitted_val_df = pd.read_csv(os.path.join(data_folder, "val.csv"))

    train = create_dict_dataset(splitted_train_df, data_folder)
    test = create_dict_dataset(splitted_test_df, data_folder)
    val = create_dict_dataset(splitted_val_df, data_folder)

    save_dicts(train, os.path.join(data_folder, "grouped_train"))
    save_dicts(test, os.path.join(data_folder, "grouped_test"))
    save_dicts(val, os.path.join(data_folder, "grouped_val"))

    return train, test, val

def return_reading_seq(seq, sequence_length):
    """Adjusts the input tensor sequence to have a fixed length.

    If the sequence is shorter than the specified length,
    it pads the sequence with a constant value (-1).
    If the sequence is longer, it truncates the beginning of the sequence,
    keeping only the last `sequence_length` elements.
    """
    if len(seq) < sequence_length:
        pad_length = sequence_length - len(seq)
        seq = torch.cat((seq, torch.full((pad_length, seq.shape[1]), -1, dtype=torch.float32)))
    elif len(seq) > sequence_length:
        seq = seq[-sequence_length:]
    return seq

def prepare_features(df):
    """Prepares and cleans the features from a pandas DataFrame for model input."""
    columns_to_drop = [
        'win_w', 'win_h', 'radiologist',
        'case', 'fname', 'gaze_path',
        'label'
    ]
    df = df.drop(columns_to_drop, axis=1)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(['Unnamed: 0'], axis=1)
    if 'folder_num' in df.columns:
        df = df.drop(['folder_num'], axis=1)
    if {'elapsed_time', 'start_time'}.issubset(df.columns):
        df = df.drop(['elapsed_time', 'start_time'], axis=1)
    if 'timestamp' in df.columns:
        df = df.drop(['timestamp'], axis=1)
    if {'speed_mean', 'speed_std'}.issubset(df.columns):
        df = df.drop(['speed_mean', 'speed_std'], axis=1)
    if {'acceleration_mean', 'acceleration_std'}.issubset(df.columns):
        df = df.drop(['acceleration_mean', 'acceleration_std'], axis=1)
    if {'x', 'y'}.issubset(df.columns):
        df = df.drop(['x', 'y'], axis=1)
    if {'x0', 'y0'}.issubset(df.columns):
        df = df.drop(['x0', 'y0'], axis=1)
    if 'rad_name' in  df.columns:
        df = df.drop(['rad_name'], axis=1)
    if 'doctor_confidence' in  df.columns:
        df = df.drop(['doctor_confidence'], axis=1)
    if 'index_row' in  df.columns:
        df = df.drop(['index_row'], axis=1)
    return torch.tensor(df.values, dtype=torch.float32)

def load_image_embeddings(data_folder, path):
    path_to_emb = os.path.join(data_folder, path, "gru", 'images_encoder.pt')
    emb = torch.load(path_to_emb, map_location=torch.device('cpu'))
    return emb

class ReadingDataset(Dataset):
    def __init__(self, grouped_data, sequence_length, data_type, data_folder):
        """
        A dataset that handles both standard features and additional embeddings.

        Args:
            grouped_data (dict): Dictionary containing data groups.
            sequence_length (int): Sequence length for time-series processing.
            embeddings (list, optional): List of embeddings corresponding to each group.
        """
        self.sequences = []
        self.labels = []
        self.embeddings = [] if data_type == "emb" else None

        for i, (prefix, data) in enumerate(grouped_data.items()):
            path = os.path.join(data_folder, str(int(data['case'].iloc[0])), data['gaze_path'].iloc[0])

            data = data.dropna().reset_index(drop=True)
            if data.empty:
                print(f"Skipping empty dataset for prefix: {prefix} ; Path: {path}")
                continue
            y_main = torch.tensor(data['label'].values, dtype=torch.float32)
            x_main = prepare_features(data)

            if data_type == "emb":
                current_emb = load_image_embeddings(data_folder, path)
                try:
                    seq = torch.cat((x_main, current_emb), dim=1)
                except RuntimeError:
                    print(len(current_emb), len(x_main), path)
            else:
                seq = x_main  # No embeddings, use only prepared features

            # Process sequences
            seq = return_reading_seq(seq, sequence_length)

            # If embeddings exist, split them after sequence processing
            if data_type == "emb":
                x_main_seq, current_emb_seq = torch.split(seq, [x_main.shape[1], current_emb.shape[1]], dim=1)
                self.embeddings.extend(current_emb_seq.unsqueeze(0))
                self.sequences.extend(x_main_seq.unsqueeze(0))
            else:
                self.sequences.extend(seq.unsqueeze(0))
            self.labels.extend(y_main[0].unsqueeze(0))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.embeddings is not None:
            return self.sequences[idx], self.embeddings[idx], self.labels[idx]
        return self.sequences[idx], self.labels[idx]


def get_train_val_test(**kwargs):
    """Loads or prepares train, validation, and test datasets and 
    returns corresponding DataLoader objects.

    Parameters:
        kwargs (dict): Optional keyword arguments.
            - data_folder (str): Path to the folder containing the data.
            - seq_length (int): Sequence length for data processing (default: 240).
            - type (str): Type of data to be processed, e.g., "tabular" (default: "tabular").
            - batch_size (int): Batch size for the DataLoader (default: 64).

    Returns:
        tuple: (train_loader, val_loader, test_loader) where each is a 
        DataLoader object.
    """
    data_folder = kwargs.get("data_folder", "")
    sequence_length = kwargs.get("seq_length", 240)
    data_type = kwargs.get("type", "tabular")
    batch_size = kwargs.get("batch_size", 64)

    train_path = os.path.join(data_folder, "grouped_train.pkl")
    val_path = os.path.join(data_folder, "grouped_val.pkl")
    test_path = os.path.join(data_folder, "grouped_test.pkl")
    if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
        train = load_pickle_file(train_path)
        val = load_pickle_file(val_path)
        test = load_pickle_file(test_path)
    else:
        train, test, val = prepare_longitudinal_dataset(data_folder)
    
    train_dataset = ReadingDataset(train, sequence_length, data_type, data_folder)
    test_dataset = ReadingDataset(test, sequence_length, data_type, data_folder)
    val_dataset = ReadingDataset(val, sequence_length, data_type, data_folder)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=3)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=3)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=3)

    return train_loader, val_loader, test_loader
    
    

    



