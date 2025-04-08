import os
import torch
from .utils import load_pickle_file, save_dicts
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# Please split your data and prepare DataFrames: ROOT_PATH/train.csv, ROOT_PATH/test.csv, and ROOT_PATH/val.csv
ROOT_PATH = '../allData/gaze_data'
PATH_TO_SAVEDATA = '../project_folder_name/data'
FOLDER_NAME = 'project_folder'
FEATURES_NAME = 'features.csv'
EMB_NAME = 'composite_embeddings.pt'


def create_dict_dataset(df):
    """Takes gaze feature data and embeddings from a structured dataset and creates a dictionary 
    of DataFrames grouped by 'index_row'."""
    index_row = 0
    final_df = pd.DataFrame()
    for index, row in df.iterrows():
        if str(int(row['case'])) == '600':
            path = os.path.join(ROOT_PATH, row['gaze_path'][:-13], FOLDER_NAME)
        else:
            path = os.path.join(ROOT_PATH, str(int(row['case'])), row['gaze_path'][:-8], FOLDER_NAME)
        path_to_features = os.path.join(path, FEATURES_NAME)
        path_to_emb = os.path.join(path, EMB_NAME)
        if os.path.exists(path_to_features) and os.path.exists(path_to_emb):
            current_feature_df = pd.read_csv(path_to_features)
            if 'Unnamed: 0' in current_feature_df.columns:
                merged_df = current_feature_df.drop(['Unnamed: 0'], axis=1)
            current_emb = torch.load(path_to_emb)

            assert current_emb.shape[0] == current_feature_df.shape[0], f"Shape mismatch: {current_emb.shape} vs {current_feature_df.shape}"
            
            current_emb_df = pd.DataFrame(current_emb.numpy(), 
                                columns=[f'emb_{i}' for i in range(current_emb.shape[1])])
            merged_df = pd.concat([current_feature_df.reset_index(drop=True), current_emb_df], axis=1)
            merged_df['index_row'] = index_row
            final_df = pd.concat([final_df, merged_df], ignore_index=True)
            index_row+=1
        if os.path.exists(path_to_features) and not os.path.exists(path_to_emb):
            print(f"Path to features exists but not to embeddings: {path_to_features}")
        if not os.path.exists(path_to_features):
            print(f"Path to features does not exist: {path_to_features}")
    final_df = dict(list(final_df.groupby(['index_row'])))
    return final_df

def prepare_longitudinal_dataset():
    splitted_train_df = pd.read_csv(os.path.join(PATH_TO_SAVEDATA, "train.csv"))
    splitted_test_df = pd.read_csv(os.path.join(PATH_TO_SAVEDATA, "test.csv"))
    splitted_val_df = pd.read_csv(os.path.join(PATH_TO_SAVEDATA, "val.csv"))

    train = create_dict_dataset(splitted_train_df)
    test = create_dict_dataset(splitted_test_df)
    val = create_dict_dataset(splitted_val_df)
    save_dicts(train, os.path.join(PATH_TO_SAVEDATA, "grouped_elsevier25_train"))
    save_dicts(test, os.path.join(PATH_TO_SAVEDATA, "grouped_elsevier25_test"))
    save_dicts(val, os.path.join(PATH_TO_SAVEDATA, "grouped_elsevier25_val"))
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
        'win_w', 'win_h', 'rad_name',#'win_width', 'win_height', 'radiologist',
        'case', 'fname',
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
    if 'gaze_path' in  df.columns:
        df = df.drop(['gaze_path'], axis=1)
    if 'folder_num' in  df.columns:
        df = df.drop(['folder_num'], axis=1)
    return torch.tensor(df.values, dtype=torch.float32)

class ReadingDataset(Dataset):
    def __init__(self, grouped_data, sequence_length, data_type):
        self.sequences = []
        self.labels = []
        self.embeddings = [] if data_type == "emb" else None
        for i, (prefix, data) in enumerate(grouped_data.items()):
            data = data.dropna().reset_index(drop=True)
            if data.empty:
                print(f"Skipping empty dataset for prefix: {prefix}")
                continue
            y_main = torch.tensor(data['label'].values, dtype=torch.float32)
            emb_cols = [col for col in data.columns if col.startswith('emb_')]
            x_embed = torch.tensor(data[emb_cols].values, dtype=torch.float32)
            x_main = prepare_features(data.drop(columns=emb_cols))

            x_main = return_reading_seq(x_main, sequence_length)
            x_embed = return_reading_seq(x_embed, sequence_length) if data_type == "emb" else None

            self.sequences.extend(x_main.unsqueeze(0))
            self.labels.extend(y_main[0].unsqueeze(0))

            if data_type == "emb":
                self.embeddings.extend(x_embed.unsqueeze(0))

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
            - seq_length (int): Sequence length for data processing (default: 240).
            - type (str): Type of data to be processed, e.g., "tabular" (default: "tabular").
            - batch_size (int): Batch size for the DataLoader (default: 64).

    Returns:
        tuple: (train_loader, val_loader, test_loader) where each is a 
        DataLoader object.
    """
    sequence_length = kwargs.get("seq_length", 240)
    data_type = kwargs.get("type", "tabular")
    batch_size = kwargs.get("batch_size", 64)

    train_path = os.path.join(PATH_TO_SAVEDATA, "grouped_elsevier25_train.pkl")
    val_path = os.path.join(PATH_TO_SAVEDATA, "grouped_elsevier25_val.pkl")
    test_path = os.path.join(PATH_TO_SAVEDATA, "grouped_elsevier25_test.pkl")
    if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
        train = load_pickle_file(train_path)
        val = load_pickle_file(val_path)
        test = load_pickle_file(test_path)
    else:
        train, test, val = prepare_longitudinal_dataset()
    
    train_dataset = ReadingDataset(train, sequence_length, data_type)
    test_dataset = ReadingDataset(test, sequence_length, data_type)
    val_dataset = ReadingDataset(val, sequence_length, data_type)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=3)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=3)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=3)

    return train_loader, val_loader, test_loader
