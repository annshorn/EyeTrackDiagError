import torch
import numpy as np
import torch.nn as nn
from torch.nn import init
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAUROC


class BaseModel(pl.LightningModule):
    """
    A base model class.

    Attributes:
        learning_rate (float): The learning rate for optimizer.
        wd (float): Weight decay parameter for regularization.
        loss (torch.nn.modules.loss): Loss function initialized as BCELoss 
        for binary classification.
        auc (torchmetrics.BinaryAUROC): Metric for measuring the quality of 
        predictions via AUC-ROC score.
    """

    def __init__(self, learning_rate, weight_decay):
        super(BaseModel, self).__init__()
        self.learning_rate = learning_rate
        self.wd = weight_decay

        self.loss = nn.BCELoss()
        self.auc = BinaryAUROC()

        # Saving hyperparameters
        self.save_hyperparameters()

    def forward(self, *args, **kwargs):
        """Forward pass through the model. Needs to be defined with specific 
        model layers in a subclass."""
        return self.model(*args, **kwargs)

    def weight_pool(self, output):
        """Applies a weighted pooling to the output based on the output's 
        sequence length, enhancing the model's attention to specific features 
        over time."""
        weights = np.arange(0, output.size(1))
        weights = weights / np.sum(weights)
        weights = np.repeat(np.expand_dims(weights, axis=0), output.size(0), axis=0)
        weights = np.repeat(np.expand_dims(weights, axis=2), 2, axis=2)
        weights = torch.tensor(weights, device=output.device)
        results = torch.mul(output, weights)
        results = torch.sum(results, axis=1)
        return results

    def training_step(self, batch, batch_idx):
        """Training step."""
        if len(batch) == 2:
            x, y = batch
            y_hat = self(x)
        elif len(batch) == 3:
            x, emb, y = batch
            y_hat = self(x, emb)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True)

        auc_score = self.auc(y_hat, y.int())
        self.log('train_auc', auc_score, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        if len(batch) == 2:
            x, y = batch
            y_hat = self(x)
        elif len(batch) == 3:
            x, emb, y = batch
            y_hat = self(x, emb)
        val_loss = self.loss(y_hat, y)
        self.log('val_loss', val_loss)

        auc_score = self.auc(y_hat, y.int())
        self.log('val_auc', auc_score, on_step=False, on_epoch=True, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        """Sets up the optimizer with the learning rate and weight decay 
        parameters specified at initialization."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.wd)
        return optimizer


class TabularModelPL(BaseModel):
    """
    A model for tabular data processing that extends the BaseModel,
    incorporating a GRU layer followed by a linear layer and softmax activation
    for sequence processing and classification tasks. 

    Attributes:
        gru (nn.GRU): Gated Recurrent Unit layer.
        fc (nn.Linear): Fully connected layer.
        softmax (nn.Softmax): Softmax activation layer.

    Inputs:
        x (Tensor): A batch of input sequences, shaped 
        (batch_size, sequence_length, input_size), where each entry is a 
        vector representing a time step in the sequence.

    Returns:
        Tensor: Probability of the positive class.
    """

    def __init__(self, input_size, hidden_size, num_layers, output_size, learning_rate, weight_decay):
        super(TabularModelPL, self).__init__(learning_rate, weight_decay)
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=2)

        init.xavier_uniform_(self.fc.weight)
        init.constant_(self.fc.bias, 0)

    def forward(self, x):
        """Forward pass."""
        out, hn = self.gru(x)
        out = self.fc(out)
        out = self.softmax(out)
        out = self.weight_pool(out)
        return out[:, 1].to(torch.float32)


class EmbeddingModelPL(BaseModel):
    """
    Extends BaseModel to handle inputs with additional embedding features.

    Attributes:
        seq (nn.Sequential): Processes the embedding vector through two linear
        layers with a ReLU activation in between to produce a feature vector.
        gru (nn.GRU): Gated Recurrent Unit layer.
        fc (nn.Linear): Fully connected layer.
        softmax (nn.Softmax): Softmax activation layer.

    Inputs:
        x (Tensor): A batch of input sequences, shaped 
        (batch_size, sequence_length, input_size), where each entry is a 
        vector representing a time step in the sequence.
        emb (Tensor): The embedding tensor, shaped 
        (batch_size, sequence_length, input_size).
    
    Returns:
        Tensor: Probability of the positive class.
    """

    def __init__(self, input_size, hidden_size, num_layers, output_size, learning_rate, weight_decay):
        super(EmbeddingModelPL, self).__init__(learning_rate, weight_decay)
        self.seq = nn.Sequential(
            nn.Linear(288, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x, emb):
        """Forward pass."""
        feat = self.seq(emb)
        out = torch.cat((x, feat), dim=2)
        out, hn = self.gru(out)
        out = self.fc(out)
        out = self.softmax(out)
        out = self.weight_pool(out)
        return out[:, 1].to(torch.float32)
