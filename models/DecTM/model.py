import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


class DecTM(nn.Module):
    '''
    Discovering Topics in Long-tailed Corpora with Causal Intervention. ACL 2021 findings.
    Xiaobao Wu, Chunping Li, Yishu Miao.
    '''
    def __init__(self, vocab_size, num_topics=50, en_units=200, dropout=0.4):
        super().__init__()

        self.num_topics = num_topics

        self.a = 1 * np.ones((1, num_topics)).astype(np.float32)
        self.mu2 = nn.Parameter(torch.as_tensor((np.log(self.a).T - np.mean(np.log(self.a), 1)).T))
        self.var2 = nn.Parameter(torch.as_tensor((((1.0 / self.a) * (1 - (2.0 / num_topics))).T + (1.0 / (num_topics * num_topics)) * np.sum(1.0 / self.a, 1)).T))

        self.mu2.requires_grad = False
        self.var2.requires_grad = False

        self.fc11 = nn.Linear(vocab_size, en_units)
        self.fc12 = nn.Linear(en_units, en_units)
        self.fc21 = nn.Linear(en_units, num_topics)
        self.fc22 = nn.Linear(en_units, num_topics)

        # align with the default parameters of tf.contrib.layers.batch_norm
        # https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/contrib/layers/batch_norm
        # center=True (add bias(beta)), scale=False (weight(gamma) is not used)
        self.mean_bn = nn.BatchNorm1d(num_topics, eps=0.001, momentum=0.001, affine=True)
        self.mean_bn.weight.data.copy_(torch.ones(num_topics))
        self.mean_bn.weight.requires_grad = False

        self.logvar_bn = nn.BatchNorm1d(num_topics, eps=0.001, momentum=0.001, affine=True)
        self.logvar_bn.weight.data.copy_(torch.ones(num_topics))
        self.logvar_bn.weight.requires_grad = False

        self.decoder_bn = nn.BatchNorm1d(vocab_size, eps=0.001, momentum=0.001, affine=True)
        self.decoder_bn.weight.data.copy_(torch.ones(vocab_size))
        self.decoder_bn.weight.requires_grad = False

        self.fc1_drop = nn.Dropout(dropout)
        self.theta_drop = nn.Dropout(dropout)

        self.beta = nn.Parameter(nn.init.xavier_uniform_(torch.empty(num_topics, vocab_size)))

    def get_beta(self):
        return self.beta

    def get_theta(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        theta = F.softmax(z, dim=1)
        theta = self.theta_drop(theta)
        if self.training:
            return theta, mu, logvar
        else:
            return theta

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + (eps * std)
        else:
            return mu

    def encode(self, x):
        e1 = F.softplus(self.fc11(x))
        e1 = F.softplus(self.fc12(e1))
        e1 = self.fc1_drop(e1)
        return self.mean_bn(self.fc21(e1)), self.logvar_bn(self.fc22(e1))

    def decode(self, theta):
        norm_theta = F.normalize(theta, dim=1)
        norm_beta = F.normalize(self.beta, dim=0)
        d1 = F.softmax(self.decoder_bn(torch.matmul(norm_theta, norm_beta)), dim=1)
        return d1

    def forward(self, x):
        theta, mu, logvar = self.get_theta(x)
        recon_x = self.decode(theta)
        loss = self.loss_function(x, recon_x, mu, logvar)
        return {'loss': loss}

    def loss_function(self, x, recon_x, mu, logvar):
        recon_loss = -(x * (recon_x + 1e-10).log()).sum(axis=1)
        var = logvar.exp()
        var_division = var / self.var2
        diff = mu - self.mu2
        diff_term = diff * diff / self.var2
        logvar_division = self.var2.log() - logvar
        KLD = 0.5 * ((var_division + diff_term + logvar_division).sum(axis=1) - self.num_topics)
        loss = (recon_loss + KLD).mean()
        return loss


class Model(DecTM):
    pass


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    # === HYPERPARAMS BEGIN ===
    num_epochs: int = 10, 
    batch_size: int = 64, 
    learning_rate: float = 0.001,
    en_units: int = 200,
    dropout: float = 0.4,
    # === HYPERPARAMS END ===
) -> tuple[Model, dict]:
    model = Model(
        vocab_size=X_train.shape[1],
        num_topics=len(set(y_train)),
        en_units=en_units,
        dropout=dropout
    )
    
    model.to("cpu")
    model.train()
    
    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    dataset = TensorDataset(X_train_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        for (batch, ) in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            batch = batch.to("cpu")
            optimizer.zero_grad()
            output = model(batch)
            loss = output['loss']
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader)}")

    hyperparams = {
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "en_units": en_units,
        "dropout": dropout,
        "num_topics": len(set(y_train)),
    }

    return model, hyperparams


def predict(model: Model, X_test: np.ndarray, batch_size: int = 64) -> np.ndarray:
    model.eval()
    model.to("cpu")
    
    if hasattr(X_test, "toarray"):
        X_test = X_test.toarray()
    
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    dataloader = DataLoader(X_test_tensor, batch_size=batch_size)
    
    all_theta = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            batch = batch.to("cpu")
            theta = model.get_theta(batch)
            if isinstance(theta, tuple):
                theta = theta[0]
            all_theta.append(theta.cpu())
            
    theta_matrix = torch.cat(all_theta, dim=0).numpy()
    return theta_matrix
