"""IDEC: autoencodeur pré-entraîné + KL sur affectations de Student.

Inspiré de Guo et al., IJCAI 2017 (doi:10.24963/ijcai.2017/243).
Adaptation temporelle et pragmatique ; pas une reproduction exacte des hyperparamètres
ou du protocole de l'article. Aucune étiquette n'entre dans la fonction de perte.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.cluster import KMeans
from .training import encode_all, set_seed


def student_soft_assign(z, centers, alpha=1.0):
    """q_ij selon la distribution t de Student, de somme 1 par observation."""
    distances = torch.sum((z[:, None, :] - centers[None, :, :]) ** 2, dim=2)
    q = (1 + distances / alpha).pow(-(alpha + 1) / 2)
    return q / q.sum(dim=1, keepdim=True).clamp_min(1e-12)


def sharpen_target(q):
    """p_ij = (q_ij² / f_j) / sum_j(q_ij² / f_j)."""
    frequency = q.sum(dim=0, keepdim=True).clamp_min(1e-12)
    weight = (q ** 2) / frequency
    return (weight / weight.sum(dim=1, keepdim=True).clamp_min(1e-12)).detach()


class IDEC(nn.Module):
    """Surcouche d'un autoencodeur fournissant encode/reconstruct.

    ADAPTER_PLUS_TARD : nb clusters, poids KL, et fréquence de recalcul
    des pseudo-étiquettes à sélectionner sur validation exclusivement.
    """
    def __init__(self, autoencoder, n_clusters=5, latent_dim=16):
        super().__init__()
        self.autoencoder = autoencoder
        self.centers = nn.Parameter(torch.zeros(n_clusters, latent_dim))

    def soft_assign(self, x):
        return student_soft_assign(self.autoencoder.encode(x), self.centers)

    def predict(self, x):
        return self.soft_assign(x).argmax(dim=1)


def fit_idec(autoencoder, x_train, n_clusters=5, epochs=10, batch_size=64,
             lr=1e-4, gamma=0.1, seed=42, device='cpu'):
    """Ajuste IDEC sur x_train préalablement réservé à l'entraînement.

    Le modèle AE DOIT avoir été pré-entraîné auparavant.
    Les pseudo-cibles sont recalculées une fois par époque sur train seulement.
    """
    if len(x_train) < n_clusters:
        raise ValueError('Le nombre de fenêtres train doit dépasser n_clusters.')
    if epochs < 1:
        raise ValueError('epochs doit être >= 1')
    set_seed(seed)
    autoencoder.to(device)
    z_train = encode_all(autoencoder, x_train, device=device)
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(z_train)
    idec = IDEC(autoencoder, n_clusters, latent_dim=z_train.shape[1]).to(device)
    with torch.no_grad():
        idec.centers.copy_(torch.as_tensor(km.cluster_centers_, dtype=torch.float32, device=device))
    optim = torch.optim.Adam(idec.parameters(), lr=lr)
    data = torch.as_tensor(x_train, dtype=torch.float32)
    loader = DataLoader(TensorDataset(data, torch.arange(len(data))),
                        batch_size=batch_size, shuffle=True)
    history = []
    for epoch in range(1, epochs + 1):
        idec.eval()
        with torch.no_grad():
            q_parts = []
            for start in range(0, len(data), batch_size):
                q_parts.append(idec.soft_assign(data[start:start+batch_size].to(device)).cpu())
            target = sharpen_target(torch.cat(q_parts, dim=0))
        idec.train()
        sum_loss = 0.
        for batch, indices in loader:
            batch = batch.to(device)
            p = target[indices].to(device)
            q = idec.soft_assign(batch)
            rec = idec.autoencoder.reconstruct(batch)
            loss_r = F.mse_loss(rec, batch)
            loss_c = F.kl_div(q.clamp_min(1e-12).log(), p, reduction='batchmean')
            loss = loss_r + gamma * loss_c
            if not torch.isfinite(loss):
                raise FloatingPointError('Perte IDEC non finie')
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            sum_loss += float(loss.item()) * len(batch)
        history.append({'epoch': epoch, 'loss': sum_loss / len(data)})
    return idec, history


@torch.no_grad()
def predict_idec(idec, x, batch_size=256, device='cpu'):
    idec.eval().to(device)
    predictions = []
    for start in range(0, len(x), batch_size):
        batch = torch.as_tensor(x[start:start+batch_size], device=device)
        predictions.append(idec.predict(batch).cpu().numpy())
    return np.concatenate(predictions)