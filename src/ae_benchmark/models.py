"""Architectures proposées comme candidats, non comme vainqueurs du benchmark.

Toutes acceptent [batch, 2, fenêtre] et fournissent encode() + reconstruct().
ADAPTER_PLUS_TARD: dimensions, capacité, pertes et contraintes réelles selon dataset.
"""
import torch
from torch import nn
from torch.nn import functional as F


class BaseAE(nn.Module):
    def loss(self, x, noise_std=0.):
        if noise_std > 0:
            noisy = x + noise_std * torch.randn_like(x)
        else:
            noisy = x
        recon = self.reconstruct(noisy)
        return F.mse_loss(recon, x)


class MLPAE(BaseAE):
    def __init__(self, n_channels=2, length=256, latent_dim=16):
        super().__init__()
        self.n_channels, self.length = n_channels, length
        dim = n_channels*length
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(dim,256),nn.GELU(),
                                     nn.Linear(256,64), nn.GELU(),nn.Linear(64,latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim,64),nn.GELU(),
                                     nn.Linear(64,256),nn.GELU(),nn.Linear(256,dim))
    def encode(self,x): return self.encoder(x)
    def decode(self,z): return self.decoder(z).reshape(-1,self.n_channels,self.length)
    def reconstruct(self,x): return self.decode(self.encode(x))
    def forward(self,x): return self.reconstruct(x)


class TemporalEncoder(nn.Module):
    def __init__(self, n_channels=2, latent_dim=16):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(n_channels,32,7,stride=2,padding=3),nn.GELU(),
                                  nn.Conv1d(32,64,5,stride=2,padding=2),nn.GELU(),
                                  nn.Conv1d(64,96,3,stride=2,padding=1),nn.GELU(),
                                  nn.AdaptiveAvgPool1d(8))
        self.head = nn.Linear(96*8,latent_dim)
    def forward(self,x): return self.head(self.net(x).flatten(1))


class TemporalDecoder(nn.Module):
    def __init__(self, n_channels=2, length=256, latent_dim=16):
        super().__init__()
        self.length = length
        self.project = nn.Linear(latent_dim,96*8)
        self.net = nn.Sequential(nn.Conv1d(96,64,5,padding=2),nn.GELU(),
                                 nn.Conv1d(64,32,5,padding=2),nn.GELU(),
                                 nn.Conv1d(32,n_channels,5,padding=2))
    def forward(self,z):
        x = self.project(z).reshape(-1,96,8)
        x = F.interpolate(x, size=self.length,mode="linear",align_corners=False)
        return self.net(x)


class CNNAE(BaseAE):
    def __init__(self,n_channels=2,length=256,latent_dim=16):
        super().__init__()
        self.encoder=TemporalEncoder(n_channels,latent_dim)
        self.decoder=TemporalDecoder(n_channels,length,latent_dim)
    def encode(self,x): return self.encoder(x)
    def decode(self,z): return self.decoder(z)
    def reconstruct(self,x): return self.decode(self.encode(x))
    def forward(self,x): return self.reconstruct(x)


class DenoisingCNNAE(CNNAE):
    """Bruit uniquement pendant entraînement; score de test sur entrée propre."""
    def loss(self,x,noise_std=.15):
        return super().loss(x,noise_std=noise_std)


class ConvVAE(BaseAE):
    def __init__(self,n_channels=2,length=256,latent_dim=16,kl_weight=0.001):
        super().__init__()
        self.encoder=TemporalEncoder(n_channels,latent_dim*2)
        self.decoder=TemporalDecoder(n_channels,length,latent_dim)
        self.kl_weight=kl_weight
    def distribution(self,x): return self.encoder(x).chunk(2,dim=1)
    def encode(self,x): return self.distribution(x)[0]  # moyenne déterministe
    def decode(self,z): return self.decoder(z)
    def reconstruct(self,x): return self.decode(self.encode(x))
    def forward(self,x): return self.reconstruct(x)
    def loss(self,x,noise_std=0.):
        mu, logvar = self.distribution(x)
        logvar=logvar.clamp(-12,8)
        z=mu+torch.randn_like(mu)*torch.exp(.5*logvar)
        mse=F.mse_loss(self.decode(z),x)
        kl=-.5*torch.mean(torch.sum(1+logvar-mu.square()-torch.exp(logvar),dim=1))
        return mse+self.kl_weight*kl


class MultiScaleAttentionBlock(nn.Module):
    """Convolutions 3, 7, 15 + attention canal (squeeze/excitation)."""
    def __init__(self,n_channels=2):
        super().__init__()
        self.branches=nn.ModuleList([nn.Conv1d(n_channels,16,k,padding=k//2) for k in (3,7,15)])
        self.att=nn.Sequential(nn.AdaptiveAvgPool1d(1),nn.Conv1d(48,12,1),nn.GELU(),
                               nn.Conv1d(12,48,1),nn.Sigmoid())
    def forward(self,x):
        h=torch.cat([F.gelu(layer(x)) for layer in self.branches],dim=1)
        return h*self.att(h)


class MultiScaleAE(BaseAE):
    def __init__(self,n_channels=2,length=256,latent_dim=16):
        super().__init__()
        self.front=MultiScaleAttentionBlock(n_channels)
        self.enc=TemporalEncoder(48,latent_dim)
        self.decoder=TemporalDecoder(n_channels,length,latent_dim)
    def encode(self,x): return self.enc(self.front(x))
    def decode(self,z): return self.decoder(z)
    def reconstruct(self,x): return self.decode(self.encode(x))
    def forward(self,x): return self.reconstruct(x)


class MemoryAE(BaseAE):
    """Prototype de mémoire latente apprenable; la littérature a d'autres variantes.

    ADAPTER_PLUS_TARD: taille mémoire, température et régularisation de parcimonie
    à optimiser sur validation, surtout en cas de reconstruction de défauts.
    """
    def __init__(self,n_channels=2,length=256,latent_dim=16,memory_size=32,temperature=.15):
        super().__init__()
        self.encoder=TemporalEncoder(n_channels,latent_dim)
        self.decoder=TemporalDecoder(n_channels,length,latent_dim)
        self.memory=nn.Parameter(torch.randn(memory_size,latent_dim)*.1)
        self.temperature=temperature
    def encode(self,x): return self.encoder(x)
    def read_memory(self,z):
        logits=F.normalize(z,dim=1)@F.normalize(self.memory,dim=1).T
        attention=(logits/self.temperature).softmax(dim=1)
        return attention @ self.memory
    def decode(self,z): return self.decoder(self.read_memory(z))
    def reconstruct(self,x): return self.decode(self.encode(x))
    def forward(self,x): return self.reconstruct(x)


from .advanced_models import (SparseMemAE,CascadedMemAE,AdaptiveDenoisingAE,MaskedSwinAE,AdversarialAE)


MODEL_REGISTRY={
    "mlp":MLPAE, "cnn":CNNAE, "denoise":DenoisingCNNAE,
    "vae":ConvVAE,"multiscale":MultiScaleAE,"memory":MemoryAE,
    "memae":SparseMemAE,"cmae":CascadedMemAE,
    "adae":AdaptiveDenoisingAE,"masked_swin":MaskedSwinAE,
    "aae":AdversarialAE,
}


def build_model(name,length,channels=2,latent_dim=16):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Modèle inconnu : {name}. Choix : {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](n_channels=channels,length=length,latent_dim=latent_dim)