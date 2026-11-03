"""Architectures de recherche inspirées d'articles, adaptées aux vibrations 1D.

Ces classes ne sont PAS des réimplémentations officielles des auteurs.
MemAE (Gong 2019) : mémoire parcimonieuse et pénalité d'entropie.
CMAE (Lu 2025) : approximation à deux échelles/mémoires, SANS suivi d'ordres.
MaskedSwinAE (Luo & Liu 2025) : patchs masqués, attention locale/décalée 1D.
ADAE (Lu et al. 2024) : unités de rétrécissement adaptatif et connexions résiduelles.

ADAPTER_PLUS_TARD: les capacités, fonctions de perte, fenêtrage, échelles et
protocoles de validation doivent être calibrés sur les acquisitions réelles.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .models import BaseAE, TemporalEncoder, TemporalDecoder


class SparseMemory(nn.Module):
    """Lecture différentiable d'une mémoire, avec sélection parcimonieuse top-k.

    Adaptation pratique du hard-shrinkage de MemAE, PAS la formule exacte.
    Le top-k garantit au moins un item actif et évite la division par zéro.
    """
    def __init__(self, latent_dim=16, n_items=32, topk=4, temperature=0.15):
        super().__init__()
        if not (1 <= topk <= n_items):
            raise ValueError('Exiger 1 <= topk <= n_items')
        if temperature <= 0:
            raise ValueError('temperature doit être positive')
        self.memory = nn.Parameter(torch.randn(n_items, latent_dim) * 0.1)
        self.topk = topk
        self.temperature = temperature

    def forward(self, z):
        logits = F.normalize(z, dim=1, eps=1e-8) @ F.normalize(self.memory, dim=1, eps=1e-8).T
        logits = logits / self.temperature
        top_values, top_index = logits.topk(self.topk, dim=1)
        weights = logits.new_zeros(logits.shape).scatter(1, top_index, F.softmax(top_values, dim=1))
        read = weights @ self.memory
        entropy = -(weights * weights.clamp_min(1e-12).log()).sum(dim=1).mean()
        return read, entropy, weights


class SparseMemAE(BaseAE):
    """MemAE temporel adapté : reconstruction normale + entropie de la mémoire."""
    def __init__(self, n_channels=2, length=256, latent_dim=16,
                 memory_size=32, topk=4, entropy_weight=0.0002):
        super().__init__()
        self.encoder = TemporalEncoder(n_channels, latent_dim)
        self.decoder = TemporalDecoder(n_channels, length, latent_dim)
        self.memory = SparseMemory(latent_dim, memory_size, topk)
        self.entropy_weight = entropy_weight

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        memory_z, _, _ = self.memory(z)
        return self.decoder(memory_z)

    def reconstruct(self, x):
        return self.decode(self.encode(x))

    def forward(self, x):
        return self.reconstruct(x)

    def loss(self, x, noise_std=0.):
        z = self.encoder(x)
        mem_z, ent, _ = self.memory(z)
        return F.mse_loss(self.decoder(mem_z), x) + self.entropy_weight * ent


class CascadedMemAE(BaseAE):
    """Adaptation *proposée* de CMAE à 2 échelles sans suivi d'ordres.

    Le contexte grossier guide la reconstruction fine : deux mémoires,
    deux encodeurs et deux décodeurs. La partie 'order tracking' de l'article
    IEEE 2025 est distincte et non exécutée sans vitesse instantanée.
    """
    def __init__(self, n_channels=2, length=256, latent_dim=16,
                 memory_size=32, topk=4, entropy_weight=0.0002):
        super().__init__()
        if length < 32:
            raise ValueError('CMAE exige une fenêtre de 32 points minimum')
        coarse_length = math.ceil(length / 2)
        self.length = length
        self.encoder_coarse = TemporalEncoder(n_channels, latent_dim)
        self.encoder_fine = TemporalEncoder(n_channels, latent_dim)
        self.memory_coarse = SparseMemory(latent_dim, memory_size, topk)
        self.memory_fine = SparseMemory(latent_dim, memory_size, topk)
        self.decoder_coarse = TemporalDecoder(n_channels, coarse_length, latent_dim)
        self.decoder_fine = TemporalDecoder(n_channels, length, latent_dim)
        self.context_to_fine = nn.Linear(2 * latent_dim, latent_dim)
        self.fusion = nn.Sequential(nn.Conv1d(2*n_channels, 16, 5, padding=2), nn.GELU(),
                                    nn.Conv1d(16, n_channels, 3, padding=1))
        self.entropy_weight = entropy_weight

    def encode(self, x):
        coarse = F.avg_pool1d(x, kernel_size=2, stride=2, ceil_mode=True)
        return self.context_to_fine(torch.cat((self.encoder_fine(x),
                                                self.encoder_coarse(coarse)), dim=1))

    def _read_and_decode(self, x):
        coarse = F.avg_pool1d(x, 2, 2, ceil_mode=True)
        coarse_lat = self.encoder_coarse(coarse)
        coarse_mem, entropy_coarse, _ = self.memory_coarse(coarse_lat)
        fine_lat = self.encoder_fine(x)
        context = self.context_to_fine(torch.cat((fine_lat, coarse_mem), dim=1))
        fine_mem, entropy_fine, _ = self.memory_fine(context)
        reconstructed_coarse = self.decoder_coarse(coarse_mem)
        up_coarse = F.interpolate(reconstructed_coarse, size=x.shape[-1], mode='linear', align_corners=False)
        reconstructed_fine = self.decoder_fine(fine_mem)
        reconstruction = self.fusion(torch.cat((up_coarse, reconstructed_fine), dim=1))
        return reconstruction, reconstructed_coarse, coarse, entropy_coarse + entropy_fine

    def reconstruct(self, x):
        return self._read_and_decode(x)[0]

    def forward(self, x):
        return self.reconstruct(x)

    def loss(self, x, noise_std=0.):
        out, coarse_out, coarse, ent = self._read_and_decode(x)
        return (F.mse_loss(out, x) + 0.25 * F.mse_loss(coarse_out, coarse)
                + self.entropy_weight * ent)


class AdaptiveShrinkageUnit(nn.Module):
    """Atténuation adaptative par canal, inspirée des ASU de l'ADAE 2024."""
    def __init__(self, channels):
        super().__init__()
        hidden = max(4, channels // 4)
        self.gate = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Conv1d(channels, hidden, 1),
                                  nn.GELU(), nn.Conv1d(hidden, channels, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.gate(x.abs())


class AdaptiveDenoisingAE(BaseAE):
    """ADAE 1D simplifié : shrinkage encoder/décodeur + skip local + dropout.

    La stratégie de bruit gaussien est expérimentale, non un élément constant
    des mesures ; elle est soumise à ablation bruit connu/inconnu.
    """
    def __init__(self, n_channels=2, length=256, latent_dim=16, dropout=0.10):
        super().__init__()
        self.length = length
        self.channels = n_channels
        self.extract = nn.Conv1d(n_channels, 32, 7, padding=3)
        self.encoder_path = nn.Sequential(AdaptiveShrinkageUnit(32), nn.Dropout1d(dropout),
                        nn.Conv1d(32,64,5,stride=2,padding=2), nn.GELU(),
                        AdaptiveShrinkageUnit(64), nn.Conv1d(64,64,3,padding=1),
                        nn.GELU(), nn.AdaptiveAvgPool1d(8))
        self.to_latent = nn.Linear(64*8, latent_dim)
        self.from_latent = nn.Linear(latent_dim,64*8)
        self.decoder_path = nn.Sequential(AdaptiveShrinkageUnit(64),
                   nn.Conv1d(64,32,5,padding=2), nn.GELU(),
                   AdaptiveShrinkageUnit(32), nn.Conv1d(32,n_channels,3,padding=1))

    def encode(self, x):
        h = F.gelu(self.extract(x))
        return self.to_latent(self.encoder_path(h).flatten(1))

    def decode(self, z):
        h = self.from_latent(z).reshape(-1,64,8)
        return self.decoder_path(F.interpolate(h, size=self.length, mode='linear', align_corners=False))

    def reconstruct(self, x):
        return self.decode(self.encode(x))

    def forward(self, x):
        return self.reconstruct(x)

    def loss(self, x, noise_std=0.12):
        return F.smooth_l1_loss(self.reconstruct(x + noise_std*torch.randn_like(x)), x)


class WindowAttentionBlock(nn.Module):
    """Attention 1D sur fenêtres locales, décalées une couche sur deux.

    Masque les interactions artificielles créées par le bouclage de torch.roll.
    Adaptation Swin sans patch-merging hiérarchique de l'article original.
    """
    def __init__(self, width=48, heads=4, window_size=4, shifted=False):
        super().__init__()
        if width % heads:
            raise ValueError('width doit être divisible par heads')
        self.window_size = window_size
        self.shift = window_size//2 if shifted else 0
        self.ln1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 2*width), nn.GELU(), nn.Linear(2*width,width))

    def forward(self, z):
        batch, tokens, dim = z.shape
        if tokens % self.window_size:
            raise ValueError('Nombre de patchs incompatible avec la fenêtre attention')
        y = self.ln1(z)
        if self.shift:
            y = torch.roll(y, shifts=-self.shift, dims=1)
        windows = y.reshape(batch, tokens//self.window_size, self.window_size, dim)
        windows = windows.reshape(-1,self.window_size,dim)
        attn_mask = None
        if self.shift:
            # Chaque fenêtre décalée contient des indices physiques : supprimer
            # uniquement les liens qui feraient le tour de la série.
            orig = (torch.arange(tokens, device=z.device) + self.shift) % tokens
            orig = orig.reshape(-1,self.window_size)
            boundary = (orig[:,:,None]-orig[:,None,:]).abs() > self.window_size
            if boundary.any():
                attn_mask = torch.zeros((tokens//self.window_size,self.window_size,self.window_size),
                                         device=z.device,dtype=z.dtype)
                attn_mask.masked_fill_(boundary, -1e4)
                attn_mask = attn_mask.repeat(batch,1,1)
                attn_mask = attn_mask.repeat_interleave(self.attn.num_heads,dim=0)
        attended,_ = self.attn(windows,windows,windows,attn_mask=attn_mask,need_weights=False)
        attended = attended.reshape(batch,tokens,dim)
        if self.shift:
            attended = torch.roll(attended,shifts=self.shift,dims=1)
        z = z + attended
        return z + self.ff(self.ln2(z))


class MaskedSwinAE(BaseAE):
    """Masked autoencoder temporel à attention locale/décalée.

    À l'entraînement on masque des *patchs* entiers et la perte donne plus
    de poids aux patchs masqués ; en évaluation, l'entrée est complète.
    """
    def __init__(self,n_channels=2,length=256,latent_dim=16,patch_size=8,
                 width=48,heads=4,window_size=4,mask_ratio=0.35):
        super().__init__()
        if length % patch_size or length // patch_size < window_size:
            raise ValueError('longueur doit être divisible par patch_size; suffisamment de patchs')
        if (length//patch_size) % window_size:
            raise ValueError('nombre de patchs doit être multiple de window_size')
        if not (0 < mask_ratio < 1):
            raise ValueError('mask_ratio doit être dans (0,1)')
        self.n_channels,self.length = n_channels,length
        self.patch_size,self.n_patches=patch_size,length//patch_size
        self.mask_ratio=mask_ratio
        self.input_proj = nn.Linear(n_channels*patch_size,width)
        self.pos_embed = nn.Parameter(torch.zeros(1,self.n_patches,width))
        self.mask_token = nn.Parameter(torch.randn(1,1,width)*0.02)
        self.blocks = nn.ModuleList([WindowAttentionBlock(width,heads,window_size,shifted=bool(i%2))
                                      for i in range(4)])
        self.latent = nn.Linear(width,latent_dim)
        self.latent_to_width = nn.Linear(latent_dim,width)
        self.output_proj = nn.Linear(width,n_channels*patch_size)

    def _patchify(self,x):
        # [B,C,L] -> [B,T,C*P]
        return x.reshape(len(x),self.n_channels,self.n_patches,self.patch_size).permute(0,2,1,3).reshape(len(x),self.n_patches,-1)

    def _unpatchify(self,p):
        return p.reshape(len(p),self.n_patches,self.n_channels,self.patch_size).permute(0,2,1,3).reshape(len(p),self.n_channels,self.length)

    def _token_features(self,x,mask=None):
        h = self.input_proj(self._patchify(x)) + self.pos_embed
        if mask is not None:
            h = torch.where(mask[:,:,None], self.mask_token+self.pos_embed, h)
        for block in self.blocks:
            h = block(h)
        return h

    def encode(self,x):
        h=self._token_features(x)
        return self.latent(h.mean(dim=1))

    def reconstruct(self,x):
        h=self._token_features(x)
        return self._unpatchify(self.output_proj(h + self.latent_to_width(self.latent(h.mean(dim=1)))[:,None,:]))

    def forward(self,x):
        return self.reconstruct(x)

    def loss(self,x,noise_std=0.):
        batch = len(x)
        # Au moins un patch masqué et au moins un visible par observation.
        count = min(self.n_patches-1,max(1,round(self.n_patches*self.mask_ratio)))
        mask = torch.rand((batch,self.n_patches),device=x.device).argsort(dim=1) < count
        h = self._token_features(x,mask=mask)
        pred = self._unpatchify(self.output_proj(h + self.latent_to_width(self.latent(h.mean(dim=1)))[:,None,:]))
        sq = (self._patchify(pred)-self._patchify(x)).square().mean(dim=2)
        masked_loss=(sq*mask).sum()/mask.sum().clamp_min(1)
        visible_loss=(sq*(~mask)).sum()/(~mask).sum().clamp_min(1)
        return masked_loss+0.1*visible_loss


class AdversarialAE(BaseAE):
    """AAE 1D à a priori gaussien, pour comparer AAE + GMM sur embeddings.

    AdversarialAE+GMM ≠ EM-AAE de l'article IEEE TII 2023 :
    il manque l'apprentissage EM conjoint et la synchronisation des domaines.
    """
    def __init__(self,n_channels=2,length=256,latent_dim=16,adv_weight=0.01):
        super().__init__()
        self.encoder = TemporalEncoder(n_channels,latent_dim)
        self.decoder = TemporalDecoder(n_channels,length,latent_dim)
        self.discriminator = nn.Sequential(nn.Linear(latent_dim,32),nn.LeakyReLU(.2),
                            nn.Linear(32,16),nn.LeakyReLU(.2),nn.Linear(16,1))
        self.adv_weight=adv_weight
    def encode(self,x): return self.encoder(x)
    def decode(self,z): return self.decoder(z)
    def reconstruct(self,x): return self.decode(self.encode(x))
    def forward(self,x): return self.reconstruct(x)