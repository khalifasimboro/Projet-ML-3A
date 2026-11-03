"""Entraînement et calcul de score avec initialisations reproductibles."""
import random
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset
from torch.nn import functional as F


def set_seed(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def train_model(model,x_train,x_val=None,epochs=15,batch_size=64,lr=1e-3,device="cpu",seed=42):
    set_seed(seed)
    model=model.to(device)
    is_adversarial = hasattr(model,"discriminator")
    if is_adversarial:
        discriminator_params=list(model.discriminator.parameters())
        generator_params=[p for name,p in model.named_parameters() if not name.startswith("discriminator.")]
        optimizer=torch.optim.Adam(generator_params,lr=lr)
        disc_optimizer=torch.optim.Adam(discriminator_params,lr=lr)
    else:
        optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    dataset=TensorDataset(torch.from_numpy(x_train))
    loader=DataLoader(dataset,batch_size=batch_size,shuffle=True)
    record=[]
    best_state=None;best_loss=float("inf")
    for epoch in range(1,epochs+1):
        model.train();total=0.;count=0
        for (batch,) in loader:
            batch=batch.to(device)
            if is_adversarial:
                model.discriminator.train()
                latent=model.encode(batch).detach()
                prior=torch.randn_like(latent)
                disc_optimizer.zero_grad(set_to_none=True)
                d_fake=model.discriminator(latent)
                d_real=model.discriminator(prior)
                d_loss=(F.binary_cross_entropy_with_logits(d_fake,torch.zeros_like(d_fake))
                        + F.binary_cross_entropy_with_logits(d_real,torch.ones_like(d_real)))*.5
                d_loss.backward()
                disc_optimizer.step()
                # Entraîner l'encodeur à rendre les encodages indiscernables de N(0,I).
                for p in discriminator_params:p.requires_grad_(False)
                z=model.encode(batch)
                logits=model.discriminator(z)
                loss=(F.mse_loss(model.decode(z),batch)
                      +model.adv_weight*F.binary_cross_entropy_with_logits(logits,torch.ones_like(logits)))
            else:
                loss=model.loss(batch)
            optimizer.zero_grad(set_to_none=True)
            if not torch.isfinite(loss): raise FloatingPointError("Perte non finie durant entraînement")
            loss.backward()
            optimizer.step()
            if is_adversarial:
                for p in discriminator_params:p.requires_grad_(True)
            total+=loss.item()*len(batch);count+=len(batch)
        val_loss=total/count
        if x_val is not None and len(x_val):
            # Pour le VAE, validation déterministe en reconstruction; pas de tirage latent.
            errors=reconstruction_scores(model,x_val,device=device)
            val_loss=float(np.mean(errors))
        record.append({"epoch":epoch,"train_loss":total/count,"validation_mse":val_loss})
        if val_loss < best_loss:
            best_loss=val_loss
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    if best_state is not None: model.load_state_dict(best_state)
    return record

@torch.no_grad()
def encode_all(model,x,batch_size=256,device="cpu"):
    model.eval().to(device)
    outs=[]
    for start in range(0,len(x),batch_size):
        x_b=torch.from_numpy(x[start:start+batch_size]).to(device)
        outs.append(model.encode(x_b).cpu().numpy())
    return np.concatenate(outs,axis=0)

@torch.no_grad()
def reconstruction_scores(model,x,batch_size=256,device="cpu"):
    """Erreur MSE moyenne par fenêtre; sans ajustement sur test."""
    model.eval().to(device)
    out=[]
    for start in range(0,len(x),batch_size):
        batch=torch.from_numpy(x[start:start+batch_size]).to(device)
        out.append(torch.mean((model.reconstruct(batch)-batch)**2,dim=(1,2)).cpu().numpy())
    return np.concatenate(out,axis=0)