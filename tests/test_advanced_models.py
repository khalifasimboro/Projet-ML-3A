"""Tests déterministes d'interface, gradients et adaptations aux signaux."""
import numpy as np
import pytest
import torch
from src.ae_benchmark.models import build_model
from src.ae_benchmark.advanced_models import SparseMemory, AdaptiveShrinkageUnit
from src.ae_benchmark.signal_processing import signal_representation, angular_order_envelope
from src.ae_benchmark.training import train_model


@pytest.mark.parametrize('model_name', ['memae','cmae','adae','masked_swin','aae'])
def test_advanced_architecture_shapes_and_gradients(model_name):
    torch.manual_seed(9)
    x = torch.randn(4,2,64)
    model = build_model(model_name, length=64, channels=2, latent_dim=8)
    assert model.reconstruct(x).shape == x.shape
    assert model.encode(x).shape == (len(x),8)
    loss = model.loss(x)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_sparse_memory_topk_and_normalization():
    mem=SparseMemory(latent_dim=8,n_items=11,topk=3,temperature=.2)
    z=torch.randn(5,8,requires_grad=True)
    reconstruction,ent,weights=mem(z)
    assert reconstruction.shape==(5,8)
    assert torch.all((weights>0).sum(dim=1)==3)
    assert torch.allclose(weights.sum(dim=1),torch.ones(5),atol=1e-6)
    assert ent.item() > 0
    (reconstruction.square().mean()+ent*.001).backward()
    assert mem.memory.grad is not None
    assert torch.isfinite(mem.memory.grad).all()


def test_memae_training_updates_memory():
    rng=np.random.default_rng(6)
    x=rng.normal(size=(16,2,64)).astype(np.float32)
    model=build_model('memae',64,2,8)
    before=model.memory.memory.detach().clone()
    hist=train_model(model,x,x,epochs=1,batch_size=8,seed=6)
    assert len(hist)==1
    assert not torch.allclose(before,model.memory.memory)


def test_aae_discriminator_optimizes_separately():
    x=np.random.default_rng(2).normal(size=(12,2,64)).astype(np.float32)
    m=build_model('aae',64,2,8)
    before=m.discriminator[0].weight.detach().clone()
    train_model(m,x,x,epochs=1,batch_size=6,seed=2)
    assert not torch.allclose(before,m.discriminator[0].weight)


def test_swin_patch_ratio_and_shift_attention():
    m=build_model('masked_swin',64,2,8)
    x=torch.randn(2,2,64)
    a=m.loss(x)
    assert a.ndim==0 and torch.isfinite(a)
    assert m.blocks[1].shift==2
    with pytest.raises(ValueError):
        build_model('masked_swin',66,2,8)


def test_representations_are_finite_and_sized():
    x=np.random.default_rng(0).normal(size=(4,2,64)).astype(np.float32)
    for name,length in [('raw',64),('envelope',64),('log_spectrum',33)]:
        y=signal_representation(x,name)
        assert y.shape==(4,2,length)
        assert np.isfinite(y).all()
    with pytest.raises(ValueError):signal_representation(x,'wavelet_fake')


def test_order_envelope_requires_actual_rpm_and_fs():
    fs=800.
    t=np.arange(0,5.,1/fs)
    rpm=600 + 70*np.sin(2*np.pi*.6*t)
    cycles=np.cumsum(rpm/60)/fs
    vibration=np.sin(2*np.pi*3*cycles)*(1+.2*np.sin(2*np.pi*.4*t))
    e,axis=angular_order_envelope(vibration,rpm,fs,samples_per_revolution=32)
    assert e.ndim==1 and len(e)==len(axis)
    assert np.all(np.diff(axis)>0) and np.isfinite(e).all()
    with pytest.raises(ValueError):angular_order_envelope(vibration,np.full(2,600.),fs)
    with pytest.raises(ValueError):angular_order_envelope(vibration,np.zeros_like(rpm),fs)


def test_demo_advanced_model_pipeline(tmp_path):
    import argparse
    from run_benchmark import run
    config=argparse.Namespace(plots=False,demo=True,phase='screening',channels='both',representation='envelope',
          manifest=None,task='anomaly',models='memae,cmae,adae,masked_swin,aae',epochs=1,
          idec_epochs=0,idec_backbone='cnn',idec_gamma=.1,window_length=64,stride=32,
          max_windows_per_file=None,latent_dim=8,batch_size=64,lr=.001,seed=42,cpu=True,out=str(tmp_path))
    df=run(config)
    assert {'memae','cmae','adae','masked_swin','aae'}.issubset(set(df.model))
    assert (df.phase=='screening').all()
    assert (tmp_path/'anomaly'/'memae.pt').is_file()
    assert (tmp_path/'anomaly'/'masked_swin.pt').is_file()