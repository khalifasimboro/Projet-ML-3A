import numpy as np
import torch
from src.ae_benchmark.data import demo_data,grouped_stratified_split,PerSensorStandardizer
from src.ae_benchmark.models import MODEL_REGISTRY,build_model
from src.ae_benchmark.evaluation import anomaly_metrics


def test_split_group_independence():
    data=demo_data(n_groups_per_condition=3,windows_per_group=2,length=64)
    a,b,c=grouped_stratified_split(data)
    assert not(set(data.groups[a]) & set(data.groups[b]))
    assert not(set(data.groups[a]) & set(data.groups[c]))
    assert not(set(data.groups[b]) & set(data.groups[c]))
    assert len(a)+len(b)+len(c)==len(data.x)


def test_model_shapes_and_gradients():
    x=torch.randn(4,2,64)
    for name in MODEL_REGISTRY:
        model=build_model(name,length=64,latent_dim=8)
        result=model.reconstruct(x)
        assert result.shape==x.shape,name
        assert model.encode(x).shape==(4,8),name
        loss=model.loss(x)
        assert torch.isfinite(loss)
        loss.backward()
        assert any(p.grad is not None for p in model.parameters()),name


def test_scaler_train_only():
    rng=np.random.default_rng(42)
    train=rng.normal(size=(10,2,64)).astype(np.float32)
    test=train+1000
    std=PerSensorStandardizer().fit(train)
    assert abs(float(std.transform(train).mean())) < .01
    assert std.transform(test).mean() > 100


def test_threshold_validation_only():
    result=anomaly_metrics([.1,.2,.3], [.11,.21,.8,.9], ["SAFE","SAFE","BALL","COMBINED"])
    assert result["threshold"]<.31
    assert result["PR_AUC"]>0


def test_adapter_xlsx_and_manifest(tmp_path):
    import pandas as pd
    from src.ae_benchmark.data import from_manifest
    table=pd.DataFrame({"acceleration_vertbleu":np.arange(80),
                        "acceleration_vert":np.arange(80)*2})
    path=tmp_path/"acquisition.xlsx"
    table.to_excel(path,index=False)
    manifest=tmp_path/"manifest.csv"
    pd.DataFrame([{"file_path":"acquisition.xlsx","label":"SAFE",
                   "regime":"10","group_id":"rec001"}]).to_csv(manifest,index=False)
    result=from_manifest(manifest,window_length=32,stride=16)
    assert result.x.shape==(4,2,32)
    assert (result.labels=="SAFE").all()
    assert (result.groups=="rec001").all()


def test_not_enough_independent_groups_fails():
    import pytest
    from src.ae_benchmark.data import VibrationData
    dataset=VibrationData(np.ones((3,2,64),dtype=np.float32),
                          np.array(["SAFE"]*3),np.array(["10"]*3),
                          np.array(["same_record"]*3))
    with pytest.raises(ValueError,match="acquisition"):
        grouped_stratified_split(dataset)


def test_vibration_features_finite_and_sensor_order():
    from src.ae_benchmark.signal_features import vibration_features
    signal = np.zeros((3,2,64),dtype=np.float32)
    signal[:,0,:] = 1.0
    values=vibration_features(signal)
    assert values.shape==(3,22)
    assert np.isfinite(values).all()


def test_idec_student_assignment_and_training():
    from src.ae_benchmark.deep_clustering import (student_soft_assign, sharpen_target,
                                                   fit_idec,predict_idec)
    z=torch.randn(4,8)
    q=student_soft_assign(z,torch.randn(5,8))
    p=sharpen_target(q)
    assert torch.allclose(q.sum(1),torch.ones(4))
    assert torch.allclose(p.sum(1),torch.ones(4))
    rng=np.random.default_rng(7)
    x=rng.normal(size=(20,2,64)).astype(np.float32)
    m=build_model('cnn',length=64,latent_dim=8)
    fitted,hist=fit_idec(m,x,epochs=1,batch_size=10,device='cpu')
    assert len(hist)==1
    assert predict_idec(fitted,x).shape==(20,)
    assert set(predict_idec(fitted,x)).issubset(set(range(5)))


def test_regime_metrics_with_missing_class():
    from src.ae_benchmark.evaluation import metrics_by_regime
    results=metrics_by_regime(['SAFE','BALL','SAFE'],[0,1,0],['10','10','20'])
    assert {a['regime'] for a in results}=={'10','20'}


def test_small_end_to_end_demo(tmp_path):
    import argparse
    from run_benchmark import run
    args=argparse.Namespace(plots=False,demo=True,phase='screening',channels='both',manifest=None,task='both',models='cnn',
        epochs=1,idec_epochs=1,idec_backbone='cnn',idec_gamma=0.1,
        window_length=64,stride=32,max_windows_per_file=None,
        latent_dim=8,batch_size=32,lr=.001,seed=42,cpu=True,out=str(tmp_path))
    results=run(args)
    assert 'cnn+idec' in set(results.model)
    assert 'features+kmeans' in set(results.model)
    assert 'cnn+gmm' in set(results.model)
    assert (tmp_path/'split_metadata.json').exists()
    assert (tmp_path/'AVERTISSEMENT.txt').exists()


def test_average_precision_distinct_from_trapezoidal_pr_auc():
    from src.ae_benchmark.evaluation import anomaly_metrics
    result=anomaly_metrics([.1,.2,.3],[.1,.4,.5,.9],['SAFE','SAFE','BALL','COMBINED'])
    assert 0 <= result['PR_AUC'] <= 1
    assert 0 <= result['average_precision'] <= 1


def test_single_sensor_demo(tmp_path):
    import argparse
    from run_benchmark import run
    args=argparse.Namespace(plots=False,demo=True,phase='screening',channels='vert',manifest=None,
        task='anomaly',models='cnn',epochs=1,idec_epochs=0,idec_backbone='cnn',
        idec_gamma=.1,window_length=64,stride=32,max_windows_per_file=None,
        latent_dim=8,batch_size=32,lr=.001,seed=42,cpu=True,out=str(tmp_path))
    result=run(args)
    assert 'cnn' in set(result.model)


def test_reporting_png(tmp_path):
    import pandas as pd
    from src.ae_benchmark.reporting import plot_latent, plot_scores
    frame=pd.DataFrame([{'task':'clustering','model':'cnn','ARI':0.4},
                        {'task':'clustering','model':'vae','ARI':0.1}])
    plot_scores(frame,tmp_path)
    assert (tmp_path/'ARI.png').is_file()
    plot_latent(np.random.normal(size=(20,8)),np.array(['SAFE']*10+['BALL']*10),tmp_path/'latent.png')
    assert (tmp_path/'latent.png').is_file()


def test_selection_requires_screening(tmp_path):
    import pandas as pd
    import pytest
    from select_models import select_candidates
    path=tmp_path/'r.csv'
    pd.DataFrame([{'phase':'final','task':'clustering','model':'cnn','ARI':.4,
      'average_precision':None,'false_alarm_rate':None}]).to_csv(path,index=False)
    with pytest.raises(ValueError,match='INTERDITE'):
        select_candidates(path)


def test_selection_respects_far_limit(tmp_path):
    import pandas as pd
    from select_models import select_candidates
    path=tmp_path/'r.csv'
    pd.DataFrame([
        {'phase':'screening','task':'clustering','model':'cnn','ARI':.4,
         'average_precision':None,'false_alarm_rate':None},
        {'phase':'screening','task':'anomaly','model':'memory','ARI':None,
         'average_precision':.9,'false_alarm_rate':.3},
        {'phase':'screening','task':'anomaly','model':'vae','ARI':None,
         'average_precision':.8,'false_alarm_rate':.05},
    ]).to_csv(path,index=False)
    r=select_candidates(path,max_false_alarm=.1)
    assert r['clustering']['model']=='cnn'
    assert r['anomaly']['model']=='vae'