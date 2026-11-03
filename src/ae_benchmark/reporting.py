"""Figures exploratoires. Ne jamais publier des figures générées en mode --demo."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_scores(frame, outdir):
    """Trace séparément les métriques principales, sans valeur synthétique ajoutée."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for metric in ('ARI', 'NMI', 'PR_AUC', 'average_precision', 'F1', 'false_alarm_rate'):
        if metric not in frame:
            continue
        subset = frame[['task', 'model', metric]].dropna()
        if subset.empty:
            continue
        fig, ax = plt.subplots(figsize=(max(7, len(subset) * 0.55), 4.3))
        names = [f'{row.task} / {row.model}' for row in subset.itertuples()]
        ax.bar(np.arange(len(subset)), subset[metric].to_numpy(dtype=float))
        ax.set_xticks(np.arange(len(names)), names, rotation=60, ha='right', fontsize=8)
        ax.set_ylabel(metric)
        ax.set_title(f'Comparaison exploratoire — {metric}')
        ax.grid(axis='y', alpha=.3)
        fig.tight_layout()
        fig.savefig(outdir / f'{metric}.png', dpi=160)
        plt.close(fig)


def plot_latent(z, labels, filename):
    """Projection PCA 2D à visée VISUELLE, jamais une preuve de cluster séparé."""
    z = np.asarray(z, dtype=float)
    labels = np.asarray(labels)
    if len(z) < 3 or z.ndim != 2:
        return
    if z.shape[1] >= 2:
        projected = PCA(n_components=2).fit_transform(z)
    else:
        projected = np.column_stack((z[:, 0], np.zeros(len(z))))
    fig, ax = plt.subplots(figsize=(6, 5))
    for label in sorted(np.unique(labels)):
        sel = labels == label
        ax.scatter(projected[sel, 0], projected[sel, 1], label=label, s=12, alpha=.7)
    ax.legend(fontsize=8)
    ax.set_xlabel('PCA 1 (espace latent)')
    ax.set_ylabel('PCA 2 (espace latent)')
    ax.set_title('Visualisation descriptive des états connus')
    fig.tight_layout()
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename, dpi=160)
    plt.close(fig)