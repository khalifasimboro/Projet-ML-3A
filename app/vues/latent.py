"""Espace latent et clustering : comment l'autoencodeur organise les signaux."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

from ballguard.config import CLASSES, VITESSES
from ui import donnees
from ui.composants import ncle
from ui.composants import carte_chaleur, entete, html, kpi, lecture, pastille_etat, pied, titre_carte, tracer
from ui.i18n import T, nom_etat, nombre
from ui.theme import P, SERIE


@st.cache_data(show_spinner=False, max_entries=40)
def _mesures(rep, algo, k):
    bd = donnees.base()
    test = (bd.split == "test").to_numpy()
    cl = donnees.clusters_test(rep, algo, k)
    Z = donnees.artefacts()["latent"][f"Z_{rep}"][test]
    return {"ARI": adjusted_rand_score(bd.classe[test], cl), "NMI": normalized_mutual_info_score(bd.classe[test], cl),
            "silhouette": silhouette_score(Z, cl), "ARI_vitesse": adjusted_rand_score(bd.vitesse[test], cl)}


def page():
    p, art = P(), donnees.artefacts()
    bd = donnees.base()
    test = bd[bd.split == "test"]
    entete(T("Modèles", "Models"), T("Espace latent et clustering", "Latent space & clustering"),
           T("L’autoencodeur résume chaque signal en 6 nombres. Cette page montre comment les signaux s’y organisent, et pourquoi la vitesse devait être corrigée.",
             "The autoencoder summarises each signal in 6 numbers. This page shows how signals arrange themselves there, and why speed had to be corrected."))

    lecture(T("L’encodeur résume chaque signal en 6 nombres : c’est l’espace latent. Pour l’afficher, la projection t-SNE le ramène en 2 ou 3 dimensions. "
              "Chaque point est un signal de test. Si les couleurs (vrais états) forment des îlots séparés, le modèle distingue les défauts <b>sans avoir vu les étiquettes</b>. "
              "Changez la représentation pour voir l’effet de la correction de vitesse.",
              "The encoder summarises each signal in 6 numbers: the latent space. To display it, the t-SNE projection brings it down to 2 or 3 dimensions. "
              "Each point is a test signal. If the colours (real states) form separate islands, the model tells faults apart <b>without seeing the labels</b>. "
              "Switch the representation to see the effect of the speed correction."),
            [("ARI", T("Accord entre clusters et vrais états : 1 = identiques, 0 = hasard.", "Agreement between clusters and real states: 1 = identical, 0 = chance.")),
             ("Silhouette", T("Netteté des groupes, de −1 à 1.", "Group sharpness, from −1 to 1.")),
             (T("ARI vs vitesse", "ARI vs speed"), T("Doit rester proche de 0 : sinon, les groupes suivent la vitesse et non le défaut.", "Should stay close to 0, otherwise groups follow speed instead of the fault.")),
             (T("k", "k"), T("Nombre de groupes demandé à l’algorithme. Il y a 5 états réels : k = 5 est le bon choix.", "Number of groups asked of the algorithm. There are 5 real states: k = 5 is the right choice."))])

    with st.container(border=True, key=ncle()):
        a, b = st.columns([3, 1.3])
        libelles = {"sans": T("MLP-AE sans correction", "MLP-AE, no correction"), "corr": T("MLP-AE corrigé (retenu)", "MLP-AE corrected (selected)"),
                    "cnn": T("CNN-AE spectral corrigé", "Spectral CNN-AE corrected")}
        rep = a.segmented_control(T("Représentation", "Representation"), list(libelles), format_func=libelles.get, default="corr", key="lat_rep") or "corr"
        algo = b.segmented_control(T("Algorithme", "Algorithm"), ["km", "gmm"], format_func={"km": "K-means", "gmm": "GMM"}.get, default="km", key="lat_algo") or "km"
        b_, c, d = st.columns([1, 2, 2])
        dim = b_.segmented_control("Projection", ["2D", "3D"], default="2D", key="lat_dim") or "2D"
        coul = c.segmented_control(T("Couleur", "Colour"), ["etat", "cluster", "vitesse"], default="etat", key="lat_col",
                                   format_func={"etat": T("État", "State"), "cluster": "Cluster", "vitesse": T("Vitesse", "Speed")}.get) or "etat"
        k = d.slider(T("Nombre de clusters (k)", "Number of clusters (k)"), 2, 15, 5, key="lat_k")
    corrige = rep != "sans"

    with st.spinner(T("Calcul du clustering…", "Computing the clustering…")):
        cl, m = donnees.clusters_test(rep, algo, k), _mesures(rep, algo, k)
    c = st.columns(4)
    kpi(c[0], f"ARI {T('vs état', 'vs state')}", nombre(m["ARI"], 3), T("1 = clusters identiques aux états", "1 = clusters match states"))
    kpi(c[1], "NMI", nombre(m["NMI"], 3), T("Information partagée", "Shared information"))
    kpi(c[2], "Silhouette", nombre(m["silhouette"], 3), T("Netteté des groupes", "Group sharpness"))
    kpi(c[3], f"ARI {T('vs vitesse', 'vs speed')}", nombre(m["ARI_vitesse"], 3), T("Doit être proche de 0", "Should be close to 0"),
        couleur=p["ok"] if abs(m["ARI_vitesse"]) < 0.05 else p["alarm"])
    st.write("")

    g, d = st.columns([1.9, 1])
    with g.container(border=True, key=ncle()):
        titre_carte(T("Espace latent (jeu de test, projection t-SNE)", "Latent space (test set, t-SNE projection)"),
                    T("Après correction : un groupe par état, vitesses mélangées.", "After correction: one group per state, speeds mixed.") if corrige
                    else T("Sans correction : chaque état est coupé en îlots par la vitesse.", "No correction: each state is split into islands by speed."))
        if coul == "etat":
            html("".join(pastille_etat(e) for e in CLASSES))
            groupes = [(nom_etat(e), p[e], (test.classe == e).to_numpy()) for e in CLASSES]
        elif coul == "vitesse":
            groupes = [(f"{T('vitesse', 'speed')} {v}", col, (test.vitesse == v).to_numpy()) for v, col in zip(VITESSES, (p["BALL"], p["OUTER_RACE"], p["COMBINED"]))]
        else:
            groupes = [(f"cluster {j}", SERIE[j], cl == j) for j in range(k)]
        XY = art["latent"][f"tsne{dim[0]}_{rep}"]
        info = np.array([f"{nom_etat(e)} · {T('vitesse', 'speed')} {v} · cluster {j}" for e, v, j in zip(test.classe, test.vitesse, cl)])
        fig = go.Figure()
        for nom, col, masque in groupes:
            if dim == "3D":
                fig.add_scatter3d(x=XY[masque, 0], y=XY[masque, 1], z=XY[masque, 2], mode="markers", name=nom, marker=dict(size=3, color=col, opacity=0.85),
                                  text=info[masque], hovertemplate="%{text}<extra></extra>")
            else:
                fig.add_scattergl(x=XY[masque, 0], y=XY[masque, 1], mode="markers", name=nom, marker=dict(size=6, color=col, opacity=0.8),
                                  text=info[masque], hovertemplate="%{text}<extra></extra>")
        if dim == "3D":
            axe = dict(gridcolor=p["line"], backgroundcolor="rgba(0,0,0,0)", showticklabels=False, title="")
            fig.update_layout(scene=dict(xaxis=axe, yaxis=axe, zaxis=axe, bgcolor="rgba(0,0,0,0)"))
            tracer(fig, 520, legende=coul != "etat", marges=dict(l=0, r=0, t=0, b=0))
        else:
            fig.update_xaxes(showticklabels=False, title="t-SNE 1")
            fig.update_yaxes(showticklabels=False, title="t-SNE 2")
            tracer(fig, 520, legende=coul != "etat", marges=dict(l=30, r=10, t=10, b=60))
    with d:
        with st.container(border=True, key=ncle()):
            titre_carte(T("États × clusters", "States × clusters"), T("Une diagonale pleine = clustering parfait", "A full diagonal = perfect clustering"))
            z = [[int(((test.classe == e).to_numpy() & (cl == j)).sum()) for j in range(k)] for e in CLASSES]
            carte_chaleur(z, [f"c{j}" for j in range(k)], [nom_etat(e) for e in CLASSES], 250, texte=k <= 8, marges=dict(l=118, r=8, t=6, b=30))
        with st.container(border=True, key=ncle()):
            titre_carte(T("Choix de k", "Choosing k"), T("Silhouette sur le train, sans étiquettes", "Silhouette on train, no labels"))
            ks, cs = art["clustering"]["ks"], art["clustering"]
            fig = go.Figure()
            fig.add_scatter(x=ks, y=cs["sil_sans"], name=T("sans correction", "no correction"), line=dict(color=p["muted"], width=1.5))
            fig.add_scatter(x=ks, y=cs["sil_corr"], name=T("corrigé", "corrected"), line=dict(color=p["accent"], width=2.2))
            fig.add_scatter(x=[k], y=[(cs["sil_corr"] if corrige else cs["sil_sans"])[k - 2]], mode="markers", showlegend=False,
                            marker=dict(size=12, color=p["fg"], line=dict(color=p["accent"], width=3)))
            fig.update_layout(xaxis_title="k")
            tracer(fig, 230, marges=dict(l=40, r=8, t=6, b=60))
    pied()
