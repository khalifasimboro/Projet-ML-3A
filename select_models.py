#!/usr/bin/env python
"""Sélection PROVISOIRE uniquement sur validation (GT2.05).

Un seul run / une seule graine n'est pas une preuve de supériorité. Après réception
les vraies données : réaliser plusieurs initialisations, intégrer la robustesse
inter-régimes, puis geler le choix avant exécution --phase final.
"""
import argparse
import json
from pathlib import Path
import pandas as pd


def select_candidates(results_csv, max_false_alarm=0.10):
    df=pd.read_csv(results_csv)
    required={'phase','task','model','ARI','average_precision','false_alarm_rate'}
    missing=required-set(df.columns)
    if missing:
        raise ValueError(f'Colonnes manquantes: {sorted(missing)}')
    if set(df['phase'])!={'screening'}:
        raise ValueError('Sélection INTERDITE à partir du test final ; utiliser seulement --phase screening')
    result={'status':'provisional_validation_only','max_false_alarm':max_false_alarm,
            'warnings':['Nécessite répétitions sur plusieurs seeds et régimes.',
                        'Ne PAS modifier ces choix après avoir consulté le test final.']}
    cl=df[(df.task=='clustering') & df.ARI.notna()]
    if not cl.empty:
        winner=cl.sort_values(['ARI','model'],ascending=[False,True]).iloc[0]
        result['clustering']={'model':str(winner.model),'validation_ARI':float(winner.ARI)}
    anomaly=df[(df.task=='anomaly') & df.average_precision.notna() &
               df.false_alarm_rate.notna() & (df.false_alarm_rate<=max_false_alarm)]
    if not anomaly.empty:
        winner=anomaly.sort_values(['average_precision','model'],ascending=[False,True]).iloc[0]
        result['anomaly']={'model':str(winner.model),
                           'validation_average_precision':float(winner.average_precision),
                           'validation_false_alarm_rate':float(winner.false_alarm_rate)}
    elif not df[df.task=='anomaly'].empty:
        result['anomaly']={'status':'NO_CANDIDATE_UNDER_FALSE_ALARM_LIMIT'}
    if not ('clustering' in result or 'anomaly' in result):
        raise ValueError('Aucun modèle évaluable dans ce tableau')
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--results',type=Path,required=True)
    ap.add_argument('--out',type=Path,default=Path('selected_models.json'))
    ap.add_argument('--max-false-alarm',type=float,default=0.10)
    args=ap.parse_args()
    if not (0 <= args.max_false_alarm <= 1):
        ap.error('--max-false-alarm doit être entre 0 et 1')
    result=select_candidates(args.results,args.max_false_alarm)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=='__main__': main()