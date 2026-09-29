"""B2: is the low weight-space agreement an artefact of normalising the exact and
predicted fronts over their own ideals and nadirs?  Recompute with COMMON anchors
(the exact front's ideal and nadir applied to both) and compare."""
import numpy as np, pandas as pd, os
R="/private/tmp/claude-501/-Users-amin-Documents-7-Overleaf-Isa-Article/80df824b-267f-4787-aaf2-a5a1ba1cee5c/scratchpad/Supplementary_File_S1_analysis_reproducibility/results"
HZ=[2020,2050,2100]
MAIN5=["shared_mtl_nn","shared_mtl_nn_mgda","independent_stl_nn","random_forest","gradient_boosting"]
rng=np.random.default_rng(42); W=rng.dirichlet(np.ones(4),size=10000)
def _half_up(x, d=2):
    f=10**d
    return float(int(x*f+(0.5 if x>=0 else -0.5)))/f

def pmask(F):
    leq=(F[None,:,:]<=F[:,None,:]).all(axis=2); lt=(F[None,:,:]<F[:,None,:]).any(axis=2)
    return ~(leq&lt).any(axis=1)
def win(F,ids,mn=None,sp=None):
    if mn is None: mn=F.min(0)
    if sp is None: sp=F.max(0)-mn
    Fn=(F-mn)/np.where(sp>0,sp,1.0); o=np.argsort(ids)
    return ids[o[np.argmin((W@Fn.T)[:,o],axis=1)]]
ex=pd.read_csv(f"{R}/exhaustive_analysis/pareto_fronts.csv")
EW={};ANCH={}
for hz in HZ:
    s=ex[ex.horizon==hz]
    F=np.column_stack([s.annual_heating_energy_gj,s.cost_rate_sum_proxy,s.carbon_rate_sum_proxy,-s.days_below_24_c]).astype(float)
    mn=F.min(0); sp=F.max(0)-mn; ANCH[hz]=(mn,sp)
    EW[hz]=win(F,s.simulation_id.to_numpy(int))
rows=[]
for design,path,clip in [("all-predicted",f"{R}/surrogate_validation_family4/oof_predictions.csv",True),
                         ("exact-index",f"{R}/surrogate_validation/oof_predictions.csv",False)]:
    oof=pd.read_csv(path)
    for model,mdf in oof.groupby("model"):
        if model not in MAIN5: continue
        for rep,rdf in mdf.groupby("repeat_seed"):
            for hz in HZ:
                sub=rdf[rdf.horizon==hz]
                d24=np.clip(sub.predicted_days_below_24_c.to_numpy(float),0,365)
                if clip:
                    c=np.clip(sub.predicted_cost_rate_sum_proxy.to_numpy(float),0,None)
                    g=np.clip(sub.predicted_carbon_rate_sum_proxy.to_numpy(float),0,None)
                else:
                    c=sub.cost_rate_sum_proxy.to_numpy(float); g=sub.carbon_rate_sum_proxy.to_numpy(float)
                F=np.column_stack([sub.predicted_annual_heating_energy_gj.to_numpy(float),c,g,-d24])
                ids=sub.simulation_id.to_numpy(int); m=pmask(F)
                own=np.mean(win(F[m],ids[m])==EW[hz])*100
                mn,sp=ANCH[hz]
                com=np.mean(win(F[m],ids[m],mn,sp)==EW[hz])*100
                rows.append(dict(task_design=design,model=model,repeat_seed=int(rep),horizon=hz,
                                 own_anchor_pct=_half_up(own,2),common_anchor_pct=_half_up(com,2)))
df=pd.DataFrame(rows); df["delta"]=df.common_anchor_pct-df.own_anchor_pct
df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)),"mcda_common_anchor.csv"),index=False)
print("Weight-space agreement, own-front anchors vs common (exact-front) anchors\n")
for d in ["all-predicted","exact-index"]:
    print(f"== {d} ==")
    for m in MAIN5:
        s=df[(df.task_design==d)&(df.model==m)]
        print(f"  {m:<22} own {s.own_anchor_pct.mean():5.1f}%   common {s.common_anchor_pct.mean():5.1f}%   delta {s.delta.mean():+5.2f}")
    print()
print(f"mean |delta| over all 90 composites: {df.delta.abs().mean():.2f} points")
print(f"max  |delta|:                        {df.delta.abs().max():.2f} points")
