"""Decision fidelity diagnostics from retained predictions, without model fitting.

Usage: python within_fold_selection_check.py --s1-root PACKAGE --output-dir OUT
Defaults are relative to installation in PACKAGE/supplementary_analysis/.
The input package is read-only. No simulations or training are performed.
"""
from pathlib import Path
import argparse
import json
import hashlib
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--s1-root', type=Path, default=SCRIPT_DIR.parent)
parser.add_argument('--output-dir', type=Path, default=SCRIPT_DIR/'decision_fidelity_diagnostics')
args = parser.parse_args()
ROOT = args.s1_root.resolve()
OUT = args.output_dir.resolve()
OUT.mkdir(parents=True, exist_ok=True)
S1 = ROOT/'results'
MODELS = ['shared_mtl_nn', 'shared_mtl_nn_mgda', 'independent_stl_nn',
          'random_forest', 'gradient_boosting']
DESIGNS = {'all-predicted': 'surrogate_validation_family4',
           'exact-index': 'surrogate_validation'}
PROFILES = {'cost_priority': [.1,.7,.1,.1],
            'carbon_priority': [.1,.1,.7,.1],
            'energy_priority': [.7,.1,.1,.1],
            'days_below_24_priority': [.1,.1,.1,.7],
            'balanced': [.25,.25,.25,.25]}
WEIGHTS = np.random.default_rng(42).dirichlet(np.ones(4), size=10000)

def pareto(a):
    # Row i is dominated if some row j is <= on every objective and < on one.
    dominates = (a[None,:,:] <= a[:,None,:]).all(2) & (a[None,:,:] < a[:,None,:]).any(2)
    return ~dominates.any(1)

def normalize(a, low=None, span=None):
    if low is None: low = a.min(0)
    if span is None: span = a.max(0) - low
    return np.divide(a-low, span, out=np.zeros_like(a), where=span>0), low, span

def stable_choice(scores):
    # Candidate rows are sorted by simulation ID: first near-minimum gives tie rule.
    return np.flatnonzero(np.isclose(scores, scores.min(), rtol=0, atol=1e-12))[0]

def weight_winners(normalized, positions):
    # Exact-equality argmin is the existing weight-sweep tie convention.
    scores = sum(WEIGHTS[:,j,None] * normalized[None,:,j] for j in range(4))
    return positions[np.argmin(scores, axis=1)]

groups, decisions, fingerprints = [], [], {}
exact_cache = {}
global_cache = {}
for design, directory in DESIGNS.items():
    path = S1 / directory / 'oof_predictions.csv'
    fingerprints[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    frame = pd.read_csv(path)
    frame = frame[frame.model.isin(MODELS)]
    for (seed, fold, model, horizon), part in frame.groupby(['repeat_seed','outer_fold','model','horizon'], sort=True):
        part = part.sort_values('simulation_id', kind='stable')
        ids = part.simulation_id.to_numpy(int)
        assert len(ids)==125 and len(set(ids))==125
        exact = np.column_stack([part.true_annual_heating_energy_gj,
                                 part.cost_rate_sum_proxy, part.carbon_rate_sum_proxy,
                                 -part.true_days_below_24_c])
        key = (int(seed), int(fold), int(horizon))
        if key not in exact_cache:
            mask = pareto(exact)
            positions = np.flatnonzero(mask)
            scaled, low, span = normalize(exact[mask])
            exact_cache[key] = (ids.copy(), exact.copy(), mask, positions, scaled,
                                low, span, weight_winners(scaled, positions))
        ref_ids, ref_exact, exact_mask, exact_positions, exact_scaled, low, span, exact_weights = exact_cache[key]
        assert np.array_equal(ids,ref_ids) and np.allclose(exact,ref_exact,rtol=0,atol=0)
        if int(horizon) not in global_cache:
            allpart = frame[(frame.repeat_seed==seed)&(frame.model==model)&(frame.horizon==horizon)].sort_values('simulation_id')
            allvalues = np.column_stack([allpart.true_annual_heating_energy_gj,
                                        allpart.cost_rate_sum_proxy, allpart.carbon_rate_sum_proxy,
                                        -allpart.true_days_below_24_c])
            assert len(allvalues)==625
            global_cache[int(horizon)] = set(allpart.simulation_id.to_numpy(int)[pareto(allvalues)])
        cost = (np.maximum(part.predicted_cost_rate_sum_proxy.to_numpy(),0)
                if design=='all-predicted' else part.cost_rate_sum_proxy.to_numpy())
        carbon = (np.maximum(part.predicted_carbon_rate_sum_proxy.to_numpy(),0)
                  if design=='all-predicted' else part.carbon_rate_sum_proxy.to_numpy())
        predicted = np.column_stack([part.predicted_annual_heating_energy_gj, cost, carbon,
                                    -np.clip(part.predicted_days_below_24_c.to_numpy(),0,365)])
        predicted_mask = pareto(predicted)
        predicted_positions = np.flatnonzero(predicted_mask)
        predicted_scaled, _, _ = normalize(predicted[predicted_mask])
        predicted_weights = weight_winners(predicted_scaled, predicted_positions)
        overlap = np.count_nonzero(exact_mask & predicted_mask)
        precision = overlap / np.count_nonzero(predicted_mask)
        recall = overlap / np.count_nonzero(exact_mask)
        info = {'design':design,'model':model,'repeat_seed':int(seed),
                'outer_fold':int(fold),'horizon':int(horizon)}
        groups.append(dict(info,candidate_count=len(ids),exact_front_size=int(exact_mask.sum()),
                           predicted_front_size=int(predicted_mask.sum()),precision=precision,
                           recall=recall,f1=2*precision*recall/(precision+recall),
                           weight_space_agreement=float(np.mean(ids[exact_weights]==ids[predicted_weights]))))
        exact_all_scaled,_,_ = normalize(exact,low,span)
        for profile, weights in PROFILES.items():
            w = np.array(weights)
            e = exact_positions[stable_choice(exact_scaled@w)]
            p = predicted_positions[stable_choice(predicted_scaled@w)]
            regret = float((exact_all_scaled[p]-exact_all_scaled[e])@w)
            assert regret>=-1e-12
            decisions.append(dict(info,profile=profile,exact_id=int(ids[e]),predicted_id=int(ids[p]),
                                  selection_agreement=bool(ids[e]==ids[p]),
                                  selected_on_exact_subset_front=bool(exact_mask[p]),
                                  selected_on_exact_full_grid_front=bool(ids[p] in global_cache[int(horizon)]),
                                  regret=max(0,regret)))

g=pd.DataFrame(groups); d=pd.DataFrame(decisions)
assert len(g)==450 and len(d)==2250
summary=[]
for (design,model), p in d.groupby(['design','model'], sort=False):
    q=g[(g.design==design)&(g.model==model)]
    mm=p[~p.selection_agreement]
    summary.append({'design':design,'model':model,'group_count':len(q),
                    'profile_count':len(p),'profile_matches':int(p.selection_agreement.sum()),
                    'profile_agreement':float(p.selection_agreement.mean()),
                    'f1_mean':float(q.f1.mean()),'f1_sd':float(q.f1.std(ddof=1)),
                    'weight_agreement_mean':float(q.weight_space_agreement.mean()),
                    'weight_agreement_sd':float(q.weight_space_agreement.std(ddof=1)),
                    'mismatches':len(mm),'mismatch_subset_nondominated_fraction':float(mm.selected_on_exact_subset_front.mean()),
                    'mismatch_fullgrid_nondominated_fraction':float(mm.selected_on_exact_full_grid_front.mean()),
                    'mismatch_median_regret':float(mm.regret.median()),
                    'all_profile_median_regret':float(p.regret.median()),'max_regret':float(p.regret.max())})
agg=[]
for design,p in d.groupby('design'):
    mm=p[~p.selection_agreement]
    agg.append({'design':design,'profile_comparisons':len(p),'mismatches':len(mm),
                'mismatch_subset_nondominated':int(mm.selected_on_exact_subset_front.sum()),
                'mismatch_subset_nondominated_fraction':float(mm.selected_on_exact_subset_front.mean()),
                'mismatch_fullgrid_nondominated_fraction':float(mm.selected_on_exact_full_grid_front.mean()),
                'mismatch_median_regret':float(mm.regret.median()),
                'max_regret':float(mm.regret.max())})
g.to_csv(OUT/'group_metrics.csv',index=False)
d.to_csv(OUT/'profile_decisions.csv',index=False)
pd.DataFrame(summary).to_csv(OUT/'summary.csv',index=False)
(OUT/'verification.json').write_text(json.dumps({'input_sha256':fingerprints,
    'weight_seed':42,'weight_draws':10000,'candidate_count_per_group':125,
    'groups':len(g),'profile_decisions':len(d),'models':summary,'aggregate':agg},indent=2)+'\n')

# Exact winner-to-runner-up margins are counted once per candidate set/profile,
# rather than duplicated across surrogate models or task designs.
margin_rows=[]
def add_margins(scope, identifiers, values, extra):
    mask=pareto(values)
    positions=np.flatnonzero(mask)
    scaled,low,span=normalize(values[mask])
    allscaled,_,_=normalize(values,low,span)
    for profile,weights in PROFILES.items():
        scores=scaled@np.array(weights)
        order=np.argsort(scores,kind='stable')
        alls=allscaled@np.array(weights)
        allo=np.argsort(alls,kind='stable')
        margin_rows.append(dict(scope=scope,profile=profile,**extra,
             candidate_count=len(values),exact_front_size=int(mask.sum()),
             winner_id=int(identifiers[positions[order[0]]]),
             runner_up_front_id=int(identifiers[positions[order[1]]]),
             front_score_margin=float(scores[order[1]]-scores[order[0]]),
             runner_up_all_id=int(identifiers[allo[1]]),
             all_candidates_score_margin=float(alls[allo[1]]-alls[allo[0]]),
             d24_front_span=float(span[3])))

for (seed,fold,horizon),cache in exact_cache.items():
    add_margins('within125',cache[0],cache[1],
                dict(repeat_seed=seed,outer_fold=fold,horizon=horizon))
all_config_path=S1/'exhaustive_analysis/all_configurations.csv'
configs=pd.read_csv(all_config_path)
for horizon,part in configs.groupby('horizon'):
    part=part.sort_values('simulation_id')
    values=np.column_stack([part.annual_heating_energy_gj,part.cost_rate_sum_proxy,
                            part.carbon_rate_sum_proxy,-part.days_below_24_c])
    add_margins('full625',part.simulation_id.to_numpy(int),values,
                dict(repeat_seed=-1,outer_fold=-1,horizon=int(horizon)))
margins=pd.DataFrame(margin_rows)
margin_summary=[]
for scope,part in margins.groupby('scope'):
    for profile,p in [('all_profiles',part),*list(part.groupby('profile'))]:
        margin_summary.append(dict(scope=scope,profile=profile,n=len(p),
          front_margin_below_004=int((p.front_score_margin<.004).sum()),
          fraction_front_margin_below_004=float((p.front_score_margin<.004).mean()),
          median_front_margin=float(p.front_score_margin.median()),
          median_all_candidate_margin=float(p.all_candidates_score_margin.median()),
          changed_runner_up_count=int((p.runner_up_front_id!=p.runner_up_all_id).sum()),
          strictly_lower_all_candidate_margin_count=int((p.all_candidates_score_margin<p.front_score_margin-1e-12).sum())))

# Retained full-grid decisions follow the manuscript's clipping and tie policies.
full=[]
for design,directory in DESIGNS.items():
    p=pd.read_csv(S1/directory/'weighted_profile_validation.csv')
    p=p[p.model.isin(MODELS)].copy()
    p=p.rename(columns={'true_score_regret':'regret',
                         'predicted_selection_on_exact_pareto':'selected_on_exact_subset_front',
                         'exact_simulation_id':'exact_id','predicted_simulation_id':'predicted_id'})
    p['design']=design;p['scope']='full625';full.append(p)
full=pd.concat(full,ignore_index=True)
within=d.copy();within['scope']='within125'
both=pd.concat([full,within],ignore_index=True)
tails=[]
for (scope,design),part in both.groupby(['scope','design']):
    for model,p in [('all_models',part),*list(part.groupby('model'))]:
        for population,pp in [('all_profile_comparisons',p),('mismatches',p[~p.selection_agreement])]:
            rr=pp.regret.to_numpy(float)
            tails.append(dict(scope=scope,design=design,model=model,population=population,n=len(rr),
                 median_regret=float(np.median(rr)),p95_regret=float(np.quantile(rr,.95,method='linear')),
                 max_regret=float(rr.max()),count_regret_above_005=int((rr>.05).sum()),
                 fraction_regret_above_005=float((rr>.05).mean())))

associations=[]
for scope,part in both.groupby('scope'):
    join=['horizon','profile'] if scope=='full625' else ['repeat_seed','outer_fold','horizon','profile']
    part=part.merge(margins[margins.scope==scope][join+['front_score_margin']],on=join,validate='many_to_one')
    for design,p in part.groupby('design'):
        for condition,pp in [('margin_lt_004',p[p.front_score_margin<.004]),
                             ('margin_ge_004',p[p.front_score_margin>=.004])]:
            associations.append(dict(scope=scope,design=design,margin_bin=condition,n=len(pp),
                         identity_agreement=float(pp.selection_agreement.mean()),
                         median_regret=float(pp.regret.median())))

# Inspect the most consequential exact-index held-out-subset disagreement.
worstrow=d[d.design=='exact-index'].sort_values('regret',ascending=False,kind='stable').iloc[0]
winfo={k:(v.item() if isinstance(v,np.generic) else v) for k,v in worstrow.to_dict().items()}
raw=pd.read_csv(S1/DESIGNS['exact-index']/'oof_predictions.csv')
subset=raw[(raw.model==worstrow.model)&(raw.repeat_seed==worstrow.repeat_seed)&
           (raw.outer_fold==worstrow.outer_fold)&(raw.horizon==worstrow.horizon)].sort_values('simulation_id')
cache=exact_cache[(int(worstrow.repeat_seed),int(worstrow.outer_fold),int(worstrow.horizon))]
ids,values,mask,positions,scaled,low,span,_=cache
ep=int(np.flatnonzero(ids==worstrow.exact_id)[0]);pp=int(np.flatnonzero(ids==worstrow.predicted_id)[0])
w=np.asarray(PROFILES[worstrow.profile]);contribution=np.divide(values[pp]-values[ep],span,out=np.zeros(4),where=span>0)*w
winfo.update(profile_weights_E_C_G_minusD=w.tolist(),exact_front_minimization_ideal=low.tolist(),
             exact_front_spans_E_C_G_minusD=span.tolist(),weighted_regret_contributions_E_C_G_minusD=contribution.tolist(),
             objective_order=['E_H','I_C','I_G','minus_D24'],
             exact_selection_true_E_C_G_D=[*values[ep,:3].tolist(),float(-values[ep,3])],
             surrogate_selection_true_E_C_G_D=[*values[pp,:3].tolist(),float(-values[pp,3])])
for name,which in [('exact_selection',worstrow.exact_id),('surrogate_selection',worstrow.predicted_id)]:
    row=subset[subset.simulation_id==which].iloc[0]
    winfo[name+'_raw_predicted_E_D']=[float(row.predicted_annual_heating_energy_gj),float(row.predicted_days_below_24_c)]
assert abs(sum(contribution)-winfo['regret'])<1e-12

# Physical comparison is descriptive checking of retained exact outputs.
refs=pd.read_csv(S1/'exhaustive_analysis/reference_solutions.csv')
selections=pd.read_csv(S1/'exhaustive_analysis/mcdm_weighted_sum_selections.csv')
physical=[]
for horizon,part in configs.groupby('horizon'):
    baseline=part[part.simulation_id==1].iloc[0]
    minimum=part.loc[part.annual_heating_energy_gj.idxmin()]
    worst=part.loc[part.days_below_24_c.idxmin()]
    physical.append(dict(horizon=int(horizon),baseline_E_H=float(baseline.annual_heating_energy_gj),
        baseline_D24=int(baseline.days_below_24_c),minimum_heating_id=int(minimum.simulation_id),
        minimum_heating_E_H=float(minimum.annual_heating_energy_gj),minimum_heating_D24=int(minimum.days_below_24_c),
        minimum_heating_reduction_percent=float(100*(1-minimum.annual_heating_energy_gj/baseline.annual_heating_energy_gj)),
        minimum_heating_additional_warm_days=int(baseline.days_below_24_c-minimum.days_below_24_c),
        largest_additional_warm_days=int(baseline.days_below_24_c-worst.days_below_24_c),
        first_id_with_largest_warm_penalty=int(worst.simulation_id),
        ids_with_largest_warm_penalty=part.loc[part.days_below_24_c==worst.days_below_24_c,'simulation_id'].astype(int).tolist()))
late=selections[(selections.horizon==2100)&(selections.profile=='days_below_24_priority')].iloc[0]
physical_case={k:(v.item() if isinstance(v,np.generic) else v) for k,v in late.to_dict().items()}
physical_case['baseline_D24']=321
physical_case['change_D24']=int(late.days_below_24_c-321)

margins.to_csv(OUT/'exact_score_margins.csv',index=False)
pd.DataFrame(margin_summary).to_csv(OUT/'margin_summary.csv',index=False)
pd.DataFrame(tails).to_csv(OUT/'regret_tails.csv',index=False)
pd.DataFrame(associations).to_csv(OUT/'margin_identity_association.csv',index=False)
(OUT/'worst_exact_index_within_fold.json').write_text(json.dumps(winfo,indent=2)+'\n')
(OUT/'physical_reference_checks.json').write_text(json.dumps({'minimum_heating_by_weather':physical,
                    'late_century_D24_priority_facade_roof':physical_case},indent=2)+'\n')

# Tables suitable for inclusion in S1; quantities use the same conventions as CSVs.
model_names={'shared_mtl_nn':'Equally weighted joint network',
             'shared_mtl_nn_mgda':'Gradient-balanced joint network',
             'independent_stl_nn':'Single-task networks','random_forest':'Random forest',
             'gradient_boosting':'Gradient-boosted trees'}
model_caption=('Decision fidelity within each held-out fold. Every comparison uses the same '
    '125 candidates for exact and predicted decisions. Each formulation and task design has '
    '45 repeat--fold--weather evaluations and 225 named-profile comparisons. F1 and sampled-weight '
    'agreement are means with sample standard deviations over the 45 evaluations; these '
    'evaluations share data. Nondominance and regret refer to the exact front of the 125 candidates. '
    'The nondominated percentage and median regret include mismatches only.')
model_tex=[r'\begin{table}[htbp]',r'\centering',r'\small',
 r'\caption{'+model_caption+'}',r'\label{tab:s1-heldout-fidelity}',
 r'\setlength{\tabcolsep}{4pt}',r'\resizebox{\textwidth}{!}{%',
 r'\begin{tabular}{lrrrrrr}',r'\toprule',
 r'Formulation & F1 & \shortstack{Profile matches\\$n/225$ (\%)} & \shortstack{Weight\\agreement (\%)} & \shortstack{Nondominated\\mismatches (\%)} & \shortstack{Median\\mismatch regret} & \shortstack{Maximum\\regret} \\',r'\midrule']
ordered_summary=sorted(summary,key=lambda r:(list(DESIGNS).index(r['design']),MODELS.index(r['model'])))
short_model_names={'shared_mtl_nn':'Equal joint network','shared_mtl_nn_mgda':'Gradient-balanced joint',
                   'independent_stl_nn':'Single-task networks','random_forest':'Random forest',
                   'gradient_boosting':'Gradient-boosted trees'}
table_design=None
for r in ordered_summary:
    if r['design']!=table_design:
        if table_design is not None:model_tex.append(r'\midrule')
        model_tex.append(r'\multicolumn{7}{l}{\textit{'+r['design']+r' design}} \\')
        table_design=r['design']
    model_tex.append(f"{short_model_names[r['model']]} & "
        f"${r['f1_mean']:.3f} \\pm {r['f1_sd']:.3f}$ & "
        f"{r['profile_matches']}/{r['profile_count']} ({100*r['profile_agreement']:.1f}\\%) & "
        f"${100*r['weight_agreement_mean']:.1f} \\pm {100*r['weight_agreement_sd']:.1f}$ & "
        f"{100*r['mismatch_subset_nondominated_fraction']:.1f} & "
        f"{r['mismatch_median_regret']:.4f} & {r['max_regret']:.4f} \\\\")
model_tex += [r'\bottomrule',r'\end{tabular}}',r'\end{table}',r'']
tex=[r'\begin{table}[htbp]',r'\centering',r'\small',
 r'\caption{Exact weighted-sum winner-to-runner-up margins. Each candidate set is normalised over its own exact Pareto front. Full-grid entries count 15 weather--profile decisions; held-out entries count 225 repeat--fold--weather--profile decisions.}',
 r'\label{tab:s1-candidate-margins}',r'\setlength{\tabcolsep}{4pt}',r'\begin{tabular}{llrrr}',r'\toprule',
 r'Candidate set & Profile & Decisions & Margin $<0.004$ (\%) & Median margin \\',r'\midrule']
names={'all_profiles':'All profiles','balanced':'Equal weight','energy_priority':'Heating priority',
       'cost_priority':'Cost priority','carbon_priority':'GWP priority','days_below_24_priority':'Warm-side priority'}
scope_names={'full625':'Full grid (625)','within125':'Held-out (125)'}
for r in margin_summary:
    tex.append(f"{scope_names[r['scope']]} & {names[r['profile']]} & {r['n']} & {100*r['fraction_front_margin_below_004']:.1f} & {r['median_front_margin']:.5f} \\\\")
tex += [r'\bottomrule',r'\end{tabular}',r'\end{table}',r'',r'\begin{table}[htbp]',r'\centering',r'\small',
 r"\caption{Regret tails pooled over the five main formulations. Regret uses each candidate set's exact-front normalisation. The 95th percentile uses linear interpolation. Mismatch rows exclude exact package agreements. Evaluations share data and profiles.}",
 r'\label{tab:s1-candidate-regret-tails}',r'\setlength{\tabcolsep}{4pt}',r'\resizebox{\textwidth}{!}{%',r'\begin{tabular}{lllrrrrr}',r'\toprule',
 r'Candidate set & Design & Population & $n$ & Median & 95th pct. & Maximum & $R>0.05$ (\%) \\',r'\midrule']
for r in tails:
    if r['model']!='all_models':continue
    pop='All' if r['population']=='all_profile_comparisons' else 'Mismatches'
    tex.append(f"{scope_names[r['scope']]} & {r['design']} & {pop} & {r['n']} & {r['median_regret']:.4f} & {r['p95_regret']:.4f} & {r['max_regret']:.4f} & {100*r['fraction_regret_above_005']:.1f} \\\\")
weather_names={2020:'present',2050:'mid-century',2100:'late-century'}
worst_caption=(f"Physical outcomes and weighted contributions for the largest held-out-subset regret under the exact-index design. The {model_names[winfo['model']].lower()} (repeat seed {winfo['repeat_seed']}, fold {winfo['outer_fold']}) selects package {winfo['predicted_id']} instead of package {winfo['exact_id']} under {weather_names[winfo['horizon']]} weather and the {names[winfo['profile']].lower()} profile. Positive contributions increase regret; index savings partially offset the heating and warm-side penalties.")
tex += [r'\bottomrule',r'\end{tabular}}',r'\end{table}',r'',r'\begin{table}[htbp]',r'\centering',r'\small',
 r'\caption{'+worst_caption+'}',
 r'\label{tab:s1-worst-heldout-decision}',r'\begin{tabular}{lrrr}',r'\toprule',
 f"Quantity & Exact selection ({winfo['exact_id']}) & Surrogate selection ({winfo['predicted_id']}) & Contribution to regret \\\\",r'\midrule']
for i,name in enumerate([r'$E_H$ (GJ/year)',r'$I_C$ (index points)',r'$I_G$ (index points)',r'$D_{24}$ (days)']):
    ev=winfo['exact_selection_true_E_C_G_D'][i];pv=winfo['surrogate_selection_true_E_C_G_D'][i]
    tex.append(f"{name} & {ev:.3f} & {pv:.3f} & {contribution[i]:+.5f} \\\\")
tex += [r'\midrule',f"Total regret & & & {winfo['regret']:.5f} \\\\",r'\bottomrule',r'\end{tabular}',r'\end{table}',r'']
(OUT/'s1_diagnostic_tables.tex').write_text('\n'.join(model_tex+tex))

md=['## Table S1.21. Decision fidelity within held-out folds','',
    model_caption.replace('repeat--fold--weather','repeat–fold–weather'),'',
    'Each formulation and task design has 3 repeat seeds × 5 held-out folds × 3 weather cases = '
    '45 candidate-set evaluations, with 5 profiles each (225 profile comparisons). Sampled-weight '
    'agreement uses the same 10,000 Dirichlet(1,1,1,1) draws (seed 42) in every evaluation. '
    'Both fronts use their own ideal/nadir normalisation; selected packages are scored with '
    'the exact-front normalisation. Raw predictions are clipped for decision analysis as in Methods. '
    'These repeated comparisons are dependent, and the reported SDs describe variation across '
    'the 45 evaluations rather than uncertainty from independent deployment trials.','',
    '| Design | Formulation | Pareto F1 (mean ± SD) | Profile matches / 225 (%) | Weight agreement, % (mean ± SD) | Mismatches on subset exact front (%) | Median mismatch regret | Maximum regret |',
    '|---|---|---:|---:|---:|---:|---:|---:|']
for r in ordered_summary:
    md.append(f"| {r['design']} | {model_names[r['model']]} | "
        f"{r['f1_mean']:.3f} ± {r['f1_sd']:.3f} | {r['profile_matches']}/{r['profile_count']} "
        f"({100*r['profile_agreement']:.1f}) | {100*r['weight_agreement_mean']:.1f} ± "
        f"{100*r['weight_agreement_sd']:.1f} | {100*r['mismatch_subset_nondominated_fraction']:.1f} | "
        f"{r['mismatch_median_regret']:.4f} | {r['max_regret']:.4f} |")
exact_aggregate=next(r for r in agg if r['design']=='exact-index')
md += ['',f"Pooling the five exact-index formulations, {exact_aggregate['mismatch_subset_nondominated']} "
    f"of {exact_aggregate['mismatches']} mismatches "
    f"({100*exact_aggregate['mismatch_subset_nondominated_fraction']:.1f}%) remain nondominated "
    f"within their 125-candidate set; their median regret is {exact_aggregate['mismatch_median_regret']:.4f}. "
    f"Only {100*exact_aggregate['mismatch_fullgrid_nondominated_fraction']:.1f}% are also on the "
    '625-package exact front. The subset and full-grid references therefore should not be interchanged.',
    '', '## Table S1.22. Exact winner-to-runner-up score margins','',
    'Margins are the second-lowest minus the lowest weighted score among exact-front members, '
    'using each candidate set’s own exact-front ideal/nadir. The full-grid calculation counts '
    '15 weather–profile decisions once each; the held-out calculation counts 225 '
    'repeat–fold–weather–profile decisions once each. Values are not duplicated across '
    'formulations or task designs.','',
    '| Candidate set | Profile | Decisions | Margin < 0.004 (%) | Median margin |',
    '|---|---|---:|---:|---:|']
for r in margin_summary:
    md.append(f"| {scope_names[r['scope']]} | {names[r['profile']]} | {r['n']} | "
              f"{100*r['fraction_front_margin_below_004']:.1f} | {r['median_front_margin']:.5f} |")
md += ['', 'The full-grid median is 0.001705 (0.0017 rounded), and the held-out-subset median '
    'is 0.006630. Among exact-index comparisons within held-out sets, identity agreement is '
    '37.4% when the exact margin is below 0.004 and 85.0% otherwise. This is a descriptive '
    'association: reducing the candidate count also changes the front and its normalisation, '
    'so the comparison does not isolate a causal effect of score margins or cross-fitting. '
    'The diagnostic CSV also reports the runner-up among all candidates under the same '
    'exact-front scaling; this differs from the front-only runner-up in one of the '
    '225 held-out decisions, without changing its classification at 0.004.',
    '', '## Table S1.23. Regret tails for full-grid and held-out decisions','',
    'Values pool the five main formulations. All-profile rows include exact package agreements '
    '(zero regret); mismatch rows exclude them. Each formulation contributes 45 full-grid '
    'profile comparisons (3 repeats × 3 weather cases × 5 profiles) or 225 held-out profile '
    'comparisons (3 repeats × 5 folds × 3 weather cases × 5 profiles). The 95th percentile '
    'uses linear interpolation. Regret uses the exact-front scaling of the corresponding '
    '625- or 125-candidate set, so magnitudes across candidate-set sizes use different anchors.','',
    '| Candidate set | Design | Population | Comparisons | Median | 95th percentile | Maximum | Regret > 0.05, count (%) |',
    '|---|---|---|---:|---:|---:|---:|---:|']
for r in tails:
    if r['model']!='all_models':continue
    pop='All profiles' if r['population']=='all_profile_comparisons' else 'Mismatches'
    md.append(f"| {scope_names[r['scope']]} | {r['design']} | {pop} | {r['n']} | "
        f"{r['median_regret']:.4f} | {r['p95_regret']:.4f} | {r['max_regret']:.4f} | "
        f"{r['count_regret_above_005']} ({100*r['fraction_regret_above_005']:.1f}) |")
md += ['', '## Table S1.24. Physical outcomes of the largest exact-index held-out regret','',
    worst_caption, '',
    f"| Quantity | Exact selection ({winfo['exact_id']}) | Surrogate selection ({winfo['predicted_id']}) | Weighted contribution to regret |",
    '|---|---:|---:|---:|']
for i,name in enumerate(['Annual heating, $E_H$ (GJ/year)','Cost index, $I_C$ (points)',
                         'GWP index, $I_G$ (points)','Warm-side count, $D_{24}$ (days)']):
    ev=winfo['exact_selection_true_E_C_G_D'][i];pv=winfo['surrogate_selection_true_E_C_G_D'][i]
    md.append(f'| {name} | {ev:.3f} | {pv:.3f} | {contribution[i]:+.5f} |')
md += [f"| Total regret | | | {winfo['regret']:.5f} |", '',
    'The unrenovated package selected by the surrogate uses 14.86 GJ/year more heating '
    'and has four fewer days below 24 °C than the exact selection. The exact subset front '
    'spans six $D_{24}$ days, giving the warm-side contribution $0.7 × 4/6 = 0.46667$. '
    'Cost and GWP index savings partly offset that penalty. Both selected packages are '
    'nondominated within the subset and within the full grid. Raw $D_{24}$ predictions '
    'for the exact and surrogate selections are 365.68 and 371.77 days, respectively, '
    'and both enter the decision analysis at the clipped value of 365 days.','',
    'The retained exact-output checks in `physical_reference_checks.json` also confirm that '
    'the late-century façade/roof warm-side-priority package (ID 15) cuts heating by '
    '30.02% and gains five $D_{24}$ days relative to the unrenovated package. Among '
    'the minimum-heating selections across the three weather cases, the late-century '
    'selection has the largest warm-side penalty (57 additional warm days). Across all '
    '625 late-century packages, the largest penalty is 59 days (IDs 580, 604 and 605).','',
    'Tables S1.21–S1.24 are generated from retained outputs by '
    '`supplementary_analysis/within_fold_selection_check.py`. Machine-readable tables, '
    'case details, input SHA256 fingerprints and LaTeX tables are in '
    '`supplementary_analysis/decision_fidelity_diagnostics/`. No model fitting or '
    'EnergyPlus execution is performed.','']
(OUT/'s1_diagnostic_tables.md').write_text('\n'.join(md))
report={'margins':margin_summary,'regret_tails_pooled':[r for r in tails if r['model']=='all_models'],
        'margin_identity_association':associations,'worst_within_exact_index':winfo,
        'physical_checks':physical,'late_facade_roof_case':physical_case,
        'limitations':['Within-fold evaluation changes both model composition and candidate count.',
          'Margin/identity association is descriptive and does not identify the cause of disagreements.',
          'Subset nondominance and regret refer to 125 candidates, not the 625-package global reference.',
          'Repeated partitions, weather cases, profiles and weight vectors yield dependent comparisons.']}
for extra_path in [all_config_path,S1/'exhaustive_analysis/mcdm_weighted_sum_selections.csv',
                   *[S1/directory/'weighted_profile_validation.csv' for directory in DESIGNS.values()]]:
    fingerprints[str(extra_path.relative_to(ROOT))]=hashlib.sha256(extra_path.read_bytes()).hexdigest()
report['input_sha256']=fingerprints
(OUT/'extended_verification.json').write_text(json.dumps(report,indent=2)+'\n')
(OUT/'README.md').write_text('''# Decision fidelity diagnostics

Generated from retained out-of-fold predictions and exact enumeration outputs.
No model is fitted and no physical simulation is run. Inputs remain read-only.

The script requires Python, NumPy and pandas. Install it in the S1 package's
`supplementary_analysis` directory to use its package-relative defaults, or pass
`--s1-root` and `--output-dir` explicitly.

- `summary.csv`: 45 held-out-fold/weather evaluations and 225 profile comparisons
  per model/task design. Weight agreement uses the same 10,000 Dirichlet(1,1,1,1)
  draws (seed 42) for every evaluation.
- `group_metrics.csv` and `profile_decisions.csv`: underlying 125-package results.
- `exact_score_margins.csv`: 15 full-grid and 225 held-out-subset exact decisions,
  counted once each rather than duplicated over models or task designs. The main
  margin compares the best two exact-front members. An additional column compares
  the best two candidates anywhere in the candidate set under the same scaling.
- `margin_summary.csv`: summaries of the exact-front margins.
- `margin_identity_association.csv`: descriptive association between exact score
  margin and surrogate identity agreement. Shared data and profiles limit causal
  interpretation.
- `regret_tails.csv`: median, linearly interpolated 95th percentile, maximum and
  share strictly above 0.05, separately for all comparisons and mismatches only.
  Full-grid and held-out-subset regret use their respective exact-front scaling.
- `worst_exact_index_within_fold.json`: physical outcomes, raw predictions and
  additive weighted contributions for the largest exact-index subset regret.
- `physical_reference_checks.json`: retained simulation checks of the headline
  façade/roof comparison and minimum-heating packages across weather cases.
- `s1_diagnostic_tables.md` and `.tex`: Tables S1.21–S1.24, including model-level
  held-out-fold fidelity. LaTeX requires booktabs and graphicx.

Clipping follows Methods: D24 to [0,365], predicted cost/GWP to nonnegative values.
Each exact and reconstructed front uses its own ideal/nadir. Named profiles break
ties within 1e-12 by lowest simulation ID; sampled weights use exact argmin over
ID-sorted candidates. A lower bound or zero-width objective is handled as in S1.
Nondominance and regret in subset evaluations concern those 125 candidates.
Shrinking the candidate set also changes the decision problem; differences from
the 625-package composite do not isolate an effect of cross-fitting alone.
Exact-front winner-to-runner-up margins have medians 0.001705 (full grid) and
0.006630 (held-out subsets). Margin/identity association is descriptive, not causal.
''')
print(json.dumps(report,indent=2))
