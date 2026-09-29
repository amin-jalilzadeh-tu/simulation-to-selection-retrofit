"""Generate manuscript Figures 2 and 3 from the retained exhaustive results.

The pair shows all package outcomes and floor-stratified component comparisons.
Run directly or through make_attainment_figure.py; S1_EXHAUSTIVE and FIGDIR
select alternate input and output directories. No review-output files are used.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, ConnectionPatch
import numpy as np
import make_main_figures as mm

OUT = None
DATA = None
WEATHER = ("2020", "2050", "2100")
LABEL = {"2020": "Present TMYx", "2050": "SSP2-4.5 / 2050", "2100": "SSP5-8.5 / 2080"}
LABELS = [LABEL[h] for h in WEATHER]
COMPS = ("floor", "windows", "facade", "roof")
NAMES = {"floor": "Ground floor", "windows": "Windows", "facade": "Façade", "roof": "Roof"}
INK, MUTED, GRID = "#27333D", "#5F6971", "#E4E8EB"
BLUE, ORANGE, GREY = "#2D7896", "#B57239", "#4F5A64"
GROUP_COLORS = {"existing": BLUE, "insulated": ORANGE, "changed": GREY}
WIDTH = SIZE = .94 * 465 / 72.27


def base_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.labelsize": 8, "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK,
        "axes.edgecolor": "#939EA6", "axes.linewidth": .6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.facecolor": "white", "axes.unicode_minus": True,
    })

def soften(ax, grid="both"):
    ax.set_axisbelow(True)
    ax.grid(axis=grid, color=GRID, linewidth=.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2.5, width=.55, color="#87939C")

def save(fig, stem):
    # Audit every text object's bounding box before export.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    clipped = []
    for text in fig.findobj(match=matplotlib.text.Text):
        if not text.get_visible() or not text.get_text():
            continue
        box = text.get_window_extent(renderer)
        if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width+1 or box.y1 > fig.bbox.height+1:
            clipped.append(text.get_text())
    assert not clipped, f"Text outside media box: {clipped}"
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png", dpi=300)
    plt.close(fig)

def package_rows(hz):
    base = mm.PKG[(hz, 1)]
    return [{"weather_code": hz, "weather": LABEL[hz], "simulation_id": sim,
             "heating_reduction_percent": 100*(base["EH"]-p["EH"])/base["EH"],
             "additional_threshold_days": base["D24"]-p["D24"],
             "floor_group": "existing" if p["floor"] == 0 else "insulated"}
            for (h, sim), p in sorted(mm.PKG.items()) if h == hz]

def paired_rows(hz, comp):
    idx = {tuple(p[c] for c in mm.COL): (sim, p)
           for (h, sim), p in mm.PKG.items() if h == hz}
    results = []
    for _, (sim0, a) in idx.items():
        if a[comp] != 0:
            continue
        key = tuple(mm.STRONG[comp] if c == comp else a[c] for c in mm.COL)
        sim1, b = idx[key]
        assert all(a[c] == b[c] for c in mm.COL if c != comp)
        results.append({"weather_code": hz, "weather": LABEL[hz], "component": comp,
                        "existing_option_simulation_id": sim0,
                        "replacement_option_simulation_id": sim1,
                        "floor_group": "changed" if comp == "floor" else
                                       ("existing" if a["floor"] == 0 else "insulated"),
                        "floor_state_before": a["floor"], "floor_state_after": b["floor"],
                        "heating_saved_gj": a["EH"]-b["EH"],
                        "additional_threshold_days": a["D24"]-b["D24"]})
    assert len(results) == 125
    for metric, reference in zip(("heating_saved_gj", "additional_threshold_days"),
                                 mm.matched_effects(hz, comp)):
        np.testing.assert_allclose(sorted(r[metric] for r in results), sorted(reference))
    return results

def write_csv(name,rows):
    folder = OUT / "retrofit_tradeoff_data"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder/name).open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

def audit():
    packages=[]; pairs=[]; summaries=[]; cases=[]
    expected=[((-4,2),(53.5,-1),(29.7,-4)),
              ((-7,39),(57.4,39),(29.7,-3)),
              ((-8,59),(59.0,57),(27.6,-4))]
    for hz,(bounds,minimum,equal) in zip(WEATHER,expected):
        rows=package_rows(hz); packages.extend(rows)
        assert len(rows)==625
        assert sum(r["floor_group"]=="existing" for r in rows)==125
        assert (min(r["additional_threshold_days"] for r in rows),
                max(r["additional_threshold_days"] for r in rows))==bounds
        sels={}
        for role,sid,values in (("minimum",mm.MINEH[hz],minimum),
                                ("equal",mm.SEL[(hz,"balanced")],equal)):
            r=next(r for r in rows if r["simulation_id"]==sid)
            assert (round(r["heating_reduction_percent"],1),r["additional_threshold_days"])==values
            sels[role]=r
        cases.append({"weather":LABEL[hz],"day_change_range":bounds,"selections":sels})
        for comp in COMPS:
            pr=paired_rows(hz,comp); pairs.extend(pr)
            for group in ["all"]+(["existing","insulated"] if comp!="floor" else []):
                subset=pr if group=="all" else [r for r in pr if r["floor_group"]==group]
                assert len(subset)=={"all":125,"existing":25,"insulated":100}[group]
                for key in ("heating_saved_gj","additional_threshold_days"):
                    s={"weather":LABEL[hz],"component":comp,"floor_group":group,
                       "metric":key,"n":len(subset)}
                    s.update(zip(("min","q1","median","q3","max"),
                                 np.percentile([r[key] for r in subset],[0,25,50,75,100]).tolist()))
                    summaries.append(s)
    write_csv("plotted_package_data.csv",packages)
    write_csv("plotted_paired_data.csv",pairs)
    write_csv("component_summary_data.csv",summaries)
    for comp,group,expected_median in (("windows","existing",-2),
                                       ("windows","insulated",14),
                                       ("facade","existing",-3),
                                       ("facade","insulated",14),
                                       ("floor","all",53)):
        summary=next(s for s in summaries if s["weather"]==LABEL["2100"]
                     and s["component"]==comp and s["floor_group"]==group
                     and s["metric"]=="additional_threshold_days")
        assert summary["median"]==expected_median
    source_hashes={name:hashlib.sha256((DATA/name).read_bytes()).hexdigest()
                   for name in ("all_configurations.csv","mcdm_weighted_sum_selections.csv","reference_solutions.csv")}
    (OUT/"retrofit_tradeoff_data"/"numerical_audit.json").write_text(json.dumps({
        "source_sha256":source_hashes,"weather_cases":cases,
        "n_packages":len(packages),"n_component_pairs":len(pairs),
        "paired_values_match_original_helper":True,"figure_width_inches":WIDTH,
        "group_sizes": {"existing_floor":25,"insulated_floor":100,"changed_floor":125},
        "interpretation":"Descriptive summaries of enumerated model cases, not confidence intervals."},indent=2)+"\n")

def style():
    base_style()
    plt.rcParams.update({"font.size":8, "axes.labelsize":8.5,
                         "axes.titlesize":8.5, "xtick.labelsize":8,
                         "ytick.labelsize":8, "legend.fontsize":8})

def floor_handles(include_changed=False):
    suffix = " (fixed)" if include_changed else ""
    result = [Line2D([],[],ls="",marker="o",ms=4.3,color=BLUE,label="Existing floor"+suffix),
              Line2D([],[],ls="",marker="s" if include_changed else "o",ms=4.3,
                     color=ORANGE,label="Insulated floor"+suffix)]
    if include_changed:
        result.append(Line2D([],[],ls="",marker="o",ms=4.3,color=GREY,label="Floor option changed"))
    return result

def package_points(ax,hz,inset=False):
    rows=package_rows(hz)
    for g in ("insulated","existing"):
        data=[r for r in rows if r["floor_group"]==g]
        ax.scatter([r["heating_reduction_percent"] for r in data],
                   [r["additional_threshold_days"] for r in data],
                   s=5.5 if inset else 7,color=GROUP_COLORS[g],alpha=.48,
                   linewidth=0,zorder=3)
    selected={r["simulation_id"]:r for r in rows}
    for sid,marker in ((mm.MINEH[hz],"^"),(mm.SEL[(hz,"balanced")],"D")):
        p=selected[sid]
        ax.scatter(p["heating_reduction_percent"],p["additional_threshold_days"],
                   marker=marker,s=29 if inset else 42,
                   facecolor=INK if marker=="^" else "white",
                   edgecolor="white" if marker=="^" else INK,
                   linewidth=.8,zorder=6)
    ax.scatter(0,0,marker="+",s=25,c=INK,lw=.85,zorder=7)
    ax.axhline(0,color="#7B858D",lw=.75,zorder=2)
    soften(ax)
    ax.set_xlim(-3,65)

def packages():
    fig=plt.figure(figsize=(SIZE,3.28))
    gs=fig.add_gridspec(1,3,left=.115,right=.985,bottom=.285,top=.90,wspace=.15)
    for i,hz in enumerate(WEATHER):
        ax=fig.add_subplot(gs[0,i])
        package_points(ax,hz)
        ax.set_ylim(-12,65)
        ax.set_xticks([0,20,40,60])
        ax.set_yticks([-10,0,20,40,60])
        ax.set_title(f"({chr(97+i)})  {LABELS[i]}",loc="left",pad=9,fontsize=8.2)
        if i:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel("Change in days ≥24 °C",labelpad=5)
        if i==0:
            zoom=ax.inset_axes([.16,.39,.79,.43])
            package_points(zoom,hz,inset=True)
            zoom.set_ylim(-5,3)
            zoom.set_xticks([0,30,60])
            zoom.set_yticks([-4,0,2])
            zoom.tick_params(labelsize=7.5,length=2,pad=2)
            zoom.set_title("Vertical zoom",fontsize=7.5,pad=4)
            zoom.spines[["top","right"]].set_visible(True)
            ax.add_patch(Rectangle((-3,-5),68,8,fill=False,edgecolor="#A7AFB5",lw=.6,
                                   linestyle=(0,(3,2)),zorder=4))
            for x in [-3,65]:
                ax.add_artist(ConnectionPatch(xyA=(x,-5),coordsA=zoom.transData,
                    xyB=(x,3),coordsB=ax.transData,color="#B9C0C5",lw=.6,
                    linestyle=(0,(3,2)),zorder=1))
    fig.text(.55,.174,"Heating reduction (%)",ha="center",fontsize=8.5)
    handles=floor_handles()
    handles += [Line2D([],[],ls="",marker="+",ms=6,color=INK,label="Baseline")]
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.55,.073),ncol=3,
               frameon=False,handletextpad=.45,columnspacing=1.5,borderaxespad=0)
    handles=[Line2D([],[],ls="",marker="^",ms=5.5,color=INK,label="Minimum heating"),
             Line2D([],[],ls="",marker="D",ms=4.8,color=INK,markerfacecolor="white",
                    label="Equal weights (4 objectives)")]
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.55,.003),ncol=2,
               frameon=False,handletextpad=.5,columnspacing=1.5,borderaxespad=0)
    save(fig,"fig_package_tradeoffs")

def distribution(ax,vals,y,g,seed):
    vals=np.array(vals)
    col=GROUP_COLORS[g]
    jitter=np.random.default_rng(seed).uniform(-.055,.055,len(vals))
    ax.scatter(vals,y+jitter,c=col,s=4,alpha=.28,lw=0,zorder=2)
    lo,q1,med,q3,hi=np.percentile(vals,[0,25,50,75,100])
    ax.plot([lo,hi],[y,y],color=col,lw=.7,alpha=.7,zorder=3)
    ax.plot([q1,q3],[y,y],color=col,lw=3.0,solid_capstyle="butt",zorder=4)
    ax.scatter(med,y,s=23,c=col,marker="s" if g=="insulated" else "o",
               edgecolors="white",linewidth=.65,zorder=5)

def components():
    fig=plt.figure(figsize=(SIZE,4.12))
    gs=fig.add_gridspec(2,3,left=.179,right=.985,bottom=.177,top=.918,
                        wspace=.20,hspace=.49)
    axes=[]
    for metric in range(2):
        row=[]
        for i,hz in enumerate(WEATHER):
            ax=fig.add_subplot(gs[metric,i])
            row.append(ax)
            soften(ax,grid="x")
            ax.spines["left"].set_visible(False)
            ax.set_ylim(-.48,3.62)
            ax.set_yticks([3,2,1,0])
            ax.set_yticklabels([NAMES[c] for c in COMPS] if i==0 else [])
            ax.tick_params(axis="y",length=0,pad=5,labelsize=8)
            ax.axhspan(2.65,3.35,color="#F3F4F5",zorder=-2)
            if metric==0:
                ax.set_title(f"({chr(97+i)})  {LABELS[i]}",loc="left",pad=10,fontsize=8.2)
                ax.set_xlim(0,7.2); ax.set_xticks([0,2,4,6])
            else:
                ax.text(0,1.035,f"({chr(100+i)})",transform=ax.transAxes,
                        fontsize=8,va="bottom")
                ax.set_xlim(-9,68); ax.set_xticks([0,20,40,60])
                ax.axvline(0,color="#7B858D",lw=.75,zorder=1)
            key="heating_saved_gj" if metric==0 else "additional_threshold_days"
            for j,c in enumerate(COMPS):
                rows=paired_rows(hz,c)
                for k,g in enumerate(("changed",) if c=="floor" else ("existing","insulated")):
                    vals=[r[key] for r in rows if r["floor_group"]==g]
                    y=3-j+(0 if g=="changed" else (.15 if g=="existing" else -.15))
                    distribution(ax,vals,y,g,9100+100*i+10*j+k)
        axes.append(row)
    for row,label in zip(axes,["Heating saved (GJ/year)","Change in days ≥24 °C"]):
        p=row[1].get_position()
        fig.text((gs.left+gs.right)/2,p.y0-.096,label,ha="center",fontsize=8.5)
    fig.legend(handles=floor_handles(include_changed=True),loc="lower center",
               bbox_to_anchor=(.535,.008),ncol=3,frameon=False,handletextpad=.4,
               columnspacing=1.0,borderaxespad=0,fontsize=8)
    save(fig,"fig_component_comparisons")

def make_figures(exhaustive_dir=None, output_dir=None):
    global OUT, DATA
    DATA = mm._results_dir("S1_EXHAUSTIVE", "exhaustive_analysis", exhaustive_dir)
    OUT = mm._output_dir(output_dir)
    mm._load_reference(DATA)
    with matplotlib.rc_context():
        # Keep the approved artwork independent of a caller's plotting style.
        plt.rcdefaults()
        style()
        audit()
        packages()
        components()
    print(f"Generated package and component figures in {OUT}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exhaustive-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    make_figures(args.exhaustive_dir, args.output_dir)


if __name__ == "__main__":
    main()
