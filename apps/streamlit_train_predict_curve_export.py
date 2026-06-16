# ==============================================================
# app_research_final_synthetic_full.py
# MCI→AD Synthetic ΔMRI Research App (Final Full Version)
# ==============================================================

import os, re, math, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (roc_curve, auc, precision_recall_curve, roc_auc_score,
                             average_precision_score, brier_score_loss)
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

RANDOM_STATE = 42
sns.set_style("whitegrid")
st.set_page_config(page_title="ΔMRI Synthetic Research App", layout="wide")

# ==============================================================
# Core Utilities
# ==============================================================

def ensure_X(df, feats):
    return df.reindex(columns=feats).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

def build_pipeline(name):
    steps=[("imputer",SimpleImputer(strategy="constant",fill_value=0.0))]
    if name in ["Logistic","ANN"]:
        steps.append(("scaler",StandardScaler(with_mean=True)))
    if name=="Logistic":
        clf=LogisticRegression(max_iter=2000,random_state=RANDOM_STATE,class_weight="balanced")
    elif name=="ANN":
        clf=MLPClassifier(hidden_layer_sizes=(64,32),max_iter=500,random_state=RANDOM_STATE)
    else:
        clf=XGBClassifier(n_estimators=250,max_depth=  4,learning_rate=0.05,
                          subsample=0.9,colsample_bytree=0.9,use_label_encoder=False,
                          eval_metric="logloss",random_state=RANDOM_STATE)
    steps.append(("clf",clf))
    return Pipeline(steps)

def safe_calibrated(pipe,X,y,method="sigmoid",inner_cv=3):
    if len(np.unique(y))<2: return pipe.fit(X,y),False
    try:
        cal=CalibratedClassifierCV(estimator=pipe,method=method,cv=inner_cv)
        cal.fit(X,y);return cal,True
    except: return pipe.fit(X,y),False

def safe_predict_proba(model,X):
    X=np.asarray(X,float)
    try:
        p=model.predict_proba(X)
        pos=list(model.classes_).index(1)
        return p[:,pos]
    except Exception:
        s=model.decision_function(X)
        s=(s-s.min())/(s.max()-s.min()+1e-9)
        return s

def metrics_all(y_true,y_prob):
    ok=~np.isnan(y_prob);y,p=y_true[ok],y_prob[ok]
    if len(np.unique(y))<2: return dict.fromkeys(["roc_auc","pr_auc","brier","cal_slope","cal_intercept"],np.nan)
    rocA=roc_auc_score(y,p);ap=average_precision_score(y,p);brier=brier_score_loss(y,p)
    eps=1e-6;z=np.clip(p,eps,1-eps);x=np.log(z/(1-z))
    X=np.vstack([np.ones_like(x),x]).T;beta=np.linalg.lstsq(X,y,rcond=None)[0]
    return dict(roc_auc=float(rocA),pr_auc=float(ap),brier=float(brier),
                cal_intercept=float(beta[0]),cal_slope=float(beta[1]))

def group_bootstrap_metrics(df,y,p,groups,B=500,seed=42):
    rng=np.random.default_rng(seed);ids=pd.Series(groups).astype(str).unique();mets=[]
    for _ in range(B):
        samp=rng.choice(ids,len(ids),replace=True)
        mask=pd.Series(groups).isin(samp).values
        mets.append(metrics_all(y[mask],p[mask]))
    out={}
    for k in mets[0].keys():
        arr=np.array([m[k] for m in mets if not np.isnan(m[k])])
        if len(arr)==0: out[k]=(np.nan,(np.nan,np.nan))
        else:
            lo,hi=np.percentile(arr,[2.5,97.5])
            out[k]=(float(np.nanmean(arr)),(float(lo),float(hi)))
    return out

def decision_curve(y,p,thr=np.linspace(0.01,0.99,99)):
    y,p=np.asarray(y),np.asarray(p);N=len(y)
    return pd.DataFrame([(t,((p>=t)&(y==1)).sum()/N-((p>=t)&(y==0)).sum()/N*(t/(1-t)))
                         for t in thr],columns=["threshold","net_benefit"])


def infer_window_from_name(obj, default="3-18"):
    name = getattr(obj, "name", "") if obj is not None else ""
    m = re.search(r'(\d+)-(\d+)', str(name))
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    return default

def parse_window(window_str):
    m = re.match(r'^(\d+)-(\d+)$', str(window_str))
    if not m:
        raise ValueError(f"Invalid window string: {window_str}")
    lo, hi = int(m.group(1)), int(m.group(2))
    if lo >= hi:
        raise ValueError(f"Invalid window order: {window_str}")
    return lo, hi

def save_curve_bundle(out_dir, stem, ids, y_true, p_snapshot, p_snapshot_change, dataset="synthetic",
                      calibration="sigmoid", window="3-18", seed=42, model="Logistic"):
    os.makedirs(out_dir, exist_ok=True)

    pred_df = pd.DataFrame({
        "SubjectID": pd.Series(ids).astype(str).values,
        "y_true": np.asarray(y_true).astype(int),
        "p_snapshot": np.asarray(p_snapshot, dtype=float),
        "p_snapshot_change": np.asarray(p_snapshot_change, dtype=float),
        "dataset": dataset,
        "window": window,
        "calibration": calibration,
        "seed": seed,
        "model": model,
    }).sort_values("SubjectID").reset_index(drop=True)

    pred_path = os.path.join(out_dir, f"{stem}_predictions.csv")
    pred_df.to_csv(pred_path, index=False)

    # ROC / PR / calibration / DCA points for direct plotting in ggplot2
    roc_rows = []
    pr_rows = []
    cal_rows = []
    dca_rows = []

    for tag, col in [("Snapshot only", "p_snapshot"), ("Snapshot+Change", "p_snapshot_change")]:
        y = pred_df["y_true"].to_numpy()
        p = pred_df[col].to_numpy()

        fpr, tpr, _ = roc_curve(y, p)
        roc_rows.append(pd.DataFrame({"curve_set": tag, "fpr": fpr, "tpr": tpr}))

        prec, rec, _ = precision_recall_curve(y, p)
        pr_rows.append(pd.DataFrame({"curve_set": tag, "recall": rec, "precision": prec}))

        tmp = pd.DataFrame({"y": y, "p": p}).sort_values("p").reset_index(drop=True)
        tmp["bin"] = pd.qcut(tmp.index, q=min(10, len(tmp)), labels=False, duplicates="drop")
        cal = tmp.groupby("bin", as_index=False).agg(
            mean_pred=("p", "mean"),
            obs_rate=("y", "mean"),
            n=("y", "size")
        )
        cal["curve_set"] = tag
        cal_rows.append(cal)

        dca = decision_curve(y, p, thr=np.linspace(0.05, 0.30, 51))
        dca["curve_set"] = tag
        dca_rows.append(dca)

    roc_df = pd.concat(roc_rows, ignore_index=True)
    pr_df = pd.concat(pr_rows, ignore_index=True)
    cal_df = pd.concat(cal_rows, ignore_index=True)
    dca_df = pd.concat(dca_rows, ignore_index=True)

    roc_df.to_csv(os.path.join(out_dir, f"{stem}_roc_points.csv"), index=False)
    pr_df.to_csv(os.path.join(out_dir, f"{stem}_pr_points.csv"), index=False)
    cal_df.to_csv(os.path.join(out_dir, f"{stem}_calibration_points.csv"), index=False)
    dca_df.to_csv(os.path.join(out_dir, f"{stem}_dca_points.csv"), index=False)

    return {
        "predictions": pred_path,
        "roc_points": os.path.join(out_dir, f"{stem}_roc_points.csv"),
        "pr_points": os.path.join(out_dir, f"{stem}_pr_points.csv"),
        "calibration_points": os.path.join(out_dir, f"{stem}_calibration_points.csv"),
        "dca_points": os.path.join(out_dir, f"{stem}_dca_points.csv"),
    }

# ----------------- Correct paired DeLong ---------------

import math
import numpy as np

def _compute_midrank(x):
    x = np.asarray(x)
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(N, dtype=float)
    out[J] = T
    return out

def _fast_delong(predictions_sorted_transposed, label_1_count):
    """
    predictions_sorted_transposed: shape (n_classifiers, n_examples)
    examples must be sorted so that all positives come first
    """
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    k = predictions_sorted_transposed.shape[0]

    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]

    tx = np.empty((k, m), dtype=float)
    ty = np.empty((k, n), dtype=float)
    tz = np.empty((k, m + n), dtype=float)

    for r in range(k):
        tx[r, :] = _compute_midrank(positive_examples[r, :])
        ty[r, :] = _compute_midrank(negative_examples[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / (2.0 * n)

    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m

    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n

    return aucs, delongcov

def delong_2sample_test(y_true, p1, p2):
    """
    Paired DeLong test for two correlated ROC AUCs
    """
    y_true = np.asarray(y_true).astype(int)
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)

    mask = ~(np.isnan(y_true) | np.isnan(p1) | np.isnan(p2))
    y_true = y_true[mask]
    p1 = p1[mask]
    p2 = p2[mask]

    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan, np.nan

    order = np.argsort(-y_true)   # positives first
    y_true = y_true[order]
    p1 = p1[order]
    p2 = p2[order]

    label_1_count = int(np.sum(y_true))
    preds = np.vstack([p1, p2])

    aucs, cov = _fast_delong(preds, label_1_count)
    diff = aucs[1] - aucs[0]

    if np.ndim(cov) == 0:
        var = float(cov)
    else:
        var = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])

    if var <= 0:
        return float(diff), np.nan, np.nan

    z = diff / np.sqrt(var)
    p = math.erfc(abs(z) / np.sqrt(2.0))   # two-sided p-value

    return float(diff), float(z), float(p)

# ==============================================================
# Synthetic merger
# ==============================================================

def build_single_table_synthetic(snap_file, delt_file, window="3-18"):
    snap = pd.read_csv(snap_file)
    delt = pd.read_csv(delt_file)

    snap.columns = [c.lower() for c in snap.columns]
    delt.columns = [c.lower() for c in delt.columns]

    for df in [snap, delt]:
        df["subjectid"] = df["subjectid"].astype(str)
    # snapshot also must be one row per subject
    snap = snap.drop_duplicates(subset=["subjectid"]).copy()

    if "interval_months" not in delt.columns:
        raise ValueError("mridelta.csv must contain interval_months")

    delt["interval_months"] = (
        pd.to_numeric(delt["interval_months"], errors="coerce")
        .replace(0, np.nan)
        .fillna(0.1)
    )
    # Window handling: choose a single row per subject within the requested window
    lo, hi = parse_window(window)
    delt = delt[(delt["interval_months"] >= lo) & (delt["interval_months"] <= hi)].copy()
    if len(delt) == 0:
        raise ValueError(f"No rows remained after filtering mridelta.csv to window {window}")

    midpoint = 0.5 * (lo + hi)
    delt["dist_to_mid"] = (delt["interval_months"] - midpoint).abs()
    delt = (
        delt.sort_values(["subjectid", "dist_to_mid", "interval_months"])
        .groupby("subjectid", as_index=False)
        .first()
    )

    # hippocampus
    delt["st29sv_relchange_emp"] = (
        ((delt["st29sv_follow"] - delt["st29sv_base"]) / delt["st29sv_base"])
        / (delt["interval_months"] / 12)
    )

    # ventricle
    delt["st37sv_relchange_emp"] = (
        ((delt["st37sv_follow"] - delt["st37sv_base"]) / delt["st37sv_base"])
        / (delt["interval_months"] / 12)
    )

    # entorhinal candidate (provisional: st149sv)
    if {"st149sv_base", "st149sv_follow"}.issubset(delt.columns):
        delt["st149sv_relchange_emp"] = (
            ((delt["st149sv_follow"] - delt["st149sv_base"]) / delt["st149sv_base"])
            / (delt["interval_months"] / 12)
        )
    else:
        delt["st149sv_relchange_emp"] = np.nan
        delt["st149sv_base"] = np.nan
        delt["st149sv_follow"] = np.nan

    # missing flags
    for c in ["st29sv_relchange_emp", "st37sv_relchange_emp", "st149sv_relchange_emp"]:
        delt[c + "_isna"] = delt[c].isna().astype(int)

    keep = [
        "subjectid",
        "interval_months",

        "st29sv_base", "st29sv_follow",
        "st29sv_relchange_emp", "st29sv_relchange_emp_isna",

        "st149sv_base", "st149sv_follow",
        "st149sv_relchange_emp", "st149sv_relchange_emp_isna",

        "st37sv_base", "st37sv_follow",
        "st37sv_relchange_emp", "st37sv_relchange_emp_isna",
    ]

    for extra in ["converter", "group_base", "group_follow"]:
        if extra in delt.columns:
            keep.append(extra)

    ds = snap.merge(delt[keep], on="subjectid", how="inner").rename(columns={
        "st29sv_base": "hippo_base",
        "st149sv_base": "entorh_base",
        "st37sv_base": "vent_base",

        "st29sv_relchange_emp": "hippo_relchange_emp",
        "st149sv_relchange_emp": "entorh_relchange_emp",
        "st37sv_relchange_emp": "vent_relchange_emp",

        "st29sv_relchange_emp_isna": "hippo_relchange_emp_isna",
        "st149sv_relchange_emp_isna": "entorh_relchange_emp_isna",
        "st37sv_relchange_emp_isna": "vent_relchange_emp_isna",
    })

    # label
    if "converter" in ds.columns:
        y = ds["converter"].astype(int).values
    elif "group_follow" in ds.columns:
        y = ds["group_follow"].astype(str).str.lower().str.contains("ad").astype(int).values
    elif "group_base" in ds.columns:
        y = ds["group_base"].astype(str).str.lower().str.contains("progress").astype(int).values
    else:
        raise ValueError("No converter or group labels found.")

    feats_base = ["hippo_base", "entorh_base", "vent_base"]

    feats_delta = [
        "hippo_relchange_emp",
        "entorh_relchange_emp",
        "vent_relchange_emp",
        "hippo_relchange_emp_isna",
        "entorh_relchange_emp_isna",
        "vent_relchange_emp_isna",
    ]

    feats_all = feats_base + feats_delta

    ds["_has_delta_any"] = ds[
        ["hippo_relchange_emp", "entorh_relchange_emp", "vent_relchange_emp"]
    ].notna().any(axis=1)

    ds["SubjectID"] = ds["subjectid"].astype(str)

    return ds, y, feats_base, feats_all

# ==============================================================
# Streamlit App
# ==============================================================

st.title("🧠 MCI→AD — Synthetic ΔMRI Full Research App")

tab_train,tab_oof,tab_pred=st.tabs(["Train & Compare","OOF / CV","Predict"])

# ---------------- Train ----------------
with tab_train:
    st.header("📊 Train & Compare (Hold-out)")
    snap=st.file_uploader("Upload synthetic_snapshot.csv",type="csv")
    delt=st.file_uploader("Upload synthetic_mridelta.csv",type="csv")
    inferred_window = infer_window_from_name(delt, default="3-18")
    window = st.selectbox("Analysis window", ["3-18","6-24","12-30"], index=["3-18","6-24","12-30"].index(inferred_window) if inferred_window in ["3-18","6-24","12-30"] else 0)
    method=st.selectbox("Calibration method",["sigmoid","isotonic"],index=0)
    test_size=st.slider("Hold-out test size",0.2,0.4,0.3,0.05)
    B=st.number_input("Bootstrap B",value=500,min_value=100,step=100)
    export_curve_outputs = st.checkbox("Save curve-output CSVs for ggplot figures", value=True)
    if snap and delt:
        ds,y,fb,fa=build_single_table_synthetic(snap,delt,window=window)
        groups=ds["SubjectID"].astype(str).values
        gss=GroupShuffleSplit(n_splits=1,train_size=1-test_size,random_state=RANDOM_STATE)
        tr,te=next(gss.split(ds,y,groups))
        sets=[("Snapshot only",fb),("Snapshot+Change",fa)]
        model_names=["Logistic"]
        res=[];pred_rows=[]
        st.session_state["MODELS"] = {}
        for tag,feats in sets:
            Xtr_df,Xte_df=ds.iloc[tr],ds.iloc[te]
            if tag=="Snapshot+Change":
                Xtr_df=Xtr_df[Xtr_df["_has_delta_any"]];Xte_df=Xte_df[Xte_df["_has_delta_any"]]
            Xtr,Xte=ensure_X(Xtr_df,feats),ensure_X(Xte_df,feats)
            ytr,yte=y[Xtr_df.index],y[Xte_df.index];g_te=groups[Xte_df.index]
            for m in model_names:
                pipe=build_pipeline(m);cal,_=safe_calibrated(pipe,Xtr,ytr,method=method,inner_cv=3)
                p=safe_predict_proba(cal,Xte)
                metr=metrics_all(yte,p);boot=group_bootstrap_metrics(Xte_df,yte,p,g_te,B=B)
                res.append(dict(Set=tag,Model=m,ROC_AUC=metr["roc_auc"],ROC_AUC_CI=boot["roc_auc"][1],
                                PR_AP=metr["pr_auc"],PR_AP_CI=boot["pr_auc"][1],Brier=metr["brier"],
                                Cal_slope=metr["cal_slope"],Cal_intercept=metr["cal_intercept"],n_test=len(yte)))
                pred_rows.append(pd.DataFrame({"SubjectID":Xte_df["SubjectID"],"y_true":yte,"p_pred":p,
                                               "set":tag,"model":m}))
                st.session_state["MODELS"][f"{tag}_{m}"] = (cal, feats, tag, m)
        res_df = pd.DataFrame(res)
        st.dataframe(res_df,use_container_width=True)
        # DeLong
        df_pred=pd.concat(pred_rows,ignore_index=True)
        st.subheader("ΔAUC (DeLong paired)")
        out=[]
        for m in model_names:
            b=df_pred[(df_pred["set"]=="Snapshot only")&(df_pred["model"]==m)]
            c=df_pred[(df_pred["set"]=="Snapshot+Change")&(df_pred["model"]==m)]
            inter=set(b["SubjectID"])&set(c["SubjectID"])
            if len(inter)<10: continue
            b2=b[b["SubjectID"].isin(inter)].sort_values("SubjectID")
            c2=c[c["SubjectID"].isin(inter)].sort_values("SubjectID")
            d,z,pv=delong_2sample_test(b2["y_true"],b2["p_pred"],c2["p_pred"])
            out.append(dict(Model=m,ΔAUC=round(d,4),z=round(z,3),p_value=f"{pv:.4g}",n=len(inter)))
        if out:
            st.dataframe(pd.DataFrame(out),use_container_width=True)
        else:
            st.info("교집합 부족으로 ΔAUC 생략")

        if export_curve_outputs:
            # Export one paired prediction file for the final Logistic comparison
            b=df_pred[(df_pred["set"]=="Snapshot only")&(df_pred["model"]=="Logistic")]
            c=df_pred[(df_pred["set"]=="Snapshot+Change")&(df_pred["model"]=="Logistic")]
            inter=set(b["SubjectID"]) & set(c["SubjectID"])
            if len(inter) >= 10:
                b2=b[b["SubjectID"].isin(inter)].sort_values("SubjectID")
                c2=c[c["SubjectID"].isin(inter)].sort_values("SubjectID")
                export_dir = os.path.join(os.path.dirname(__file__), "curve_outputs")
                stem = f"synthetic_seed{RANDOM_STATE}_{window}"
                saved = save_curve_bundle(
                    out_dir=export_dir,
                    stem=stem,
                    ids=b2["SubjectID"].values,
                    y_true=b2["y_true"].values,
                    p_snapshot=b2["p_pred"].values,
                    p_snapshot_change=c2["p_pred"].values,
                    dataset="synthetic",
                    calibration=method,
                    window=window,
                    seed=RANDOM_STATE,
                    model="Logistic",
                )
                st.success("Saved curve-output CSV files for ggplot figure generation.")
                st.json(saved)
            else:
                st.warning("Curve-output CSVs were not saved because the paired test-set intersection was too small.")
# ---------------- OOF ----------------
with tab_oof:
    st.header("🧪 Group-aware Cross-Validation (OOF) Snapshot vs Δ")
    snap2=st.file_uploader("snapshot.csv",type="csv",key="snap_oof")
    delt2=st.file_uploader("mridelta.csv",type="csv",key="delt_oof")
    n_splits=st.slider("GroupKFold n_splits",3,10,5,1)
    B_oof=st.number_input("Bootstrap B (OOF)",value=500,min_value=100,step=100)
    if snap2 and delt2:
        ds,y,fb,fa=build_single_table_synthetic(snap2,delt2,window=infer_window_from_name(delt2, default="3-18"))
        groups=ds["SubjectID"].astype(str).values
        cv=GroupKFold(n_splits=n_splits)
        results=[];oof_preds={}
        for tag,feats in [("Snapshot only",fb),("Snapshot+Change",fa)]:
            Xdf=ds.copy()
            if tag=="Snapshot+Change": Xdf=Xdf[Xdf["_has_delta_any"]]
            X=ensure_X(Xdf,feats);y_sub=y[Xdf.index];g_sub=groups[Xdf.index]
            oof=np.full(len(Xdf),np.nan)
            for tr,te in cv.split(X,y_sub,g_sub):
                cal,_=safe_calibrated(build_pipeline("Logistic"),X[tr],y_sub[tr])
                oof[te]=safe_predict_proba(cal,X[te])
            metr=metrics_all(y_sub,oof);boot=group_bootstrap_metrics(Xdf,y_sub,oof,g_sub,B=B_oof)
            results.append(dict(Set=tag,ROC_AUC=metr["roc_auc"],ROC_AUC_CI=boot["roc_auc"][1],
                                PR_AP=metr["pr_auc"],PR_AP_CI=boot["pr_auc"][1],
                                Brier=metr["brier"],Brier_CI=boot["brier"][1],
                                Cal_slope=metr["cal_slope"],Cal_intercept=metr["cal_intercept"],n=len(y_sub)))
            oof_preds[tag]=pd.DataFrame({"SubjectID":Xdf["SubjectID"],"y_true":y_sub,"p_oof":oof})
        st.dataframe(pd.DataFrame(results),use_container_width=True)
        # ΔAUC on OOF
        base=oof_preds["Snapshot only"];chg=oof_preds["Snapshot+Change"]
        inter=set(base["SubjectID"])&set(chg["SubjectID"])
        if len(inter)>=10:
            b=base[base["SubjectID"].isin(inter)].sort_values("SubjectID")
            c=chg[chg["SubjectID"].isin(inter)].sort_values("SubjectID")
            d,z,pv=delong_2sample_test(b["y_true"],b["p_oof"],c["p_oof"])
            st.write({"ΔAUC (OOF)":round(d,4),"z":round(z,3),"p_value":float(f"{pv:.4g}"),"n_common":len(inter)})
        else: st.info("교집합 부족으로 ΔAUC 생략")

# ---------------- Predict ----------------
with tab_pred:
    st.header("🔮 Predict on new synthetic CSV")

    # 새 snapshot + mridelta 파일 업로드
    snap3 = st.file_uploader("Upload snapshot.csv (new)", type="csv", key="snap_pred")
    delt3 = st.file_uploader("Upload mridelta.csv (new)", type="csv", key="delt_pred")

    if snap3 and delt3:
        ds_new, _, feats_base, feats_all = build_single_table_synthetic(snap3, delt3, window=infer_window_from_name(delt3, default="3-18"))
        out = ds_new[["SubjectID"]].copy()

        st.write("✅ Data preview")
        st.dataframe(ds_new.head())

        # 현재 세션에서 학습된 모델이 있는 경우
        if "MODELS" in st.session_state and st.session_state["MODELS"]:
            st.subheader("📈 Apply trained models from this session")

            for key, (model, feats, tag, mname) in st.session_state["MODELS"].items():
                Xdf = ds_new.copy()
                if tag == "Snapshot+Change":
                    Xdf = Xdf[Xdf["_has_delta_any"]]
                X = ensure_X(Xdf, feats)
                p = safe_predict_proba(model, X)
                col = f"{tag.replace(' ','_').lower()}_{mname.lower()}"
                tmp = pd.Series(np.nan, index=out.index)
                tmp.loc[Xdf.index] = (p * 100).round(1)
                out[col] = tmp

            st.write("Predictions preview", out.head())
            st.download_button(
                "⬇️ Download predictions.csv",
                out.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
            )

            # Optional: 그룹별 분포 시각화
            if "group_follow" in ds_new.columns or "converter" in ds_new.columns:
                gcol = "group_follow" if "group_follow" in ds_new.columns else "converter"
                if gcol == "converter":
                    ds_new[gcol] = ds_new[gcol].map({0: "Stable", 1: "Converted"})
                for gname in ds_new[gcol].unique():
                    sub = out.loc[ds_new[gcol] == gname]
                    if sub.empty:
                        continue
                    fig, ax = plt.subplots(figsize=(8, 3), dpi=120)
                    for c in [c for c in out.columns if c not in ["SubjectID"]]:
                        sns.kdeplot(sub[c].dropna(), ax=ax, label=c, clip=(0, 100), cut=0)
                    ax.set_xlim(0, 100)
                    ax.legend(fontsize=7)
                    ax.set_title(f"{gname} — predicted probability distributions")
                    st.pyplot(fig)
        else:
            st.info("Train탭에서 저장한 모델이 있으면 여기에 붙여 사용")
