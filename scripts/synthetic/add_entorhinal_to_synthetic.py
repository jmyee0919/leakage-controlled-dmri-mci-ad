import re
import numpy as np
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# 설정
# --------------------------------------------------
# 이 파일이 program 폴더 안에 있다고 가정
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 결과 저장 폴더
OUT_DIR = PROJECT_ROOT / "A_generated"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 라벨 추론
# --------------------------------------------------
def infer_label(df):
    if "converter" in df.columns:
        return pd.to_numeric(df["converter"], errors="coerce").fillna(0).astype(int).values
    if "group_follow" in df.columns:
        return df["group_follow"].astype(str).str.lower().str.contains("ad").astype(int).values
    if "group_base" in df.columns:
        return df["group_base"].astype(str).str.lower().str.contains("progress").astype(int).values
    raise ValueError("No converter/group label found.")


# --------------------------------------------------
# entorhinal 붙이기
# --------------------------------------------------
def make_entorh(snapshot_path, delta_path, out_snapshot_path, out_delta_path, seed=42):
    snap = pd.read_csv(snapshot_path)
    delt = pd.read_csv(delta_path)

    snap.columns = [c.lower() for c in snap.columns]
    delt.columns = [c.lower() for c in delt.columns]

    for df in [snap, delt]:
        df["subjectid"] = df["subjectid"].astype(str)

    # labels
    try:
        y = infer_label(delt)
    except Exception:
        y = infer_label(snap)

    # interval
    if "interval_months" not in delt.columns:
        raise ValueError(f"{delta_path.name} must contain interval_months")

    months = pd.to_numeric(delt["interval_months"], errors="coerce").fillna(12.0).values
    years = months / 12.0

    # y 길이 맞추기
    if len(y) != len(delt):
        tmp = snap[["subjectid"]].copy()
        tmp["converter_tmp"] = infer_label(snap)
        delt = delt.merge(tmp, on="subjectid", how="left")
        y = pd.to_numeric(delt["converter_tmp"], errors="coerce").fillna(0).astype(int).values
        delt = delt.drop(columns=["converter_tmp"])

    rng = np.random.default_rng(seed)

    # -----------------------------
    # Appendix A-inspired simple entorhinal regeneration
    # left baseline mean/sd:
    #   converted:   1400 ± 100
    #   nondemented: 1500 ± 100
    # bilateral aggregate ≈ left * 2.01
    #
    # annual rate:
    #   converted:   -0.060 ± 0.015
    #   nondemented: -0.015 ± 0.006
    # -----------------------------
    left_base = np.where(
        y == 1,
        rng.normal(1400, 100, size=len(delt)),
        rng.normal(1500, 100, size=len(delt)),
    )
    bilateral_base = left_base * 2.01

    rate = np.where(
        y == 1,
        rng.normal(-0.060, 0.015, size=len(delt)),
        rng.normal(-0.015, 0.006, size=len(delt)),
    )

    follow = bilateral_base * (1 + rate * years)
    follow = follow * (1 + rng.normal(0, 0.025, size=len(delt)))  # measurement noise

    eps = 1e-6
    relchange = ((follow - bilateral_base) / (np.abs(bilateral_base) + eps)) / years

    # delta 파일에 추가
    delt["st149sv_base"] = bilateral_base
    delt["st149sv_follow"] = follow
    delt["st149sv_delta"] = relchange

    # snapshot 파일에 subject-level baseline 추가
    ent_df = (
        delt[["subjectid", "st149sv_base"]]
        .drop_duplicates(subset=["subjectid"])
        .rename(columns={"st149sv_base": "st149sv"})
    )
    snap = snap.merge(ent_df, on="subjectid", how="left")

    # 저장
    snap.to_csv(out_snapshot_path, index=False)
    delt.to_csv(out_delta_path, index=False)

    print(f"[OK] {snapshot_path.name} + {delta_path.name}")
    print(f"     -> {out_snapshot_path.name}")
    print(f"     -> {out_delta_path.name}")


# --------------------------------------------------
# 파일 자동 찾기
# --------------------------------------------------
def find_all_pairs():
    """
    프로젝트 전체를 뒤져서
    s1(3-18).csv / m1(3-18).csv 같은 쌍을 자동으로 찾음
    """
    pattern = re.compile(r"^([sm])(\d)\(([^)]+)\)\.csv$", re.IGNORECASE)

    found = {}

    for p in PROJECT_ROOT.rglob("*.csv"):
        name = p.name

        # 이미 생성한 A_ 파일은 건너뜀
        if name.startswith("A_"):
            continue

        m = pattern.match(name)
        if not m:
            continue

        kind = m.group(1).lower()   # s or m
        seed = m.group(2)           # 1,2,3
        window = m.group(3)         # 3-18 / 6-24 / 12-30
        key = (seed, window)

        if key not in found:
            found[key] = {}

        found[key][kind] = p

    pairs = []
    for key, d in sorted(found.items()):
        if "s" in d and "m" in d:
            pairs.append((key, d["s"], d["m"]))

    return pairs


# --------------------------------------------------
# 메인 실행
# --------------------------------------------------
if __name__ == "__main__":
    pairs = find_all_pairs()

    if not pairs:
        print("No synthetic s*/m* csv pairs found.")
        print(f"Searched under: {PROJECT_ROOT}")
        raise SystemExit(1)

    print(f"Found {len(pairs)} synthetic pairs.\n")

    for (seed, window), snap_path, delta_path in pairs:
        out_snapshot = OUT_DIR / f"A_s{seed}({window}).csv"
        out_delta = OUT_DIR / f"A_m{seed}({window}).csv"

        # seed마다 고정 시드 조금 다르게
        seed_int = int(seed) + 100

        make_entorh(
            snapshot_path=snap_path,
            delta_path=delta_path,
            out_snapshot_path=out_snapshot,
            out_delta_path=out_delta,
            seed=seed_int,
        )

    print("\nAll done.")
    print(f"Generated files are in: {OUT_DIR}")