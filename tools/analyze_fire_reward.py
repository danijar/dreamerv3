#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_fire_reward.py

目的:
    「発射行動(Fire)が多いシーケンスほど、戦闘シーン(報酬が発生するシーン)で
    ある」という仮説に、データで根拠があるかを検証する。

前提(embodied/envs/dmlab.py の確認結果):
    - 行動は 0〜14 の整数1個(POPART_ACTION_SET、15種類)
    - 発射(Fire)は 14 番のみ(移動や視点移動との組み合わせ行動は無い)
    - 1ステップ = ゲーム内4フレーム(repeat=4)。報酬は4フレーム分の合計
    - reward[t] は「1つ前の行動 action[t-1] の結果」として記録される

出力(すべて --out のフォルダに保存):
    summary.txt   … 結果の文章まとめ
    summary.png   … 図(matplotlib がある場合のみ)
    sheet_*.png   … 目視確認用の画像シート(手順4)

検証の手順:
    手順0 行動の内訳を見る(発射を連打していないか)
    手順1 発射の直後に報酬が増えるか(発射しないステップとの比較つき)
    手順2 シーケンス(窓)単位で、発射回数が報酬の有無を見分けられるか(AUC)
    手順3 発射回数のグループ別に、報酬が出る窓の割合を比べる
    手順4 「発射は多いが報酬ゼロ」の窓の画像を並べて、目で戦闘か確認する

使い方:
    python analyze_fire_reward.py ~/logdir/xxx/replay --fire-id 14 --window 64

注意:
    この分析が示すのは「データを集めた時点の方策のもとでの関係」である。
    方策が変われば発射の仕方も変わるので、学習の序盤・中盤・終盤など
    異なる時期のデータでも確認することが望ましい。
"""

import argparse
import glob
import os
import sys

import numpy as np

# POPART_ACTION_SET の並び順(embodied/envs/dmlab.py のコメントに対応)
ACTION_NAMES = [
    "FW", "BW", "StrafeL", "StrafeR", "SmallLL", "SmallLR", "LargeLL",
    "LargeLR", "LookDown", "LookUp", "FW+SmallLL", "FW+SmallLR",
    "FW+LargeLL", "FW+LargeLR", "Fire",
]

LINES = []


def log(msg=""):
    print(msg)
    LINES.append(str(msg))


# ----------------------------------------------------------------------
# 読み込み
# ----------------------------------------------------------------------
def load_chunks(replay_dir, max_chunks=None):
    """各npzから action / reward / is_first だけを読む(画像は読まない)。"""
    files = sorted(glob.glob(os.path.join(replay_dir, "**", "*.npz"),
                             recursive=True))
    if not files:
        sys.exit(f"[エラー] {replay_dir} 以下に .npz が見つかりません。")
    if max_chunks:
        files = files[:max_chunks]

    chunks, paths = [], []
    for p in files:
        try:
            with np.load(p, allow_pickle=False) as z:
                miss = [k for k in ("action", "reward", "is_first")
                        if k not in z.files]
                if miss:
                    sys.exit(
                        f"[エラー] {p} にキー {miss} がありません。\n"
                        f"含まれるキー: {list(z.files)}\n"
                        "inspect_replay_data.py で構造を確認してください。")
                a, r, f = z["action"], z["reward"], z["is_first"]
        except Exception as e:  # 学習中で書き込み途中のファイルなど
            log(f"[警告] 読み込みをスキップ: {os.path.basename(p)} ({e})")
            continue
        if a.ndim > 1:  # one-hot 形式だった場合の保険
            a = a.argmax(-1)
        chunks.append((a.astype(np.int64).reshape(-1),
                       r.astype(np.float32).reshape(-1),
                       f.astype(bool).reshape(-1)))
        paths.append(p)
    if not chunks:
        sys.exit("[エラー] 読み込めるチャンクがありませんでした。")
    return chunks, paths


# ----------------------------------------------------------------------
# 統計の小道具(scipy に依存しない)
# ----------------------------------------------------------------------
def rankdata(x):
    """同順位は平均順位(1始まり)を与える。"""
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    avg = np.cumsum(cnt) - (cnt - 1) / 2.0
    return avg[inv]


def spearman(x, y):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def auc(score, positive):
    """score が positive/negative をどれだけ見分けるか(0.5=偶然, 1.0=完全)。"""
    pos = positive.astype(bool)
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = rankdata(score)
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ----------------------------------------------------------------------
# 手順0: 行動の内訳
# ----------------------------------------------------------------------
def step0_action_histogram(chunks, fire_id):
    all_a = np.concatenate([c[0] for c in chunks])
    n_act = max(int(all_a.max()) + 1, len(ACTION_NAMES))
    counts = np.bincount(all_a, minlength=n_act)
    total = counts.sum()
    log("=" * 70)
    log("手順0: 行動の内訳")
    log("=" * 70)
    log(f"総ステップ数: {total}   チャンク数: {len(chunks)}")
    for i, c in enumerate(counts):
        name = ACTION_NAMES[i] if i < len(ACTION_NAMES) else "?"
        mark = "  <-- Fire" if i == fire_id else ""
        bar = "#" * int(round(40 * c / max(total, 1)))
        log(f"  {i:2d} {name:<11s} {c:8d} ({100 * c / total:5.1f}%) {bar}{mark}")
    frac = counts[fire_id] / total
    log(f"\n発射の割合: {100 * frac:.1f}%  "
        f"(15種類が均等に選ばれた場合の目安は約 6.7%)")
    if frac > 0.5:
        log("  -> 発射を連打している可能性が高い。発射回数は戦闘の指標として"
            "使いにくい。")
    return counts


# ----------------------------------------------------------------------
# 手順1: 発射の前後で報酬が出やすくなるか(対照つき)
# ----------------------------------------------------------------------
def step1_lag_curve(chunks, fire_id, lags):
    hit_f = np.zeros(len(lags))
    n_f = np.zeros(len(lags))
    hit_n = np.zeros(len(lags))
    n_n = np.zeros(len(lags))
    for a, r, _ in chunks:
        fire = a == fire_id
        ev = r != 0
        L = len(a)
        for i, k in enumerate(lags):
            lo, hi = max(0, -k), min(L, L - k)
            if hi <= lo:
                continue
            fr = fire[lo:hi]
            e = ev[lo + k:hi + k]
            hit_f[i] += np.sum(e & fr)
            n_f[i] += np.sum(fr)
            hit_n[i] += np.sum(e & ~fr)
            n_n[i] += np.sum(~fr)
    pf = hit_f / np.maximum(n_f, 1)
    pn = hit_n / np.maximum(n_n, 1)

    log("")
    log("=" * 70)
    log("手順1: 発射の前後で報酬が出やすくなるか(対照つき)")
    log("=" * 70)
    log("lag k: ステップ t で発射したとき、t+k に報酬が記録される確率。")
    log("比較対象は「t で発射していない」場合。lift = 発射あり / 発射なし。")
    log("読み方:")
    log("  ・lift が全てのラグで 1 より大きい → 発射と報酬が『同じ場面』に")
    log("    集中している(発射は戦闘の場面にまとまって出る)。")
    log("  ・k=1〜数ステップで山がある → 発射の直後に報酬が出る直接の関係も")
    log("    ある(reward[t] は action[t-1] の結果なので k>=1 が結果側)。")
    log("  ・全ラグで lift≈1 → 発射と報酬は無関係(戦闘の指標にならない)。")
    log("  ※ 発射は連続して起きるので、k<=0 側も高くなるのは自然であり、")
    log("    k<=0 は『きれいな対照』ではない点に注意。")
    log(f"{'lag':>4s} {'P(報酬|発射)':>13s} {'P(報酬|非発射)':>15s} "
        f"{'lift':>6s}")
    for i, k in enumerate(lags):
        lift = pf[i] / pn[i] if pn[i] > 0 else float("nan")
        log(f"{k:4d} {pf[i]:13.4f} {pn[i]:15.4f} {lift:6.2f}")
    return pf, pn


# ----------------------------------------------------------------------
# 窓(シーケンス)の作成
# ----------------------------------------------------------------------
def make_windows(chunks, T, stride, fire_id, drop_cross_episode=True):
    cid, start, fire_c, absr, nev = [], [], [], [], []
    for ci, (a, r, f) in enumerate(chunks):
        L = len(a)
        if L < T:
            continue
        s = np.arange(0, L - T + 1, stride)
        cf = np.concatenate([[0], np.cumsum(a == fire_id)])
        cr = np.concatenate([[0.0], np.cumsum(np.abs(r))])
        ce = np.concatenate([[0], np.cumsum(r != 0)])
        c1 = np.concatenate([[0], np.cumsum(f)])
        keep = np.ones(len(s), bool)
        if drop_cross_episode:  # 窓の途中で新エピソードが始まるものは除外
            keep = (c1[s + T] - c1[s + 1]) == 0
        s = s[keep]
        cid.append(np.full(len(s), ci))
        start.append(s)
        fire_c.append(cf[s + T] - cf[s])
        absr.append(cr[s + T] - cr[s])
        nev.append(ce[s + T] - ce[s])
    if not cid:
        sys.exit(f"[エラー] 長さ {T} 以上のチャンクがありません。")
    return dict(chunk=np.concatenate(cid), start=np.concatenate(start),
                fire=np.concatenate(fire_c), absr=np.concatenate(absr),
                nev=np.concatenate(nev))


def bootstrap_ci(win, n_chunks, stat_fn, n_boot=200, seed=0):
    """チャンク単位のブートストラップで95%区間を出す(窓は重なるため)。"""
    rng = np.random.default_rng(seed)
    uc, first = np.unique(win["chunk"], return_index=True)
    ends = np.append(first[1:], len(win["chunk"]))
    by_chunk = [np.arange(s, e) for s, e in zip(first, ends)]
    if len(by_chunk) < 5:
        return None
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(by_chunk), len(by_chunk))
        ix = np.concatenate([by_chunk[i] for i in pick])
        v = stat_fn(ix)
        if not np.isnan(v):
            vals.append(v)
    if len(vals) < 20:
        return None
    return np.percentile(vals, [2.5, 97.5])


# ----------------------------------------------------------------------
# 手順2・3: 窓単位の関係
# ----------------------------------------------------------------------
GROUP_EDGES = [0, 1, 4, 8, 16, 32, 10 ** 9]
GROUP_LABELS = ["0", "1-3", "4-7", "8-15", "16-31", "32+"]


def step2_3_window_stats(win, T, n_chunks):
    n = len(win["fire"])
    has_r = win["nev"] > 0
    log("")
    log("=" * 70)
    log(f"手順2: 窓(長さ T={T} ステップ)単位で、発射回数は報酬の有無を見分けるか")
    log("=" * 70)
    log(f"窓の数: {n}   報酬が1回以上ある窓: {has_r.sum()} "
        f"({100 * has_r.mean():.1f}%)")
    if has_r.sum() < 10:
        log("[注意] 報酬のある窓が10未満です。結論は出せません。"
            "より多くのデータで実行してください。")

    A = auc(win["fire"], has_r)
    rho = spearman(win["fire"].astype(float), win["absr"])
    ci_a = bootstrap_ci(
        win, n_chunks, lambda ix: auc(win["fire"][ix], has_r[ix]))
    ci_r = bootstrap_ci(
        win, n_chunks,
        lambda ix: spearman(win["fire"][ix].astype(float), win["absr"][ix]))

    def fmt(v, ci):
        s = f"{v:.3f}"
        return s + (f"  (95%区間 {ci[0]:.3f}〜{ci[1]:.3f})" if ci is not None
                    else "")

    log(f"AUC(発射回数で「報酬あり窓」を見分ける): {fmt(A, ci_a)}")
    log("   0.5 = 偶然と同じ / 0.6前後 = 弱い / 0.7以上 = 実用的な目安")
    log(f"Spearman相関(発射回数 vs 報酬の絶対値合計): {fmt(rho, ci_r)}")

    log("")
    log("=" * 70)
    log("手順3: 発射回数のグループ別に見た、報酬が出る窓の割合")
    log("=" * 70)
    g = np.digitize(win["fire"], GROUP_EDGES[1:-1])
    base = has_r.mean()
    log(f"全体の平均: {100 * base:.1f}%")
    log(f"{'発射回数':>8s} {'窓の数':>8s} {'報酬あり率':>10s} "
        f"{'平均報酬和':>10s} {'全体比':>6s}")
    rates = []
    for gi, lab in enumerate(GROUP_LABELS):
        m = g == gi
        if m.sum() == 0:
            rates.append((lab, 0, float("nan")))
            log(f"{lab:>8s} {0:8d}")
            continue
        rate = has_r[m].mean()
        rates.append((lab, int(m.sum()), rate))
        ratio = rate / base if base > 0 else float("nan")
        log(f"{lab:>8s} {m.sum():8d} {100 * rate:9.1f}% "
            f"{win['absr'][m].mean():10.3f} {ratio:6.2f}")
    return A, rho, rates


# ----------------------------------------------------------------------
# 手順4: 目視確認用の画像シート
# ----------------------------------------------------------------------
def step4_sheets(win, paths, T, out_dir, n_rows, n_frames, seed):
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        log("\n[手順4] Pillow が無いため画像シートをスキップ (pip install Pillow)")
        return
    rng = np.random.default_rng(seed)
    zero = win["absr"] == 0
    fire = win["fire"]
    hi = max(1, int(np.percentile(fire[zero], 90))) if zero.any() else 1
    cats = {
        "A_reward": np.where(win["nev"] > 0)[0],
        "B_fire_no_reward": np.where(zero & (fire >= hi))[0],
        "C_no_fire_no_reward": np.where(zero & (fire == 0))[0],
    }
    log("")
    log("=" * 70)
    log("手順4: 目視確認用の画像シート")
    log("=" * 70)
    log(f"B の条件: 報酬ゼロ かつ 発射回数 >= {hi}")
    log("A=報酬あり(戦闘の見本) / B=発射は多いが報酬ゼロ / C=発射も報酬もゼロ")
    log("B が A に近い(敵が見える・レーザーが出ている)なら、報酬ゼロでも"
        "発射回数で戦闘を拾えている根拠になる。")
    log("B が C に近い(壁や通路だけ)なら、発射回数は戦闘の指標として弱い。")

    scale = 2
    fw = 64 * scale
    label_h = 14
    ids = np.linspace(0, T - 1, n_frames).astype(int)
    for cname, idx in cats.items():
        if len(idx) == 0:
            log(f"  {cname}: 該当する窓がありません。")
            continue
        pick = rng.choice(idx, size=min(n_rows, len(idx)), replace=False)
        rows = []
        for w in pick:
            path, s = paths[win["chunk"][w]], int(win["start"][w])
            try:
                with np.load(path, allow_pickle=False) as z:
                    if "image" not in z.files:
                        log("  npz に image キーが無いため画像シートを"
                            "スキップします。")
                        return
                    img = z["image"][s:s + T]
            except Exception as e:
                log(f"  [警告] 画像の読み込みに失敗: {e}")
                continue
            strip = Image.new("RGB", (n_frames * (fw + 2), label_h + fw),
                              (30, 30, 30))
            ImageDraw.Draw(strip).text(
                (2, 1),
                f"fire={int(fire[w])} reward_sum={win['absr'][w]:.0f} "
                f"{os.path.basename(path)[:28]} @{s}",
                fill=(255, 255, 255))
            for j, t in enumerate(ids):
                fr = Image.fromarray(np.asarray(img[t], dtype=np.uint8))
                fr = fr.resize((fw, fw), Image.NEAREST)
                strip.paste(fr, (j * (fw + 2), label_h))
            rows.append(strip)
        if not rows:
            continue
        sheet = Image.new("RGB", (rows[0].width, sum(r.height for r in rows)))
        y = 0
        for r in rows:
            sheet.paste(r, (0, y))
            y += r.height
        fn = os.path.join(out_dir, f"sheet_{cname}.png")
        sheet.save(fn)
        log(f"  保存: {fn}  ({len(rows)} 窓)")


# ----------------------------------------------------------------------
# 図
# ----------------------------------------------------------------------
def make_plot(out_dir, lags, pf, pn, counts, rates, fire_id):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log("\n[図] matplotlib が無いためグラフをスキップ "
            "(pip install matplotlib)")
        return
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(lags, pf, "o-", label="after Fire step")
    ax[0].plot(lags, pn, "s-", label="after non-Fire step")
    ax[0].axvline(0, color="gray", lw=0.5)
    ax[0].set_xlabel("lag k (reward at t+k)")
    ax[0].set_ylabel("P(reward event)")
    ax[0].set_title("Step 1: reward vs. Fire (with control)")
    ax[0].legend()
    labs = [r[0] for r in rates]
    vals = [0 if np.isnan(r[2]) else 100 * r[2] for r in rates]
    ax[1].bar(labs, vals)
    ax[1].set_xlabel("Fire count in window")
    ax[1].set_ylabel("% windows with reward")
    ax[1].set_title("Step 3: reward rate by Fire count")
    names = [ACTION_NAMES[i] if i < len(ACTION_NAMES) else str(i)
             for i in range(len(counts))]
    colors = ["tab:red" if i == fire_id else "tab:blue"
              for i in range(len(counts))]
    ax[2].bar(range(len(counts)), 100 * counts / counts.sum(), color=colors)
    ax[2].set_xticks(range(len(counts)))
    ax[2].set_xticklabels(names, rotation=70, fontsize=7)
    ax[2].set_ylabel("% of steps")
    ax[2].set_title("Step 0: action histogram")
    fig.tight_layout()
    fn = os.path.join(out_dir, "summary.png")
    fig.savefig(fn, dpi=120)
    log(f"\n図を保存: {fn}")


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("replay_dir", help="npz が入っているフォルダ")
    ap.add_argument("--fire-id", type=int, default=14)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--out", default="fire_reward_analysis")
    ap.add_argument("--max-chunks", type=int, default=None)
    ap.add_argument("--no-sheets", action="store_true")
    ap.add_argument("--sheet-rows", type=int, default=12)
    ap.add_argument("--sheet-frames", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    chunks, paths = load_chunks(args.replay_dir, args.max_chunks)

    counts = step0_action_histogram(chunks, args.fire_id)
    lags = list(range(-10, 21))
    pf, pn = step1_lag_curve(chunks, args.fire_id, lags)
    win = make_windows(chunks, args.window, args.stride, args.fire_id)
    A, rho, rates = step2_3_window_stats(win, args.window, len(chunks))
    if not args.no_sheets:
        step4_sheets(win, paths, args.window, args.out,
                     args.sheet_rows, args.sheet_frames, args.seed)
    make_plot(args.out, lags, pf, pn, counts, rates, args.fire_id)

    log("")
    log("=" * 70)
    log("読み方の注意")
    log("=" * 70)
    log("・数値だけで結論を出さず、手順4の画像も必ず目で確認すること。")
    log("・報酬のある窓が少ない場合、AUC は不安定になる。")
    log("・結果は「このデータを集めたときの方策」のもとでの関係である。")
    with open(os.path.join(args.out, "summary.txt"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(LINES))
    print(f"\n結果を保存しました: {args.out}/")


if __name__ == "__main__":
    main()
