#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
blind_label_tool.py

目的:
    「発射回数が多いが報酬ゼロの窓(B)」に、敵が実際に映っているかを、
    グループ名を隠した状態で人が判定し、比較する(盲検ラベリング)。

    A: 報酬あり(戦闘の見本)
    B: 報酬ゼロ・発射が多い(検証したい窓)
    D: 報酬ゼロ・発射が少ない(1〜3回)(対照。ごく普通の窓)

    B が D より敵の映る割合が明らかに高ければ、「発射回数は、報酬が
    出ていない戦闘を拾える」という根拠になる。

使い方:
    1) 作成
       python blind_label_tool.py make ~/logdir/xxx/replay --out blind_label --n 30
    2) ラベル付け
       blind_label/labeling.html をブラウザで開いて判定する
       (完了したら「CSVをダウンロード」→ labels.csv)
    3) 集計(ラベル付けが終わるまで answer_key は開かないこと)
       python blind_label_tool.py score blind_label/answer_key_DO_NOT_OPEN.csv labels.csv

判定の基準(HTMLにも同じ内容が表示される):
    各窓には64ステップを等間隔に抜いた8フレームが並ぶ。
    「敵ボット本体が映っているフレームの枚数(0〜8)」を数える。
    主要な判定は「3枚以上」(結果を見る前に決めておく)。

必要なもの:
    analyze_fire_reward.py と同じフォルダに置くこと。numpy, Pillow が必要。
"""

import argparse
import base64
import csv
import io
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_fire_reward import load_chunks, make_windows  # noqa: E402

PRIMARY_THRESHOLD = 3  # 「3フレーム以上」を主要な基準とする(事前に固定)


# ======================================================================
# make: 窓を選び、ラベル付け用HTMLと答えの表を作る
# ======================================================================
def cmd_make(args):
    try:
        from PIL import Image
    except ImportError:
        sys.exit("Pillow が必要です: pip install Pillow")

    chunks, paths = load_chunks(args.replay_dir, args.max_chunks)
    T = args.window
    win = make_windows(chunks, T, args.stride, args.fire_id)
    zero = win["absr"] == 0
    fire = win["fire"]
    if args.fire_hi:
        hi = args.fire_hi
    else:
        hi = max(1, int(np.percentile(fire[zero], 90))) if zero.any() else 1

    cand = {
        "A": np.where(win["nev"] > 0)[0],
        "B": np.where(zero & (fire >= hi))[0],
        "D": np.where(zero & (fire >= 1) & (fire <= 3))[0],
    }
    print(f"窓の総数: {len(fire)}")
    print(f"候補数  A(報酬あり): {len(cand['A'])}   "
          f"B(報酬ゼロ・発射>={hi}回): {len(cand['B'])}   "
          f"D(報酬ゼロ・発射1〜3回): {len(cand['D'])}")

    # 窓が重ならないように選ぶ(同じ場面が複数回出ないように)。
    # 数が少ない A から先に選ぶ。
    rng = np.random.default_rng(args.seed)
    taken, picked = {}, []
    for g in ("A", "B", "D"):
        n_ok = 0
        for w in rng.permutation(cand[g]):
            c, s = int(win["chunk"][w]), int(win["start"][w])
            if any(abs(s - s2) < T for s2 in taken.get(c, ())):
                continue
            taken.setdefault(c, []).append(s)
            picked.append((g, int(w)))
            n_ok += 1
            if n_ok >= args.n:
                break
        if n_ok < args.n:
            print(f"[注意] グループ {g} は重ならない窓が {n_ok} 個しか"
                  f"取れませんでした(希望 {args.n})。")
    if not picked:
        sys.exit("[エラー] 選べる窓がありません。")

    picked = [picked[i] for i in rng.permutation(len(picked))]  # 順序を混ぜる

    frame_ids = np.linspace(0, T - 1, args.frames).astype(int)
    cols = args.cols
    rows = -(-args.frames // cols)
    items, key_rows = [], []
    for k, (g, w) in enumerate(picked, start=1):
        path, s = paths[win["chunk"][w]], int(win["start"][w])
        with np.load(path, allow_pickle=False) as z:
            if "image" not in z.files:
                sys.exit("[エラー] npz に image キーがありません。")
            img = z["image"][s:s + T]
        H0, W0 = img.shape[1], img.shape[2]
        H = H0 - args.crop_hud  # 画面下のHUD(数値やアイコン)を切り落とす
        canvas = np.zeros((rows * H, cols * W0, 3), np.uint8)
        for j, t in enumerate(frame_ids):
            r, c = divmod(j, cols)
            canvas[r * H:(r + 1) * H, c * W0:(c + 1) * W0] = img[t][:H]
        buf = io.BytesIO()
        Image.fromarray(canvas).save(buf, format="PNG")
        items.append({"id": k,
                      "img": base64.b64encode(buf.getvalue()).decode()})
        key_rows.append([k, g, int(win["fire"][w]), float(win["absr"][w]),
                         int(win["nev"][w]), os.path.basename(path), s])

    os.makedirs(args.out, exist_ok=True)
    run_id = "%08x" % int(rng.integers(0, 2 ** 32))
    html = (HTML_TEMPLATE
            .replace("__ITEMS__", json.dumps(items))
            .replace("__COLS__", str(cols))
            .replace("__ROWS__", str(rows))
            .replace("__NFRAMES__", str(args.frames))
            .replace("__RUNID__", run_id))
    html_path = os.path.join(args.out, "labeling.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    key_path = os.path.join(args.out, "answer_key_DO_NOT_OPEN.csv")
    with open(key_path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["id", "group", "fire_count", "reward_sum",
                     "n_reward_events", "chunk_file", "start"])
        wr.writerows(sorted(key_rows))

    counts = {g: sum(1 for r in key_rows if r[1] == g) for g in "ABD"}
    print(f"\n作成しました: {len(items)} 窓  {counts}")
    print(f"  ラベル付け用: {html_path}  "
          f"({os.path.getsize(html_path) / 1e6:.1f} MB)")
    print(f"  答えの表    : {key_path}  <- ラベル付けが終わるまで開かない")
    print("\nWSL から Windows のブラウザで開くには:")
    print(f"  explorer.exe {args.out}   (開いたフォルダで labeling.html を"
          "ダブルクリック)")


# ======================================================================
# score: ラベルを集計してグループ間を比較する
# ======================================================================
def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def perm_test(x, y, n_perm=20000, seed=0):
    """平均の差(2群)の両側並べ替え検定。二値(0/1)にもそのまま使える。"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    obs = x.mean() - y.mean()
    pool = np.concatenate([x, y])
    nx = len(x)
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        if abs(pool[:nx].mean() - pool[nx:].mean()) >= abs(obs) - 1e-12:
            cnt += 1
    return obs, (cnt + 1) / (n_perm + 1)


def boot_diff_ci(x, y, n_boot=5000, seed=0):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rng = np.random.default_rng(seed)
    d = [rng.choice(x, len(x)).mean() - rng.choice(y, len(y)).mean()
         for _ in range(n_boot)]
    return tuple(np.percentile(d, [2.5, 97.5]))


def read_key(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return {int(r["id"]): r for r in csv.DictReader(f)}


def read_labels(path):
    out, blank = {}, 0
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            v = (r.get("label") or "").strip()
            if v == "":
                blank += 1
                continue
            out[int(r["id"])] = v
    return out, blank


def cmd_score(args):
    key = read_key(args.key)
    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    binary = {}  # rater -> {id: 0/1}(主要基準)
    for lp in args.labels:
        labels, blank = read_labels(lp)
        log("=" * 70)
        log(f"ラベル: {lp}")
        log("=" * 70)
        if blank:
            log(f"[注意] 未回答が {blank} 件あります(集計から除外)。")

        by_g = {g: [] for g in "ABD"}
        unsure = {g: 0 for g in "ABD"}
        for i, v in labels.items():
            if i not in key:
                continue
            g = key[i]["group"]
            if v == "?":
                unsure[g] += 1
            else:
                by_g[g].append(int(v))
        binary[lp] = {i: int(int(v) >= PRIMARY_THRESHOLD)
                      for i, v in labels.items() if v != "?" and i in key}

        names = {"A": "A 報酬あり", "B": "B 報酬ゼロ・発射多",
                 "D": "D 報酬ゼロ・発射少"}
        log(f"\n敵が映るフレーム数(0〜8)の分布  [?は判断不能]")
        log(f"{'グループ':<22s} {'n':>3s} {'?':>2s} "
            + " ".join(f"{k:>2d}" for k in range(9)) + "   平均")
        for g in "ABD":
            v = by_g[g]
            hist = np.bincount(v, minlength=9) if v else np.zeros(9, int)
            mean = f"{np.mean(v):.2f}" if v else "-"
            log(f"{names[g]:<20s} {len(v):>3d} {unsure[g]:>2d} "
                + " ".join(f"{h:>2d}" for h in hist[:9]) + f"   {mean}")

        log(f"\n「敵が k 枚以上映る窓」の割合(95%区間はWilson法)")
        log("  ※ 主要な基準は k=3(事前に固定)。k=1, 5 は参考。")
        for thr in (1, PRIMARY_THRESHOLD, 5):
            mark = "  <- 主要" if thr == PRIMARY_THRESHOLD else ""
            log(f"  k>={thr}{mark}")
            for g in "ABD":
                v = np.array(by_g[g])
                if len(v) == 0:
                    continue
                k = int((v >= thr).sum())
                lo, hi = wilson(k, len(v))
                log(f"    {names[g]:<20s} {k:>2d}/{len(v):<2d} = "
                    f"{100 * k / len(v):5.1f}%  ({100 * lo:.0f}〜{100 * hi:.0f}%)")

        log(f"\nグループ間の比較(主要基準 k>={PRIMARY_THRESHOLD}、"
            "並べ替え検定は両側)")
        for a, b in (("B", "D"), ("A", "B"), ("A", "D")):
            xa = (np.array(by_g[a]) >= PRIMARY_THRESHOLD).astype(float)
            xb = (np.array(by_g[b]) >= PRIMARY_THRESHOLD).astype(float)
            if len(xa) < 3 or len(xb) < 3:
                log(f"  {a} vs {b}: 件数不足")
                continue
            d, p = perm_test(xa, xb)
            lo, hi = boot_diff_ci(xa, xb)
            log(f"  {a} - {b}: 割合の差 {100 * d:+.1f} ポイント "
                f"(95%区間 {100 * lo:+.0f}〜{100 * hi:+.0f})  p={p:.4f}")
            ma, mb = np.array(by_g[a], float), np.array(by_g[b], float)
            d2, p2 = perm_test(ma, mb)
            log(f"         平均枚数の差 {d2:+.2f} 枚  p={p2:.4f}")

    if len(args.labels) >= 2:
        log("\n" + "=" * 70)
        log("判定者間の一致(主要基準 k>=3 の二値、判断不能を除く)")
        log("=" * 70)
        l1, l2 = args.labels[0], args.labels[1]
        ids = sorted(set(binary[l1]) & set(binary[l2]))
        if len(ids) >= 5:
            a = np.array([binary[l1][i] for i in ids])
            b = np.array([binary[l2][i] for i in ids])
            po = float((a == b).mean())
            pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
            kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
            log(f"  共通の窓 {len(ids)} 件  一致率 {100 * po:.1f}%  "
                f"Cohenのκ = {kappa:.2f}")
            log("  κ の目安: 0.6以上で実用的、0.4未満は基準の見直しが必要。")
        else:
            log("  共通の窓が少なすぎます。")

    log("\n" + "=" * 70)
    log("読み方の注意")
    log("=" * 70)
    log("・B が D より明らかに高く、A に近い → 発射が多い窓には、報酬が")
    log("  出ていなくても敵が映っている。発射回数で戦闘を拾える根拠になる。")
    log("・B と D に差がない → 発射回数は戦闘の指標として弱い。")
    log("・完全な盲検ではない: 発射時のレーザーなどの光は画像に映るため、")
    log("  判定者が窓の種類を推測できてしまう可能性がある。")
    log("・1つの方策のデータでの結果であり、窓の数も少ない(区間が広い)。")
    out = os.path.splitext(args.labels[0])[0] + "_result.txt"
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n結果を保存しました: {out}")


# ======================================================================
# ラベル付け用HTML
# ======================================================================
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="ja"><head><meta charset="utf-8">
<title>盲検ラベリング</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  body{font-family:sans-serif;margin:0;background:#1b1b1f;color:#e8e8ea;}
  #wrap{max-width:1040px;margin:0 auto;padding:12px 16px 40px;}
  h1{font-size:18px;margin:6px 0;}
  details{background:#26262c;border-radius:8px;padding:8px 14px;margin:8px 0;
          font-size:14px;line-height:1.7;}
  summary{cursor:pointer;font-weight:bold;}
  #progress{font-size:14px;margin:8px 0;color:#b8b8c0;}
  #imgbox{position:relative;width:100%;background:#000;}
  #im{display:block;width:100%;image-rendering:pixelated;
      image-rendering:crisp-edges;}
  #grid{position:absolute;inset:0;display:grid;pointer-events:none;
        grid-template-columns:repeat(__COLS__,1fr);
        grid-template-rows:repeat(__ROWS__,1fr);}
  #grid div{border:1px solid rgba(255,255,255,.35);position:relative;}
  #grid span{position:absolute;left:3px;top:2px;background:rgba(0,0,0,.65);
             color:#fff;font-size:13px;padding:0 5px;border-radius:3px;}
  #buttons{display:flex;flex-wrap:wrap;gap:6px;margin:12px 0;}
  button{font-size:16px;padding:8px 14px;border-radius:6px;border:1px solid #555;
         background:#33333b;color:#eee;cursor:pointer;}
  button:hover{background:#44444e;}
  button.sel{background:#2a78d6;border-color:#2a78d6;color:#fff;}
  #cur{font-size:15px;margin:6px 0;}
  #cur b{font-size:18px;}
  .small{font-size:12px;color:#9a9aa4;}
</style></head><body><div id="wrap">
<h1>盲検ラベリング</h1>
<details open><summary>判定ルール(最初に必ず読む。途中で変えないこと)</summary>
<ol>
<li>各窓には、約64ステップを等間隔に抜いた <b>__NFRAMES__ 枚のフレーム</b>が
番号つきで並びます。<b>「敵が見えるフレームが何枚あるか」</b>を
<b>0〜__NFRAMES__</b> の数字で答えてください。</li>
<li><b>「敵が見える」＝ そのフレームに敵ボット本体が映っている。</b>
ボットは、青い壁・オレンジの床とは違う色の、小さな丸みのある物体です
(黄・緑・赤と青・紫・オレンジなど)。遠くでは数ピクセルの色の点、
近くでは大きな球状の体に見えます。</li>
<li><b>数えないもの：</b>水色のリングや光の球、ピンクや白の光の柱、
レーザーの光線だけのフレーム、壁や床の模様・看板・色違いの壁面。</li>
<li><b>迷ったら数えない。</b>ボットか模様か確信が持てない点は数えません。
この基準を最後まで変えないでください。</li>
<li>画像が真っ暗・壊れているなど、どうしても判断できない窓だけ
<b>「？」</b>を選びます(壁の大写しで敵が見えないだけなら 0)。</li>
<li>小さくて見づらいときはブラウザを拡大(Ctrl と +)してください。</li>
<li>窓の種類を推測しようとせず、「敵が見えるか」だけを数えてください。</li>
</ol>
<div class="small">ショートカット: 数字キー 0〜__NFRAMES__ で回答して自動で次へ /
<b>u</b> = ？ / ← → で移動 / Delete で回答を消す。回答はこのブラウザに自動保存されます。</div>
</details>
<div id="progress"></div>
<div id="cur"></div>
<div id="imgbox"><img id="im" alt="window"><div id="grid"></div></div>
<div id="buttons"></div>
<div>
  <button id="prev">← 前へ</button>
  <button id="next">次へ →</button>
  <button id="dl">CSVをダウンロード</button>
</div>
<p class="small">全部終わったら「CSVをダウンロード」→ labels.csv ができます。</p>
</div>
<script>
const ITEMS = __ITEMS__;
const NF = __NFRAMES__;
const KEY = "blind_label___RUNID__";
let idx = 0;
let labels = {};
try { labels = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { labels = {}; }

function save() { try { localStorage.setItem(KEY, JSON.stringify(labels)); } catch (e) {} }

function buildCsv() {
  const rows = ["id,label"];
  ITEMS.forEach(it => rows.push(it.id + "," + (labels[it.id] !== undefined ? labels[it.id] : "")));
  return rows.join("\n") + "\n";
}

function answered() { return ITEMS.filter(it => labels[it.id] !== undefined).length; }

function render() {
  const it = ITEMS[idx];
  document.getElementById("im").src = "data:image/png;base64," + it.img;
  document.getElementById("progress").textContent =
    "回答済み " + answered() + " / " + ITEMS.length +
    (answered() === ITEMS.length ? "  すべて完了。CSVをダウンロードしてください。" : "");
  const cur = labels[it.id];
  document.getElementById("cur").innerHTML =
    "窓 <b>#" + it.id + "</b>  (" + (idx + 1) + "/" + ITEMS.length + ")   現在の回答: <b>" +
    (cur === undefined ? "未回答" : (cur === "?" ? "？" : cur + " 枚")) + "</b>";
  document.querySelectorAll("#buttons button").forEach(b => {
    b.classList.toggle("sel", b.dataset.v === cur);
  });
}

function setLabel(v) {
  labels[ITEMS[idx].id] = v;
  save();
  if (idx < ITEMS.length - 1) idx++;
  render();
}
function clearLabel() { delete labels[ITEMS[idx].id]; save(); render(); }
function move(d) { idx = Math.max(0, Math.min(ITEMS.length - 1, idx + d)); render(); }

function download() {
  const blob = new Blob([buildCsv()], {type: "text/csv"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "labels.csv";
  document.body.appendChild(a); a.click(); a.remove();
}

(function init() {
  const g = document.getElementById("grid");
  for (let i = 1; i <= NF; i++) {
    const d = document.createElement("div");
    const s = document.createElement("span"); s.textContent = i;
    d.appendChild(s); g.appendChild(d);
  }
  const bx = document.getElementById("buttons");
  for (let v = 0; v <= NF; v++) {
    const b = document.createElement("button");
    b.textContent = v; b.dataset.v = String(v);
    b.onclick = () => setLabel(String(v)); bx.appendChild(b);
  }
  const q = document.createElement("button");
  q.textContent = "？(判断不能)"; q.dataset.v = "?"; q.onclick = () => setLabel("?"); bx.appendChild(q);
  document.getElementById("prev").onclick = () => move(-1);
  document.getElementById("next").onclick = () => move(1);
  document.getElementById("dl").onclick = download;
  document.addEventListener("keydown", e => {
    if (e.key >= "0" && e.key <= String(NF)) setLabel(e.key);
    else if (e.key === "u" || e.key === "?") setLabel("?");
    else if (e.key === "ArrowRight") move(1);
    else if (e.key === "ArrowLeft") move(-1);
    else if (e.key === "Delete") clearLabel();
  });
  // 最初の未回答へ移動
  const first = ITEMS.findIndex(it => labels[it.id] === undefined);
  if (first >= 0) idx = first;
  render();
})();
</script></body></html>
"""


# ======================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("make", help="ラベル付け用の窓を作る")
    m.add_argument("replay_dir")
    m.add_argument("--out", default="blind_label")
    m.add_argument("--n", type=int, default=30, help="各グループの窓の数")
    m.add_argument("--fire-id", type=int, default=14)
    m.add_argument("--fire-hi", type=int, default=0,
                   help="Bの発射回数の下限(0なら報酬ゼロ窓の90パーセンタイル)")
    m.add_argument("--window", type=int, default=64)
    m.add_argument("--stride", type=int, default=16)
    m.add_argument("--frames", type=int, default=8)
    m.add_argument("--cols", type=int, default=4)
    m.add_argument("--crop-hud", type=int, default=8,
                   help="画面下から切り落とす行数(HUD除去)。0で無効")
    m.add_argument("--max-chunks", type=int, default=None)
    m.add_argument("--seed", type=int, default=0)
    m.set_defaults(func=cmd_make)

    s = sub.add_parser("score", help="ラベルを集計する")
    s.add_argument("key", help="answer_key_DO_NOT_OPEN.csv")
    s.add_argument("labels", nargs="+",
                   help="labels.csv(複数なら判定者間の一致も計算)")
    s.set_defaults(func=cmd_score)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

