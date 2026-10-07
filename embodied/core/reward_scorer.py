#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reward_scorer.py

リプレイの「窓(シーケンス)」に、重要度スコアを付けるモジュール。
2通りのスコアを mode で切り替える。

  mode="reward"      : s = Σ|r|                       (主手法)
  mode="reward_fire" : s = Σ|r| + λ · g(f)            (比較条件)

      f = 窓の中で発射(Fire)を選んだステップの割合
      g(f) = clip((f - τ) / (1 - τ), 0, 1)
      τ = fire_threshold(既定 0.5)   λ = fire_weight(既定 1.0)

  発射の項は「窓の半分以上で発射している窓」だけに効く。
  盲検ラベリングで確認できたのは、発射35回以上(64ステップ中、約55%以上)の
  窓だけなので、それより少ない発射には加点しない。τ と λ は仮の値であり、
  根拠が強いわけではない(感度を見る対象)。

  優先度 = スコア + ε。ε は「スコアが0の窓も、最低限は選ばれる」ための下駄。
  選ばれる確率は  P(i) ∝ (スコア(i) + ε)^α  (α は優先度の指数)。

使い方(単体で動く。numpy だけ必要):
  python reward_scorer.py selftest
  python reward_scorer.py preview ~/logdir/xxx/replay

  selftest : このモジュール自体が正しく動くかの確認(データ不要)
  preview  : 実データで α と ε を決める前に、サンプリングがどれだけ偏るかを見る
"""

import argparse
import glob
import os
import sys

import numpy as np

MODES = ("reward", "reward_fire")


# ======================================================================
# スコア本体
# ======================================================================
class WindowScorer:
    """窓(長さ T の報酬・行動)の重要度スコアを計算する。"""

    def __init__(self, mode="reward", eps=0.01, fire_id=14,
                 fire_weight=1.0, fire_threshold=0.5):
        if mode not in MODES:
            raise ValueError(f"mode は {MODES} のどちらか: {mode!r}")
        if not eps > 0:
            raise ValueError("eps は 0 より大きくすること"
                             "(0 だとスコア0の窓が一度も選ばれなくなる)")
        if not 0.0 <= fire_threshold < 1.0:
            raise ValueError("fire_threshold は 0 以上 1 未満")
        if fire_weight < 0:
            raise ValueError("fire_weight は 0 以上")
        self.mode = mode
        self.eps = float(eps)
        self.fire_id = int(fire_id)
        self.fire_weight = float(fire_weight)
        self.fire_threshold = float(fire_threshold)

    @property
    def needs_action(self):
        """スコアの計算に行動が必要なモードか。"""
        return self.mode == "reward_fire"

    def fire_fraction(self, action, shape):
        """窓の中で発射を選んだステップの割合。action は整数(または one-hot)。"""
        a = np.asarray(action)
        if a.ndim == len(shape) + 1:  # one-hot → 番号
            a = a.argmax(-1)
        if a.shape != tuple(shape):
            raise ValueError(f"action の形 {a.shape} が reward の形 {tuple(shape)} と違う")
        return (a == self.fire_id).mean(axis=-1)

    def score(self, reward, action=None):
        """素のスコア(ε なし)。
        reward: 形 (T,) なら 1つの窓、(N, T) なら N個の窓。
        返り値: 1つの窓ならスカラー、N個なら長さ N の配列。"""
        r = np.asarray(reward, dtype=np.float64)
        if r.ndim not in (1, 2):
            raise ValueError("reward の形は (T,) か (N, T)")
        s = np.abs(r).sum(axis=-1)
        if self.mode == "reward_fire":
            if action is None:
                raise ValueError('mode="reward_fire" には action が必要')
            f = self.fire_fraction(action, r.shape)
            g = np.clip((f - self.fire_threshold) / (1.0 - self.fire_threshold),
                        0.0, 1.0)
            s = s + self.fire_weight * g
        return s

    def priority(self, reward, action=None):
        """優先度 = スコア + ε(必ず正)。セレクタにはこちらを渡す。"""
        return self.score(reward, action) + self.eps


# ======================================================================
# 確率・偏りの計算(preview と selftest で使う)
# ======================================================================
def probabilities(priority, alpha):
    """P(i) ∝ priority(i)^alpha。alpha=0 なら一様。"""
    p = np.power(np.asarray(priority, dtype=np.float64), float(alpha))
    return p / p.sum()


def concentration(priority, alpha, flags):
    """選ばれ方の偏りを調べる。
    flags: {名前: 真偽の配列}。名前ごとに『その窓に当たる確率の合計』と
           基準の割合(一様に選んだ場合の割合)を返す。
    ess_frac: 有効サンプル率(1.0 = 一様、小さいほど少数の窓に集中)。"""
    p = probabilities(priority, alpha)
    out = {"ess_frac": float(1.0 / (len(p) * np.sum(p ** 2)))}
    for name, m in flags.items():
        m = np.asarray(m, dtype=bool)
        out[name] = (float(p[m].sum()), float(m.mean()))
    return out


# ======================================================================
# preview: 実データでの偏りを見る
# ======================================================================
def load_replay(replay_dir):
    files = sorted(glob.glob(os.path.join(replay_dir, "**", "*.npz"),
                             recursive=True))
    if not files:
        sys.exit(f"[エラー] {replay_dir} 以下に .npz が見つかりません。")
    chunks = []
    for p in files:
        try:
            with np.load(p, allow_pickle=False) as z:
                missing = [k for k in ("action", "reward") if k not in z.files]
                if missing:
                    sys.exit(f"[エラー] {p} にキー {missing} がありません。"
                             f" 含まれるキー: {list(z.files)}")
                a, r = z["action"], z["reward"]
        except Exception as e:  # 学習中で書き込み途中のファイルなど
            print(f"[警告] スキップ: {os.path.basename(p)} ({e})")
            continue
        if a.ndim > 1:
            a = a.argmax(-1)
        chunks.append((a.astype(np.int64).reshape(-1),
                       r.astype(np.float64).reshape(-1)))
    if not chunks:
        sys.exit("[エラー] 読み込めるチャンクがありません。")
    return chunks


def cmd_preview(args):
    chunks = load_replay(args.replay_dir)
    T = args.window
    n_steps = sum(len(r) for _, r in chunks)
    n_ev = int(sum(np.count_nonzero(r) for _, r in chunks))
    print(f"データ: {args.replay_dir}")
    print(f"  チャンク {len(chunks)} 個 / {n_steps} ステップ / 報酬イベント {n_ev} 件")
    if n_ev == 0:
        sys.exit("[エラー] 報酬が1件もないデータです。別のフォルダを指定してください。")

    R, A = [], []
    for a, r in chunks:
        if len(r) < T:
            continue
        sw = np.lib.stride_tricks.sliding_window_view
        R.append(sw(r, T)[::args.stride])
        A.append(sw(a, T)[::args.stride])
    if not R:
        sys.exit(f"[エラー] 長さ {T} 以上のチャンクがありません。")
    R, A = np.concatenate(R), np.concatenate(A)

    has_r = np.abs(R).sum(axis=1) > 0
    heavy = (A == args.fire_id).mean(axis=1) > args.fire_threshold
    print(f"  窓: {len(R)} 個(長さ {T}、{args.stride} ステップ間隔)")
    print(f"  報酬のある窓: {has_r.sum()} ({100 * has_r.mean():.1f}%)   "
          f"発射が過半数の窓: {heavy.sum()} ({100 * heavy.mean():.1f}%)")
    flags = {"reward": has_r, "heavy": heavy}

    for mode in MODES:
        sc = WindowScorer(mode=mode, fire_id=args.fire_id,
                          fire_weight=args.fire_weight,
                          fire_threshold=args.fire_threshold)
        s = sc.score(R, A) if mode == "reward_fire" else sc.score(R)
        print("\n" + "=" * 78)
        print(f"mode = {mode}")
        print("=" * 78)
        print("  各窓が選ばれる確率の合計(括弧内は、一様に選んだ場合の割合)")
        print(f"  {'ε':>6s} {'α':>5s} | {'報酬のある窓':>16s} | "
              f"{'発射が過半数の窓':>18s} | {'有効サンプル率':>12s}")
        for eps in args.eps:
            pr = s + eps
            for alpha in args.alphas:
                c = concentration(pr, alpha, flags)
                (m1, b1), (m2, b2) = c["reward"], c["heavy"]
                print(f"  {eps:6.3g} {alpha:5.2f} | "
                      f"{100 * m1:6.1f}% ({100 * b1:4.1f}%) | "
                      f"{100 * m2:8.1f}% ({100 * b2:4.1f}%) | "
                      f"{100 * c['ess_frac']:10.1f}%")
    print("\n読み方:")
    print("・有効サンプル率が小さいほど、少数の窓を何度も繰り返し使う(過学習の危険)。")
    print("・α=0 は従来の一様サンプリング(100%)。")
    print("・ε を小さくするほど、スコア0の窓が選ばれにくくなり、偏りが強くなる。")
    print("・ここでの確率は『バッファ全体から直接選ぶ』場合。実際の学習では、")
    print("  バッチの一部がオンラインキューから来るなど、影響は小さくなりうる。")


# ======================================================================
# selftest: モジュールが正しく動くかの確認
# ======================================================================
def cmd_selftest(_args):
    ok = 0

    def check(name, cond):
        nonlocal ok
        print(("OK   " if cond else "FAIL ") + name)
        if not cond:
            sys.exit(1)
        ok += 1

    def raises(fn):
        try:
            fn()
        except ValueError:
            return True
        return False

    T = 8
    r = np.array([0, 0, 1, 0, -2, 0, 0, 0], dtype=float)
    a_none = np.zeros(T, int)                        # 発射なし
    a_75 = np.array([14, 14, 14, 14, 14, 14, 0, 0])  # 発射 6/8 = 0.75
    a_50 = np.array([14, 14, 14, 14, 0, 0, 0, 0])    # ちょうど 0.5
    a_all = np.full(T, 14)

    sc_r = WindowScorer("reward")
    sc_f = WindowScorer("reward_fire")

    check("reward: Σ|r| = 3", sc_r.score(r) == 3.0)
    check("reward: 報酬ゼロの窓は 0", sc_r.score(np.zeros(T)) == 0.0)
    check("reward: 負の報酬も絶対値で足す", sc_r.score(-np.ones(T)) == float(T))
    check("reward_fire: 発射なし → 加点0", sc_f.score(r, a_none) == 3.0)
    check("reward_fire: ちょうど半分 → 加点0", sc_f.score(r, a_50) == 3.0)
    check("reward_fire: 75% → 加点 0.5", abs(sc_f.score(r, a_75) - 3.5) < 1e-12)
    check("reward_fire: 全部発射 → 加点 1.0",
          abs(sc_f.score(np.zeros(T), a_all) - 1.0) < 1e-12)
    check("reward_fire: fire_weight=2 なら加点も2倍",
          abs(WindowScorer("reward_fire", fire_weight=2.0)
              .score(np.zeros(T), a_all) - 2.0) < 1e-12)

    onehot = np.eye(15)[a_75]
    check("one-hot の行動でも同じ結果", sc_f.score(r, onehot) == sc_f.score(r, a_75))

    rng = np.random.default_rng(0)
    N = 50
    Rb = rng.integers(-1, 2, (N, 64)).astype(float) * (rng.random((N, 64)) < 0.05)
    Ab = rng.integers(0, 15, (N, 64))
    Ab[:20, :40] = 14  # 一部の窓は発射が多い
    loop = np.array([sc_f.score(Rb[i], Ab[i]) for i in range(N)])
    check("まとめて計算 = 1つずつ計算", np.allclose(sc_f.score(Rb, Ab), loop))
    check("まとめて計算の形は (N,)", sc_f.score(Rb, Ab).shape == (N,))

    check("優先度は常に正(全部ゼロの窓でも)",
          sc_r.priority(np.zeros(T)) == 0.01 and sc_f.priority(np.zeros(T), a_none) > 0)
    check("優先度 = スコア + ε", abs(sc_r.priority(r) - 3.01) < 1e-12)

    pr = np.array([0.01, 1.01, 2.01, 5.01])
    check("α=0 は一様", np.allclose(probabilities(pr, 0.0), 0.25))
    check("確率の合計は 1", abs(probabilities(pr, 0.7).sum() - 1.0) < 1e-12)
    check("α を上げるとスコアの高い窓がより選ばれる",
          probabilities(pr, 1.0)[3] > probabilities(pr, 0.5)[3] > probabilities(pr, 0.0)[3])

    # 実データの規模(8,544窓のうち報酬のある窓が232)を模した確認
    flag = np.arange(8544) < 232
    pri = np.where(flag, 1.0, 0.0) + 0.01
    m05 = concentration(pri, 0.5, {"r": flag})["r"][0]
    m10 = concentration(pri, 1.0, {"r": flag})["r"][0]
    check(f"ε=0.01, α=0.5 で報酬窓の確率の合計 ≈ 21.9% (計算値 {100 * m05:.1f}%)",
          abs(m05 - 0.219) < 0.002)
    check(f"ε=0.01, α=1.0 で報酬窓の確率の合計 ≈ 73.8% (計算値 {100 * m10:.1f}%)",
          abs(m10 - 0.738) < 0.002)
    check("α=0 の有効サンプル率は 100%",
          abs(concentration(pri, 0.0, {})["ess_frac"] - 1.0) < 1e-12)

    check("不正な mode はエラー", raises(lambda: WindowScorer("xxx")))
    check("ε=0 はエラー", raises(lambda: WindowScorer("reward", eps=0)))
    check("fire_threshold=1 はエラー",
          raises(lambda: WindowScorer("reward_fire", fire_threshold=1.0)))
    check("needs_action: reward は不要、reward_fire は必要",
          (not sc_r.needs_action) and sc_f.needs_action)
    check("reward_fire で action なしはエラー", raises(lambda: sc_f.score(r)))
    check("action の形が違うとエラー",
          raises(lambda: sc_f.score(r, np.zeros(T + 1, int))))
    check("reward が3次元だとエラー", raises(lambda: sc_r.score(np.zeros((2, 2, 2)))))

    print(f"\n全 {ok} 項目に合格")


# ======================================================================
def main():
    # このファイルを embodied/core/ の中から直接実行すると、同じフォルダの random.py
    # が標準ライブラリの random を隠し、numpy が読み込めなくなる。それを避ける。
    here = os.path.dirname(os.path.abspath(__file__))
    if sys.path and os.path.abspath(sys.path[0]) == here:
        sys.path.pop(0)

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("selftest", help="モジュール自体の確認(データ不要)")
    s.set_defaults(func=cmd_selftest)

    p = sub.add_parser("preview", help="実データで偏りの強さを見る")
    p.add_argument("replay_dir")
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--fire-id", type=int, default=14)
    p.add_argument("--fire-weight", type=float, default=1.0)
    p.add_argument("--fire-threshold", type=float, default=0.5)
    p.add_argument("--alphas", type=float, nargs="+", default=[0.3, 0.5, 0.7, 1.0])
    p.add_argument("--eps", type=float, nargs="+", default=[0.01, 0.1])
    p.set_defaults(func=cmd_preview)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
