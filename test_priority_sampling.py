#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_priority_sampling.py

重要度サンプリング(Scored セレクタ + WindowScorer)の組み込みが正しく動くかを、
合成データで確認するテスト。GPU・環境・jax は不要。

使い方(リポジトリの直下に置いて実行):
    python test_priority_sampling.py

確認すること:
  1. 重要度サンプラーを渡しても、黙って Uniform に置き換わらない
  2. 各項目(窓)の優先度が、保存されたデータから計算した値と一致する
     (チャンクをまたぐ窓、容量超過による削除を含む)
  3. 実際の選ばれ方が、理論値(優先度^α に比例)と一致する
  4. α=0 では一様になる
  5. バッチの取り出しと、報酬を含む系列の割合の記録
  6. online=True、保存→再読み込み、別スレッドからの同時アクセス
  7. make_replay(設定→Replay の組み立て) が意図どおり動く
"""

import ast
import importlib
import os
import sys
import tempfile
import threading
import time
import traceback
import types

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
LENGTH = 65  # = consec_train * batch_length + replay_context (1*64+1)


def load_core():
    """embodied/__init__ (jax 等を読み込む)を避けて、core だけを読み込む。"""
    for name, sub in (("embodied", ""), ("embodied.core", "/core")):
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__path__ = [ROOT + "/embodied" + sub]
            sys.modules[name] = m
    return importlib.import_module("embodied.core.replay")


replay_lib = load_core()
selectors = replay_lib.selectors
reward_scorer = replay_lib.reward_scorer


# ----------------------------------------------------------------------
# 合成データ
# ----------------------------------------------------------------------
def make_step(reward=0.0, action=0, first=False, last=False):
    return dict(
        image=np.zeros((4, 4, 3), np.uint8), reward=np.float32(reward),
        action=np.int32(action), is_first=first, is_last=last, is_terminal=last)


class Stream:
    """報酬が低確率で出て、発射(14)が連続して現れる合成データ。
    発射が続いている間は、報酬も出やすい。"""

    def __init__(self, workers=2, seed=0):
        self.rng = np.random.default_rng(seed)
        self.workers = workers
        self.burst = [0] * workers
        self.t = [0] * workers

    def step(self, w):
        rng = self.rng
        if self.burst[w] > 0:
            self.burst[w] -= 1
            action, p = 14, 0.015
        elif rng.random() < 0.005:
            self.burst[w] = int(rng.integers(20, 60))
            action, p = 14, 0.015
        else:
            action, p = int(rng.integers(0, 14)), 0.0003
        reward = 1.0 if rng.random() < p else 0.0
        first = self.t[w] % 200 == 0
        self.t[w] += 1
        return make_step(reward, action, first=first)


def fill(rep, n, stream=None):
    stream = stream or Stream()
    for i in range(n):
        w = i % stream.workers
        rep.add(stream.step(w), worker=w)
    return stream


def make_replay(alpha=0.5, mode="reward_fire", eps=0.1, capacity=1500,
                chunksize=100, online=False, directory=None, seed=0):
    scorer = reward_scorer.WindowScorer(mode=mode, eps=eps)
    sel = selectors.Scored(alpha=alpha, seed=seed)
    return replay_lib.Replay(
        length=LENGTH, capacity=capacity, chunksize=chunksize, online=online,
        selector=sel, scorer=scorer, directory=directory, save_wait=True,
        seed=seed)


def expected_priorities(rep):
    """保存されているデータから、各項目の優先度を計算し直す(組み込みとは独立)。"""
    out = {}
    for itemid, (chunkid, index) in list(rep.items.items()):
        seq = rep._getseq(chunkid, index)
        out[itemid] = rep.scorer.priority(
            seq["reward"], seq["action"] if rep.scorer.needs_action else None)
    return out


def check_priorities(rep):
    exp = expected_priorities(rep)
    tree = rep.sampler.tree
    assert set(tree.entries) == set(exp), "木に入っている項目と、バッファの項目が違う"
    assert len(rep.sampler) == len(rep.items)
    a = rep.sampler.alpha
    got = np.array([tree.entries[k].uprob for k in exp])
    want = np.array([exp[k] ** a for k in exp])
    assert np.allclose(got, want), "木の重みが、データから計算した値と違う"
    return exp


# ----------------------------------------------------------------------
# テスト
# ----------------------------------------------------------------------
def test_no_silent_fallback():
    r = make_replay()
    assert isinstance(r.sampler, selectors.Scored), type(r.sampler)
    try:  # scorer があるのに、優先度を受け取れないセレクタ → 止まる
        replay_lib.Replay(length=LENGTH, scorer=reward_scorer.WindowScorer())
    except AssertionError:
        pass
    else:
        raise AssertionError("Uniform と scorer の組み合わせが止まらなかった")
    r0 = replay_lib.Replay(length=LENGTH)
    assert isinstance(r0.sampler, selectors.Uniform), "既定は Uniform のまま"


def test_priorities_match_data():
    for mode in ("reward", "reward_fire"):
        r = make_replay(mode=mode)  # chunksize=100: 窓がチャンクをまたぐ
        fill(r, 6000)               # capacity=1500: 削除が何度も起きる
        assert len(r) == 1500, len(r)
        exp = check_priorities(r)
        vals = np.array(list(exp.values()))
        assert vals.max() > vals.min(), "優先度が全部同じ(データに偏りがない)"
        print(f"      [{mode}] 項目 {len(r)}、優先度 {vals.min():.2f}〜{vals.max():.2f}")


def test_sampling_matches_theory():
    r = make_replay(alpha=0.5, mode="reward_fire")
    fill(r, 6000)
    exp = check_priorities(r)
    ids = np.array(list(exp))
    w = np.array([exp[k] for k in ids]) ** r.sampler.alpha
    p = w / w.sum()
    has_rew = np.array([
        np.abs(r._getseq(*r.items[k])["reward"]).sum() > 0 for k in ids])
    N = 40000
    counts = {k: 0 for k in ids}
    for _ in range(N):
        counts[r.sampler()] += 1
    freq = np.array([counts[k] for k in ids]) / N
    th, emp = float(p[has_rew].sum()), float(freq[has_rew].sum())
    sigma = np.sqrt(th * (1 - th) / N)
    base = float(has_rew.mean())
    print(f"      報酬を含む窓: 一様なら {100 * base:.1f}%、理論 {100 * th:.1f}%、"
          f"実測 {100 * emp:.1f}%(±{100 * 4 * sigma:.1f}% 以内なら合格)")
    assert abs(emp - th) < 4 * sigma, (emp, th, sigma)
    assert th > 1.5 * base, "優先度をつけても報酬窓がほとんど増えない設定のテストになっている"


def test_alpha_zero_is_uniform():
    r = make_replay(alpha=0.0, mode="reward_fire")
    fill(r, 6000)
    check_priorities(r)
    tree = r.sampler.tree
    assert all(abs(e.uprob - 1.0) < 1e-12 for e in tree.entries.values())
    has_rew = np.array([
        np.abs(r._getseq(*r.items[k])["reward"]).sum() > 0 for k in r.items])
    N = 20000
    hits = sum(
        bool(np.abs(r._getseq(*r.items[r.sampler()])["reward"]).sum() > 0)
        for _ in range(N))
    base = float(has_rew.mean())
    sigma = np.sqrt(base * (1 - base) / N)
    print(f"      報酬窓: 基準 {100 * base:.1f}%、α=0 の実測 {100 * hits / N:.1f}%")
    assert abs(hits / N - base) < 4 * sigma


def test_batch_and_metrics():
    r = make_replay(alpha=0.5)
    fill(r, 4000)
    r.stats()  # 取り込み時の記録を捨てる
    n, rew = 0, 0
    for _ in range(100):
        data = r.sample(16, "train")
        assert data["reward"].shape == (16, LENGTH), data["reward"].shape
        assert data["action"].shape == (16, LENGTH)
        assert data["is_first"][:, 0].all()
        n += 16
        rew += int((np.abs(data["reward"]).sum(1) > 0).sum())
    stats = r.stats()
    assert abs(stats["rew_seq_ratio"] - rew / n) < 1e-12, (stats["rew_seq_ratio"], rew / n)
    print(f"      学習に使った系列のうち報酬を含む割合(記録値): {100 * stats['rew_seq_ratio']:.1f}%")
    # 優先度なし(従来)でも、同じ指標が記録される
    r0 = replay_lib.Replay(length=LENGTH, capacity=1500, chunksize=100)
    fill(r0, 4000)
    r0.stats()
    for _ in range(100):
        r0.sample(16, "train")
    print(f"      従来(一様)での同じ割合: {100 * r0.stats()['rew_seq_ratio']:.1f}%")


def test_online_mode():
    r = make_replay(online=True, capacity=5000)  # 削除が起きない大きさ
    fill(r, 3000)
    assert len(r.queue) > 0, "online キューが空"
    queued = len(r.queue)
    data = r.sample(16, "train")
    assert data["reward"].shape == (16, LENGTH)
    assert len(r.queue) == max(queued - 16, 0)
    print(f"      online キュー {queued} 件 → 取り出し後 {len(r.queue)} 件")


def test_save_and_load():
    with tempfile.TemporaryDirectory() as d:
        r = make_replay(directory=d, capacity=5000)
        fill(r, 2500)
        r.save()
        r2 = make_replay(directory=d, capacity=5000)
        r2.load()
        assert len(r2) > 500, len(r2)
        check_priorities(r2)
        print(f"      保存→再読み込みで {len(r2)} 項目、優先度はデータと一致")


def test_threads():
    r = make_replay(capacity=3000)
    stream = fill(r, 300)
    errors, stop = [], threading.Event()
    n_add = n_sample = 0

    def producer():
        nonlocal n_add
        try:
            i = 0
            while not stop.is_set():
                w = i % stream.workers
                r.add(stream.step(w), worker=w)
                i += 1
                n_add += 1
        except Exception:
            errors.append(traceback.format_exc())

    def consumer():
        nonlocal n_sample
        try:
            while not stop.is_set():
                r.sample(16, "train")
                n_sample += 1
        except Exception:
            errors.append(traceback.format_exc())

    ts = [threading.Thread(target=producer), threading.Thread(target=consumer)]
    [t.start() for t in ts]
    time.sleep(4)
    stop.set()
    [t.join() for t in ts]
    assert not errors, errors[0]
    assert len(r.sampler) == len(r.items)
    check_priorities(r)
    print(f"      4秒間で 追加 {n_add} ステップ、バッチ取り出し {n_sample} 回、エラーなし")


def test_make_replay_function():
    """dreamerv3/main.py の make_replay を、そのまま取り出して動かす。"""
    import elements
    from ruamel import yaml

    src = open(os.path.join(ROOT, "dreamerv3/main.py"), encoding="utf-8").read()
    fn = next(n for n in ast.parse(src).body
              if isinstance(n, ast.FunctionDef) and n.name == "make_replay")
    ns = {"elements": elements, "np": np,
          "embodied": types.SimpleNamespace(replay=replay_lib)}
    exec(compile(ast.Module([fn], []), "main.py", "exec"), ns)
    build = ns["make_replay"]

    cfgs = yaml.YAML(typ="safe").load(
        elements.Path(os.path.join(ROOT, "dreamerv3/configs.yaml")).read())

    def config(**replay):
        c = elements.Config(cfgs["defaults"]).update(cfgs["dmlab"])
        with tempfile.TemporaryDirectory() as d:
            c = c.update(logdir=d)
            c = c.update({"replay": {**c.replay, "size": 1e5, **replay}})
            return build(c, "replay"), c

    with tempfile.TemporaryDirectory():
        r, _ = config()
        assert isinstance(r.sampler, selectors.Uniform) and r.scorer is None
        r, _ = config(prio_mode="reward", prio_alpha=0.3, prio_eps=0.05)
        assert isinstance(r.sampler, selectors.Scored) and r.sampler.alpha == 0.3
        assert r.scorer.mode == "reward" and r.scorer.eps == 0.05
        r, _ = config(prio_mode="reward_fire", prio_fire_weight=2.0)
        assert r.scorer.mode == "reward_fire" and r.scorer.fire_weight == 2.0
        assert r.length == LENGTH, r.length
        try:
            config(prio_mode="reward", fracs={"uniform": 0.5, "priority": 0.5, "recency": 0.0})
        except AssertionError:
            pass
        else:
            raise AssertionError("既存の優先度付きとの併用が止まらなかった")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    order = ["test_no_silent_fallback", "test_priorities_match_data",
             "test_sampling_matches_theory", "test_alpha_zero_is_uniform",
             "test_batch_and_metrics", "test_online_mode", "test_save_and_load",
             "test_threads", "test_make_replay_function"]
    tests.sort(key=lambda kv: order.index(kv[0]))
    failed = 0
    for name, fn in tests:
        t0 = time.time()
        print(f"{name} ...")
        try:
            fn()
            print(f"OK   {name} ({time.time() - t0:.1f}秒)")
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print("\n" + ("全て合格" if not failed else f"不合格 {failed} 件"))
    sys.exit(1 if failed else 0)
