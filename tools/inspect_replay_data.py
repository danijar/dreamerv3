"""
inspect_replay_data.py

DreamerV3(embodiedライブラリ)が保存するリプレイバッファのnpzファイルを
読み込み、中に記録されているキー・形状・型・値の範囲を表示するための
調査専用スクリプト。

目的:
    「発射行動が多いシーンが戦闘シーンである」ことを示す分析スクリプトを
    作る前に、reward(報酬)とaction(行動)がどんな形式で保存されているかを
    確認する。このスクリプトはファイルを読み込むだけで、
    何も書き換えたり削除したりしない(安全に何度でも実行できる)。

使い方:
    python inspect_replay_data.py /path/to/replay/directory

    例(embodiedのデフォルト構成の場合):
    python inspect_replay_data.py ~/logdir/dmlab_lasertag/replay

    ディレクトリの中から .npz ファイルを探して、
    最初に見つかった1つの中身を詳しく表示する。
"""

import sys
import glob
import os
import numpy as np


def find_episode_files(replay_dir):
    """
    replay_dir 以下から .npz ファイルをすべて探す。
    embodied ライブラリは通常
    logdir/replay/ 以下にタイムスタンプ付きのファイル名で保存する。
    """
    if not os.path.isdir(replay_dir):
        print(f"[エラー] 指定されたパスがディレクトリではありません: {replay_dir}")
        sys.exit(1)

    pattern = os.path.join(replay_dir, "**", "*.npz")
    files = sorted(glob.glob(pattern, recursive=True))

    if not files:
        print(f"[エラー] {replay_dir} 以下に .npz ファイルが見つかりませんでした。")
        print("")
        print("考えられる原因:")
        print("  1. パスが間違っている")
        print("     → logdir(学習時に --logdir で指定したフォルダ)の中に")
        print("       'replay' という名前のフォルダがないか探してください")
        print("       例: find ~ -name '*.npz' 2>/dev/null | head -5")
        print("  2. まだ学習を実行しておらず、データが存在しない")
        print("     → 短時間だけ学習を回してデータを作る必要があります")
        sys.exit(1)

    return files


def describe_array(key, arr):
    """1つの配列(1つのキー)について詳細を表示する"""
    print(f"\nキー名        : {key}")
    print(f"  形状 (shape) : {arr.shape}")
    print(f"  型 (dtype)   : {arr.dtype}")

    # 数値配列であれば範囲も表示する
    if np.issubdtype(arr.dtype, np.number):
        flat = arr.reshape(-1) if arr.ndim > 0 else arr.reshape(1)
        if flat.size > 0:
            print(f"  最小値       : {flat.min()}")
            print(f"  最大値       : {flat.max()}")
            n_show = min(10, flat.size)
            print(f"  最初の{n_show}個    : {flat[:n_show]}")

    # reward と action は特に詳しく見る(今回の分析で最重要のため)
    if key in ("reward", "action"):
        nonzero = np.count_nonzero(arr)
        print(f"  → ゼロでない値の個数: {nonzero} / {arr.size}")

        if key == "action" and arr.ndim >= 2:
            print(f"  → 2次元目(行動の次元数と思われる): {arr.shape[1]}")
            print(f"  → 各次元ごとの最大値:")
            for i in range(arr.shape[1]):
                col = arr[:, i]
                print(f"      次元{i}: min={col.min()}, max={col.max()}, "
                      f"ゼロ以外={np.count_nonzero(col)}個")


def inspect_file(path):
    print("=" * 70)
    print(f"調査対象ファイル: {path}")
    print("=" * 70)

    data = np.load(path, allow_pickle=True)

    print(f"\n含まれているキーの数: {len(data.files)}")
    print(f"キー一覧: {list(data.files)}")
    print("-" * 70)

    for key in data.files:
        describe_array(key, data[key])

    print("\n" + "=" * 70)
    print("確認が終わりました。この出力全体をコピーして共有してください。")
    print("特に 'action' キーの形状・次元ごとの値が重要です。")
    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使い方: python inspect_replay_data.py /path/to/replay/directory")
        sys.exit(1)

    replay_dir = sys.argv[1]
    episode_files = find_episode_files(replay_dir)

    print(f"見つかったエピソードファイル数: {len(episode_files)}")
    print(f"内訳の最初の1つを詳しく調べます。\n")

    inspect_file(episode_files[0])
