from __future__ import annotations

from collections import defaultdict
import time
from typing import Counter, Tuple

from cs336_basics.bpe.pre_tokenization import pretokenize_file
from cs336_basics.bpe.bpe_train_attempt.bpe_heap import BPEHeap


def train_bpe_v2(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    num_processes=8,
    chunk_max_size=500*1024,
    split_special_token: bytes = b'<|endoftext|>',
) -> Tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # 检查正确性
    if split_special_token.decode(encoding="utf-8") not in special_tokens:
        raise ValueError("split_special_token must in special_tokens")

    # 初始化词表
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for special in special_tokens:
        vocab[len(vocab)] = special.encode("utf-8")

    # 预分词计时
    t0=time.time()
    # 获取粗粒度划分计数
    with open(input_path, "rb") as f:
        pre_token_counter: Counter[Tuple[bytes, ...]] = pretokenize_file(
            file=f,
            num_processes=num_processes,
            chunk_max_size=chunk_max_size,
            split_special_token=split_special_token,
            special_tokens=special_tokens,
        )
    # 打印预分词计时
    print(f"pretokenization: {time.time()-t0:.2f}s", flush=True)

    # 合并计时
    t1=time.time()

    # 外部换成 list 可以通过下标来锁定 pre token; 内部换成 list 方便后续合并, 就不用先删后加
    pre_token_count_list: list[tuple[list[bytes], int]] = []
    # 用来记录所有 pair 及其对应数量
    pair_count_dict: defaultdict[tuple[bytes, bytes], int] = defaultdict(int)
    # 用来记录 pair 出现在哪个 pre_token 的哪个位置
    # 这里不再细致到具体 index, 而是上升到 pre token
    pair_index_dict: defaultdict[tuple[bytes, bytes], set[int]] = defaultdict(set)
    # 根据预分词 Counter 进行转换
    for i, (key, count) in enumerate(pre_token_counter.items()):
        pre_token_count_list.append((list(key), count))
        if len(key) > 1:
            for pair in zip(key[:-1], key[1:]):
                pair_count_dict[pair] += count
                pair_index_dict[pair].add(i)
    # 初始化堆
    pair_lazy_heap = BPEHeap()
    pair_lazy_heap.from_dict(pair_count_dict)

    # 初始化合并组
    merges: list[tuple[bytes, bytes]] = []
    while (pair_lazy_heap.not_empty() and len(vocab) < vocab_size):
        chosen = None
        while (pair_lazy_heap.not_empty()):
            pair, count = pair_lazy_heap.pop()
            # 第一个判断是否为前面删除的 pair, 第二个判断是否是当前有效值
            if pair in pair_count_dict and pair_count_dict[pair] == count:
                chosen = pair
                break
        # 没有正确对, 此时 heap 一定为空, 直接进入 while 条件判断退出 while
        if chosen is None:
            continue

        # 添加到 merges
        merges.append(chosen)
        # 新 token
        new_word = chosen[0] + chosen[1]
        # 添加新 token
        vocab[len(vocab)] = new_word

        index_list = list(pair_index_dict[chosen])
        # 更新放循环外, 避免多次向堆中压入数据
        need2update: defaultdict[tuple[bytes, bytes], int] = defaultdict(int)
        for pre_token_index in index_list:
            old_pre_token_list, pre_token_count = pre_token_count_list[pre_token_index]
            # 这里浅拷贝即可
            tmp_pre_token_list: list[bytes] = list(old_pre_token_list)
            i = 0
            while (i < len(old_pre_token_list) - 1):
                if (old_pre_token_list[i], old_pre_token_list[i + 1]) == chosen:
                    tmp_pre_token_list[i] = new_word
                    tmp_pre_token_list[i + 1] = None
                    i += 2
                else:
                    i += 1
            # 更新 pre_token_count_list
            new_pre_token_list = []
            for t in tmp_pre_token_list:
                if t is not None:
                    new_pre_token_list.append(t)
            pre_token_count_list[pre_token_index] = (new_pre_token_list, pre_token_count)

            diff_pair_count_dict: defaultdict[tuple[bytes, bytes], int] = defaultdict(int)
            last = len(tmp_pre_token_list) - 2
            for i in range(len(tmp_pre_token_list) - 1):
                diff_pair_count_dict[(old_pre_token_list[i], old_pre_token_list[i + 1])] += pre_token_count
                if tmp_pre_token_list[i] == new_word:
                    need2update[chosen] -= pre_token_count
                    diff_pair_count_dict[chosen] -= pre_token_count
                    # 左侧存在更新
                    if i > 0:
                        need2update[(old_pre_token_list[i - 1], old_pre_token_list[i])] -= pre_token_count
                        diff_pair_count_dict[(old_pre_token_list[i - 1], old_pre_token_list[i])] -= pre_token_count
                        if tmp_pre_token_list[i - 1] is None:
                            need2update[(new_word, new_word)] += pre_token_count
                            diff_pair_count_dict[(new_word, new_word)] += pre_token_count
                        else:
                            need2update[(old_pre_token_list[i - 1], new_word)] += pre_token_count
                            diff_pair_count_dict[(old_pre_token_list[i - 1], new_word)] += pre_token_count
                    # 右侧存在更新
                    if i < last and tmp_pre_token_list[i + 2] != new_word:
                        need2update[(old_pre_token_list[i + 1], old_pre_token_list[i + 2])] -= pre_token_count
                        diff_pair_count_dict[(old_pre_token_list[i + 1], old_pre_token_list[i + 2])] -= pre_token_count
                        need2update[(new_word, tmp_pre_token_list[i + 2])] += pre_token_count
                        diff_pair_count_dict[(new_word, tmp_pre_token_list[i + 2])] += pre_token_count

            for o_pair, o_count in diff_pair_count_dict.items():
                if o_count == 0:
                    pair_index_dict[o_pair].remove(pre_token_index)
                elif o_count > 0:
                    pair_index_dict[o_pair].add(pre_token_index)
            for o_pair, o_count in diff_pair_count_dict.items():
                if len(pair_index_dict[o_pair]) == 0:
                    del pair_index_dict[o_pair]

            del tmp_pre_token_list, old_pre_token_list, diff_pair_count_dict

        for nu_pair, nu_count in need2update.items():
            if nu_pair not in pair_count_dict:
                pair_count_dict[nu_pair] = nu_count
                pair_lazy_heap.push((nu_pair, nu_count))
            elif pair_count_dict[nu_pair] + nu_count != 0:
                pair_count_dict[nu_pair] += nu_count
                pair_lazy_heap.push((nu_pair, pair_count_dict[nu_pair]))
            else:
                del pair_count_dict[nu_pair]

    # 打印合并计时
    print(f"merge: {time.time()-t1:.2f}s", flush=True)
    return vocab, merges
