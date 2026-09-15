from __future__ import annotations

from collections import defaultdict
import time
from typing import Counter, Tuple

from cs336_basics.bpe.pre_tokenization import pretokenize_file
from cs336_basics.bpe.bpe_train_attempt.bpe_heap import BPEHeap


def train_bpe_v1(
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
    print(f"pretokenization: {time.time()-t0:.2f}s", flush=True)


    t1=time.time()
    # 用来记录所有 pair 及其对应数量
    pair_counter: Counter[tuple[bytes, bytes]] = Counter()
    # 用来记录 pair 出现在哪个 pre_token 的哪个位置
    pair_index: defaultdict[
        tuple[bytes, bytes],
        defaultdict[tuple[bytes, ...], list[int]]
    ] = defaultdict(lambda: defaultdict(list))
    for key, count in pre_token_counter.items():
        if len(key) > 1:
            for i, pair in enumerate(zip(key[:-1], key[1:])):
                pair_counter[pair] += count
                pair_index[pair][key].append(i)
    pair_lazy_heap = BPEHeap()
    pair_lazy_heap.from_counter(pair_counter)

    # 初始化合并组
    merges: list[tuple[bytes, bytes]] = []
    while (pair_lazy_heap.not_empty() and len(vocab) < vocab_size):
        chosen = None
        while (pair_lazy_heap.not_empty()):
            pair, count = pair_lazy_heap.pop()
            # 第一个判断是否为前面删除的 pair, 第二个判断是否是当前有效值
            if pair in pair_counter and pair_counter[pair] == count:
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

        # 拷贝一遍因为后续会删除 pair_index 内容, 当 pair 为 chosen 时会导致 for 循环出错
        index_dict_list = [(key, indice.copy()) for key, indice in pair_index[chosen].items()]
        for key, indice in index_dict_list:
            # 这里取出该 pre_token 的数量, 后续更新 tuple 后要写回去
            count = pre_token_counter[key]
            del pre_token_counter[key]

            # 这里是删除当前 pre_token tuple 涉及的统计数
            for i in range(len(key) - 1):
                # 原本可以直接del, 下面分三行来删除是因为直接 del 当前序列存在相同 pair 的话就会重复 del
                pair_index[(key[i], key[i+1])][key].remove(i)
                if len(pair_index[(key[i], key[i+1])][key]) == 0:
                    del pair_index[(key[i], key[i+1])][key]
                # 这个才是可能的一个判断
                if len(pair_index[(key[i], key[i+1])]) == 0:
                    del pair_index[(key[i], key[i+1])]

                # 先把数量减下去, 后续再加回来
                pair_counter[(key[i], key[i+1])] -= count
                if pair_counter[(key[i], key[i+1])] == 0:
                    del pair_counter[(key[i], key[i+1])]

            # 这里若是存在 eee 要合并 ee, 不加以管制会产生随机性
            # 这里通过删去相邻的后一个目的是永远从前面开始合并
            i = 1
            while (i < len(indice)):
                if indice[i] == indice[i-1] + 1:
                    indice = indice[:i] + indice[i+1:]
                else:
                    i += 1

            # 新加一个 list 用来更新 new key
            key_list = list(key)
            # 新建容器用来标记要加回去的内容
            update_pair_counter: Counter[tuple[bytes, bytes]] = Counter()
            for index in indice:
                # 标记合并值
                key_list[index] = new_word
                # 后面遍历 new key 生成的 pair 并不会有这两个合并 pair 影响的前后 pair
                # 然后 update_pair_counter 中也没有这两个, 压入 heap 也就缺少这两个 pair 的有效值
                # heap 中没有有效值就会导致这两个 pair 永远不会被选为合并, 哪怕它的数量很多
                # 判断 (key[index-1], key[index]) in pair_counter 是因为前面如果删到 0 会 del
                # 那么后续也不会有这两个 pair 被选为合并, 因此也不需要更新
                # 加到 update_pair_counter 后删除原 pair_counter 是为了避免后面更新数量加两遍
                if index != 0 and (key[index-1], key[index]) in pair_counter:
                    update_pair_counter[(key[index-1], key[index])] += pair_counter[(key[index-1], key[index])]
                    del pair_counter[(key[index-1], key[index])]
                # 合并的后一个就空了, 用 None 占位
                key_list[index+1] = None
                if index + 2 != len(key) and (key[index+1], key[index+2]) in pair_counter:
                    update_pair_counter[(key[index+1], key[index+2])] += pair_counter[(key[index+1], key[index+2])]
                    del pair_counter[(key[index+1], key[index+2])]

            # 构造 new key
            new_key = []
            for b in key_list:
                if b is not None:
                    new_key.append(b)
            new_key = tuple(new_key)

            # 开始根据 new key 存在的 pair 添加进数量和索引
            for i, pair in enumerate(zip(new_key[:-1], new_key[1:])):
                # 更新索引
                pair_index[pair][new_key].append(i)
                # 原来有的话先加后删, 避免接下来循环重复加
                if pair in pair_counter:
                    update_pair_counter[pair] += (count + pair_counter[pair])
                    del pair_counter[pair]
                else:
                    update_pair_counter[pair] += count

            # 更新
            pair_counter.update(update_pair_counter)
            # 把修改过的 pair 的有效值全部压入 heap
            for em in list(update_pair_counter.items()):
                pair_lazy_heap.push(em)

            # 写回合并了 pair 的 pre_token
            pre_token_counter[new_key] += count

    print(f"merge: {time.time()-t1:.2f}s", flush=True)
    return vocab, merges
