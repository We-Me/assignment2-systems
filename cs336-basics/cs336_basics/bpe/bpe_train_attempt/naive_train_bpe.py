from __future__ import annotations

from typing import Counter, Tuple

from cs336_basics.bpe.pre_tokenization import pretokenize_file


def naive_train_bpe(
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

    # 获取粗粒度划分计数
    with open(input_path, "rb") as f:
        pre_token_counter: Counter[Tuple[bytes, ...], int] = pretokenize_file(
            file=f,
            num_processes=num_processes,
            chunk_max_size=chunk_max_size,
            split_special_token=split_special_token,
            special_tokens=special_tokens,
        )

    # 初始化合并组
    merges: list[tuple[bytes, bytes]] = []
    # 循环合并 token 直到目标规模
    while (len(vocab) < vocab_size):
        # 初始化用于统计本轮 pair 数
        pair_counter: Counter[tuple[bytes, bytes], int] = Counter()

        all_one = True
        # 统计 pair 数
        # 假设有 S 个粗粒度划分, 假设平均长度为 Lt, 这里 O(S*Lt)
        for key, count in pre_token_counter.items():
            if len(key) > 1:
                all_one = False
                for pair in zip(key[:-1], key[1:]):
                    pair_counter[pair] += count
        # 所有粗划分都合并为 1 时退出, 避免后续出错
        if all_one:
            break

        # 假设 pair 数为 P, 这里做排序复杂度为 O(PlogP)
        all_pair_counts = pair_counter.most_common()
        _, max_count = all_pair_counts[0]
        # 这里扫描复杂度为 O(P)
        tied = [p for p, c in all_pair_counts if c == max_count]
        # 这里少于 P, 但是复杂度可以看作 O(P)
        chosen = max(tied)

        merges.append(chosen)
        new_word = chosen[0] + chosen[1]
        # dict 内部会保存当前元素数量, O(1)
        vocab[len(vocab)] = new_word

        update_counter: Counter[Tuple[bytes, ...], int] = Counter()
        # 这里和统计 pair 部分逻辑差不多, 但是涉及到列表复制, 因此也可以看作 O(S*Lt*Lt)
        for key, count in pre_token_counter.items():
            new_key = list(key)
            i = 0
            while(i < len(new_key) - 1):
                if (new_key[i], new_key[i+1]) == chosen:
                    # 切片不会越界, 因此最后两个元素合并 new_key[i+2:] 为 []
                    new_key = new_key[:i] + [new_word] + new_key[i+2:]
                i += 1

            update_counter[tuple(new_key)] += count

        pre_token_counter, update_counter = update_counter, pre_token_counter
        del update_counter

    return vocab, merges
