from __future__ import annotations

from collections import OrderedDict, defaultdict
import heapq
import pickle
from typing import Iterable, Iterator

from cs336_basics.bpe.pre_tokenization import pretokenize_text


class BPETokenizer(object):
    CACHE_SIZE: int = 65536

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] = None,
    ):
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.special_tokens = special_tokens or []

        self.pre_token_encode_cache: OrderedDict[tuple[bytes, ...], tuple[int, ...]] = OrderedDict()

    @classmethod
    def from_file(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None
    ) -> BPETokenizer:
        with open(vocab_filepath, "rb") as vf:
            vocab = pickle.load(vf)

        with open(merges_filepath, "rb") as mf:
            merges = pickle.load(mf)

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def encode(self, text: str) -> list[int]:
        if text is None or text == "":
            return []

        pre_tokens: list[tuple[bytes, ...]] = pretokenize_text(text=text, special_tokens=self.special_tokens)
        num_pre_tokens = len(pre_tokens)
        pre_token_index_dict: defaultdict[tuple[bytes, ...], set[int]] = defaultdict(set)
        for i, pre_token in enumerate(pre_tokens):
            pre_token_index_dict[pre_token].add(i)

        pre_token_encode_dict: defaultdict[tuple[bytes, ...], tuple[int, ...]] = defaultdict()
        pre_token_need2encode_orgin: list[tuple[bytes, ...]] = []
        pre_token_need2encode: list[list[bytes]] = []
        for pre_token in pre_token_index_dict.keys():
            if pre_token in self.pre_token_encode_cache:
                pre_token_encode_dict[pre_token] = self.pre_token_encode_cache[pre_token]
            else:
                pre_token_need2encode_orgin.append(pre_token)
                pre_token_need2encode.append(list(pre_token))

        pair_index_dict: defaultdict[tuple[bytes, bytes], set[int]] = defaultdict(set)
        for i, key in enumerate(pre_token_need2encode):
            if len(key) > 1:
                for pair in zip(key[:-1], key[1:]):
                    pair_index_dict[pair].add(i)

        merge_pairs: list[tuple[int, tuple[bytes, bytes]]] = []
        for pair in pair_index_dict:
            merge_token_id = self.inv_vocab.get(b''.join(pair), None)
            if merge_token_id is not None:
                heapq.heappush(merge_pairs, (merge_token_id, pair))

        while len(merge_pairs) > 0:
            merge_token_id, pair = heapq.heappop(merge_pairs)
            if pair not in pair_index_dict:
                continue

            new_word = self.vocab[merge_token_id]
            index_list = list(pair_index_dict[pair])
            new2push: set[tuple[bytes, bytes]] = set()
            for pre_token_index in index_list:
                old_pre_token_list = pre_token_need2encode[pre_token_index]
                tmp_pre_token_list: list[bytes | None] = list(old_pre_token_list)
                i = 0
                while (i < len(old_pre_token_list) - 1):
                    if (old_pre_token_list[i], old_pre_token_list[i + 1]) == pair:
                        tmp_pre_token_list[i] = new_word
                        tmp_pre_token_list[i + 1] = None
                        i += 2
                    else:
                        i += 1

                # 更新 pre_token_list
                new_pre_token_list = []
                for t in tmp_pre_token_list:
                    if t is not None:
                        new_pre_token_list.append(t)
                pre_token_need2encode[pre_token_index] = new_pre_token_list

                diff_pair_dict: defaultdict[tuple[bytes, bytes], int] = defaultdict(int)
                last = len(tmp_pre_token_list) - 2
                for i in range(len(tmp_pre_token_list) - 1):
                    diff_pair_dict[(old_pre_token_list[i], old_pre_token_list[i + 1])] += 1
                    if tmp_pre_token_list[i] == new_word:
                        diff_pair_dict[pair] -= 1
                        # 左侧存在更新
                        if i > 0:
                            diff_pair_dict[(old_pre_token_list[i - 1], old_pre_token_list[i])] -= 1
                            if tmp_pre_token_list[i - 1] is None:
                                diff_pair_dict[(new_word, new_word)] += 1
                                new2push.add((new_word, new_word))
                            else:
                                diff_pair_dict[(old_pre_token_list[i - 1], new_word)] += 1
                                new2push.add((old_pre_token_list[i - 1], new_word))
                        # 右侧存在更新
                        if i < last and tmp_pre_token_list[i + 2] != new_word:
                            diff_pair_dict[(old_pre_token_list[i + 1], old_pre_token_list[i + 2])] -= 1
                            diff_pair_dict[(new_word, tmp_pre_token_list[i + 2])] += 1
                            new2push.add((new_word, tmp_pre_token_list[i + 2]))

                for o_pair, o_count in diff_pair_dict.items():
                    if o_count == 0:
                        pair_index_dict[o_pair].remove(pre_token_index)
                    elif o_count > 0:
                        pair_index_dict[o_pair].add(pre_token_index)
                for o_pair, o_count in diff_pair_dict.items():
                    if len(pair_index_dict[o_pair]) == 0:
                        del pair_index_dict[o_pair]

                del tmp_pre_token_list, old_pre_token_list, diff_pair_dict

            for new_pair in new2push:
                new_merge_token_id = self.inv_vocab.get(b''.join(new_pair), None)
                if new_merge_token_id is not None:
                    heapq.heappush(merge_pairs, (new_merge_token_id, new_pair))

        for need_origin, need_merged in zip(pre_token_need2encode_orgin, pre_token_need2encode):
            pre_token_encode_dict[need_origin] = tuple(self.inv_vocab[b] for b in need_merged)

        all_encoded: list[tuple | None] = [None] * num_pre_tokens
        pre_token_priority: list[tuple[int, tuple[bytes, ...], tuple[int, ...]]] = []
        for pre_token in pre_token_index_dict.keys():
            indice = pre_token_index_dict[pre_token]
            encode = pre_token_encode_dict[pre_token]
            heapq.heappush(pre_token_priority, (len(indice), pre_token, encode))
            for index in indice:
                all_encoded[index] = encode

        while pre_token_priority:
            _, pre_token, encode = heapq.heappop(pre_token_priority)
            if pre_token in self.pre_token_encode_cache:
                self.pre_token_encode_cache.move_to_end(pre_token)
            else:
                self.pre_token_encode_cache[pre_token] = encode

        while len(self.pre_token_encode_cache) > BPETokenizer.CACHE_SIZE:
            self.pre_token_encode_cache.popitem(last=False)

        output: list[int] = []
        for encode in all_encoded:
            output.extend(encode)
        return output

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            encode = self.encode(text=text)
            yield from encode

    def decode(self, ids: list[int]) -> str:
        output: bytes = b''.join(self.vocab[id] for id in ids)
        return output.decode("utf-8", errors="replace")
