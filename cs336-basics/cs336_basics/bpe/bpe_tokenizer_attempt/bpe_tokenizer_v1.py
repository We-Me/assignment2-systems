from __future__ import annotations

from collections import defaultdict
import heapq
import pickle
from typing import Iterable, Iterator

from cs336_basics.bpe.pre_tokenization import pretokenize_text


class BPETokenizer(object):

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

        pre_token_list: list[list[bytes]] = []
        pair_index_dict: defaultdict[tuple[bytes, bytes], set[int]] = defaultdict(set)
        for i, key in enumerate(pre_tokens):
            pre_token_list.append(list(key))
            if len(key) > 1:
                for pair in zip(key[:-1], key[1:]):
                    pair_index_dict[pair].add(i)

        merge_pairs: list[tuple[int, tuple[bytes, bytes]]] = []
        for pair in pair_index_dict:
            merge_token_id = self.inv_vocab.get(pair[0] + pair[1], None)
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
                old_pre_token_list = pre_token_list[pre_token_index]
                tmp_pre_token_list: list[bytes] = list(old_pre_token_list)
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
                pre_token_list[pre_token_index] = new_pre_token_list

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
                new_merge_token_id = self.inv_vocab.get(new_pair[0] + new_pair[1], None)
                if new_merge_token_id is not None:
                    heapq.heappush(merge_pairs, (new_merge_token_id, new_pair))

        output: list[int] = []
        for pre_token in pre_token_list:
            for token in pre_token:
                output.append(self.inv_vocab[token])
        return output

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            encode = self.encode(text=text)
            yield from encode

    def decode(self, ids: list[int]) -> str:
        output: bytes = b""
        for id in ids:
            output += self.vocab[id]
        return output.decode("utf-8", errors="replace")
