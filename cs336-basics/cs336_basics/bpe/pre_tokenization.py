from __future__ import annotations

from math import ceil
from multiprocessing import Pool
import os
import re
from typing import BinaryIO, Counter, List, Tuple

import regex


BYTES: List[bytes] = [bytes([i]) for i in range(256)]
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def pretokenize_text(text: str, special_tokens: list[str] = None) -> list[tuple[bytes, ...]]:
    pre_tokens = []
    last_end = 0
    if special_tokens and len(special_tokens) > 0:
        special_tokens.sort(key=len, reverse=True)
        for match in re.finditer("|".join(re.escape(st) for st in special_tokens or []), text):
            special_token = match.group()

            for pre_token in regex.finditer(PAT, text[last_end : match.start()]):
                pre_tokens.append(split_bytes(pre_token.group().encode("utf-8")))

            pre_tokens.append((special_token.encode("utf-8"),))
            last_end = match.end()

    for pre_token in regex.finditer(PAT, text[last_end:]):
        pre_tokens.append(split_bytes(pre_token.group().encode("utf-8")))

    return pre_tokens


def pretokenize_file(
    file: BinaryIO,
    num_processes: int,
    chunk_max_size: int,
    split_special_token: bytes,
    special_tokens: list[str]
) -> Counter[Tuple[bytes, ...], int]:
    # 保证划分 token 在特殊 token 列表中
    if split_special_token.decode(encoding="utf-8") not in special_tokens:
        raise ValueError("split_special_token must in special_tokens")

    # 获取文件大小
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    if file_size == 0:
        raise ValueError("file is empty")

    # 计算预计分块数
    expected_chunk_num = ceil(file_size / chunk_max_size)
    # 获取边界
    boundaries = find_chunk_boundaries(
        file=file,
        desired_num_chunks=expected_chunk_num,
        split_special_token=split_special_token
    )

    # 更新进程数
    num_processes = min(num_processes, len(boundaries) - 1)

    def get_chunk_data(start: int, end: int) -> bytes:
        file.seek(start)
        return file.read(end - start)

    try:
        # 并发预处理
        pool = Pool(processes=num_processes)
        async_results = [
            pool.apply_async(
                func=pretokenize_chunk,
                kwds={
                    "chunk_data": get_chunk_data(start, end),
                    "special_tokens": special_tokens
                },
            ) for start, end  in zip(boundaries[:-1], boundaries[1:])
        ]
        pool.close()
        pool.join()

    except Exception as e:
        raise RuntimeError(f"Multiprocessing failed") from e

    final_words_counter: Counter[Tuple[bytes, ...], int] = Counter()
    for result in async_results:
        final_words_counter.update(result.get())
    return final_words_counter


def pretokenize_chunk(
    chunk_data: bytes,
    special_tokens: list[str]
) -> Counter[Tuple[bytes, ...], int]:
    chunk_str = chunk_data.decode(encoding="utf-8")

    counter = Counter()
    last_end = 0

    if special_tokens and len(special_tokens) > 0:
        # 这里要使用 escape 处理原 token 文本串里面的字符, 避免后续被识别为正则表达式的规则
        # 这里使用 "|" 来表示正则表达式规则里面的或
        special_tokens_patten = "|".join(re.escape(token) for token in special_tokens)

        # 使用 finditer 避免一次性读入内存
        for match in regex.finditer(special_tokens_patten, chunk_str):
            pre_special_token_str = chunk_str[last_end:match.start()]
            for pre_token in regex.finditer(PAT, pre_special_token_str):
                pre_token_bytes = pre_token.group(0).encode("utf-8")
                counter[split_bytes(pre_token_bytes)] += 1
                del pre_token_bytes
            last_end = match.end()

    pre_token_text = chunk_str[last_end:]
    for pre_token in regex.finditer(PAT, pre_token_text):
        pre_token_bytes = pre_token.group(0).encode("utf-8")
        counter[split_bytes(pre_token_bytes)] += 1
        del pre_token_bytes

    return counter


def split_bytes(data: bytes) -> Tuple[bytes]:
    return tuple([BYTES[b] for b in data])


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # 把指针定位到距文件末尾 0 字节处, 即文件最末尾
    file.seek(0, os.SEEK_END)
    # # tell() 返回当前指针位置 -> 就是文件字节数
    file_size = file.tell()
    # 回到文件按开头, 默认的 whence 是文件开头
    file.seek(0)

    # 这里可能导致多出分块, 但是本函数原本也不保证返回精准 desired_num_chunks 数量分块
    # file_size = 10, desired_num_chunks = 3 -> chunk_size = 3
    # 切出来是 [0,3) [3,6) [6,9) [9,10)
    chunk_size = file_size // desired_num_chunks

    # 这里 range(desired_num_chunks + 1) 就是考虑到上面多一块的情况了
    # 最后一块更新为文件大小, 这里其实就是保存了所有边界值, 包括 0 这一个开头
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    # 一次读取 4k 字节, 可以理解为缓冲区大小
    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    # 第一个边界是 0, 不用更新; 最后一个是文件结尾, 不用更新
    # 这里是更新边界到特殊 token 处
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        # 到达初始化边界处, 后续向后寻找边界
        file.seek(initial_position)  # Start at boundary guess
        while True:
            # 缓存读取
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # 读取到 EOF, 证明后面也没有划分 token, 直接更新边界为文件结尾
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # 找到当前缓冲区的第一个匹配划分 token 的下标
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                # 更新边界为划分 token 的下标处, 后续处理时可能会有划分 token 在 chunk 开头
                chunk_boundaries[bi] = initial_position + found_at
                break

            # 这个缓冲区没有找到就继续读取
            initial_position += mini_chunk_size

    # 上面的处理可能会导致多个边界重复, 比如倒数第二个边界就很可能一直到文件结尾
    # 这个先做去重再排序其实也是保证边界的可用性
    return sorted(set(chunk_boundaries))


if __name__ == "__main__":
    with open(r"data/TinyStoriesV2-GPT4/TinyStoriesV2-GPT4-valid.txt", "rb") as f:
        pre_token_count1 = pretokenize_file(
            file=f,
            num_processes=8,
            chunk_max_size=500*1024,
            split_special_token=b"<|endoftext|>",
            special_tokens=["<|endoftext|>"]
        )

    with open(r"data/TinyStoriesV2-GPT4/TinyStoriesV2-GPT4-valid.txt", "rb") as f:
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        f.seek(0)

        pre_token_count2 = pretokenize_chunk(
            chunk_data=f.read(file_size - 0),
            special_tokens=["<|endoftext|>"]
        )

    print("=========== parallel=serial ===========")
    print(pre_token_count1 == pre_token_count2)
    print("=======================================")
    print("=========== parallel top-10 ===========")
    [print(key, counts) for key, counts in pre_token_count1.most_common(10)]
    print("=======================================")
    print("============ serial top-10 ============")
    [print(key, counts) for key, counts in pre_token_count2.most_common(10)]
    print("=======================================")
