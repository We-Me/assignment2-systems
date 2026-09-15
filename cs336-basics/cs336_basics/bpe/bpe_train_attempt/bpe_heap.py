from __future__ import annotations

from collections import defaultdict
from typing import Counter, List, Tuple


class BPEHeap(object):
    def __init__(self):
        self.heap_list: List[Tuple[Tuple[bytes, bytes], int]] = []
        self.last_not_leaf = -1

    def not_empty(self):
        return len(self.heap_list) > 0

    def push(self, element: Tuple[Tuple[bytes, bytes], int]) -> None:
        self.heap_list.append(element)
        self.update_last_not_leaf()
        self.up(len(self.heap_list) - 1)

    def pop(self) -> Tuple[Tuple[bytes, bytes], int] | None:
        if len(self.heap_list) == 0:
            return None
        if len(self.heap_list) == 1:
            first = self.heap_list.pop()
            self.update_last_not_leaf()
            return first
        
        first = self.heap_list[0]
        self.heap_list[0] = self.heap_list.pop()
        self.update_last_not_leaf()
        self.down(index=0)
        return first

    def from_counter(self, counter: Counter[Tuple[bytes, bytes]]):
        self.heap_list: List[Tuple[Tuple[bytes, bytes], int]] = list(counter.items())
        self.update_last_not_leaf()

        index = self.last_not_leaf
        while(index > -1):
            self.down(index=index)
            index -= 1

    def from_dict(self, dict: defaultdict[tuple[bytes, bytes], int]):
        self.heap_list: List[Tuple[Tuple[bytes, bytes], int]] = list(dict.items())
        self.update_last_not_leaf()

        index = self.last_not_leaf
        while(index > -1):
            self.down(index=index)
            index -= 1

    def update_last_not_leaf(self):
        self.last_not_leaf = (len(self.heap_list) - 2) // 2

    @staticmethod
    def compare_to(
        c: Tuple[Tuple[bytes, bytes], int], 
        o: Tuple[Tuple[bytes, bytes], int]
    ) -> int:
        (c_bytes1, c_bytes2), c_count = c
        (o_bytes1, o_bytes2), o_count = o
        if c_count > o_count:
            return 1
        elif c_count < o_count:
            return -1
        else:
            if c_bytes1 > o_bytes1:
                return 1
            elif c_bytes1 < o_bytes1:
                return -1

            if c_bytes2 > o_bytes2:
                return 1
            elif c_bytes2 < o_bytes2:
                return -1

            return 0

    def up(self, index: int):
        if len(self.heap_list) < 2:
            return
        if index < 1 or index + 1 > len(self.heap_list):
            return

        parent = (index - 1) // 2
        compare = self.compare_to(self.heap_list[index], self.heap_list[parent])
        if compare > 0:
            tmp = self.heap_list[index]
            self.heap_list[index] = self.heap_list[parent]
            self.heap_list[parent] = tmp
            self.up(parent)

        return

    def down(self, index):
        if len(self.heap_list) < 2:
            return
        if index > self.last_not_leaf or index < 0:
            return

        left_child = 2 * index + 1
        right_child = 2 * index + 2

        if right_child == len(self.heap_list):
            # 只有左孩子
            compare1 = self.compare_to(self.heap_list[index], self.heap_list[left_child])
            if compare1 < 0:
                # 显式左节点最大
                tmp = self.heap_list[index]
                self.heap_list[index] = self.heap_list[left_child]
                self.heap_list[left_child] = tmp
                self.down(left_child)
        else:
            compare1 = self.compare_to(self.heap_list[index], self.heap_list[left_child])
            compare2 = self.compare_to(self.heap_list[index], self.heap_list[right_child])
            compare3 = self.compare_to(self.heap_list[left_child], self.heap_list[right_child])
            if compare1 < 0 and compare2 < 0:
                # 显式当前节点最小
                # 孩子节点相等选左
                if compare3 < 0:
                    tmp = self.heap_list[index]
                    self.heap_list[index] = self.heap_list[right_child]
                    self.heap_list[right_child] = tmp
                    self.down(right_child)
                else:
                    tmp = self.heap_list[index]
                    self.heap_list[index] = self.heap_list[left_child]
                    self.heap_list[left_child] = tmp
                    self.down(left_child)
            elif compare1 < 0:
                # 显式左节点最大
                tmp = self.heap_list[index]
                self.heap_list[index] = self.heap_list[left_child]
                self.heap_list[left_child] = tmp
                self.down(left_child)
            elif compare2 < 0:
                # 显式右节点最大
                tmp = self.heap_list[index]
                self.heap_list[index] = self.heap_list[right_child]
                self.heap_list[right_child] = tmp
                self.down(right_child)

        # 当前节点与孩子比较后已经满足堆序，因此无需继续下沉
        return
    