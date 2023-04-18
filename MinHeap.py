class MinHeap:
    """
    Modified from FIT1008

    Min Heap implementation using a Python in-built list.
    Parent node's index is (child node's index - 1) // 2.
    Left child's index = 2 * node_idx + 1, right child index = 2 * node_idx + 2

    """

    def __init__(self):
        self.array = []

    def print_heap(self):
        print("\n")
        for vertex in self.array:
            print(vertex)

    def rise(self, added_item_idx: int):
        """
        Rise the added item to its correct position
        :param added_item_idx: The item's index in the heap array that might need to rise
        :return:
        :complexity: O(log N * comp), where N is the number of nodes in the heap and comp is the cost of comparison of items

        """

        added_item = self.array[added_item_idx]
        while added_item_idx >0 and self.array[added_item_idx].distance <self.array[(added_item_idx - 1) // 2].distance:
            # swap added element with its parent if added element is smaller than its parent node
            self.array[added_item_idx], self.array[(added_item_idx - 1) // 2] = self.array[(added_item_idx - 1) // 2], \
                                                                                self.array[added_item_idx]
            # also update index(location in heap array)
            self.array[added_item_idx].index, self.array[(added_item_idx - 1) // 2].index = self.array[(added_item_idx - 1) // 2].index, self.array[
                                                                                                added_item_idx].index
            # update swapped element's index
            added_item_idx = (added_item_idx - 1) // 2

        self.array[added_item_idx] = added_item

    def sink(self, outofplace_item_idx):
        """
        Sink the out-of-place item to its correct position
        :complexity: O(log N * comp), where N is the number of nodes in the heap and comp is the cost of comparison of items

        """
        # save the item
        outofplace_item = self.array[outofplace_item_idx]
        saved_smallestchild_index = int()

        # get the final position for the outofplace item
        while 2 * outofplace_item_idx + 2 <= len(self.array):  # while there is child nodes to check
            smallest_idx = self.smallest_child(outofplace_item_idx)
            if self.array[smallest_idx].distance >= outofplace_item.distance:
                break

            saved_oop_index = outofplace_item_idx # 0 4
            saved_smallestchild_index = smallest_idx # 1

            self.array[outofplace_item_idx] = self.array[smallest_idx]  # rise the smallest child
            # update index of the child that rises
            self.array[outofplace_item_idx].index = saved_oop_index

            outofplace_item_idx = smallest_idx

            # place outofplace item into hole
        self.array[outofplace_item_idx] = outofplace_item
        # update index of outofplace item
        self.array[outofplace_item_idx].index = saved_smallestchild_index

    def smallest_child(self, item_idx):
        """
        Returns the index of item's child that has the smaller value.
        :pre: 1 <= item_idx <= len(self.array) // 2 - 1
        :complexity: O(comp), where comp is the cost of comparing the child nodes of the item
        """
        left_idx = 2 * item_idx + 1
        right_idx = 2 * item_idx + 2
        if left_idx == len(self.array) - 1 or \
                self.array[left_idx].distance < self.array[right_idx].distance:
            return left_idx
        else:
            return right_idx

    def add(self, item):
        """
        Add an item into the heap, then rises the item to its correct postion
        :complexity: O(log N * comp), where N is the number of nodes in the heap and comp is the cost of comparison of items
        """
        self.array.append(item)
        self.rise(len(self.array) - 1)

    def getMin(self):
        """
        Replaces minimum item with last item, then sinks last item into its correct position, returns minimum item.
        :complexity: O(log N * comp), where N is the number of nodes in the heap and comp is the cost of comparison of items
        """
        if len(self.array) == 0:
            return None
        elif len(self.array) == 1:
            return self.array.pop()
        else:
            # save minimum item
            min_item = self.array[0]

            # put last item into minimum item's place
            last_item = self.array.pop()
            last_item.index = 0
            self.array[0] = last_item

            # last item may be out of place, so sink it to correct position
            self.sink(0)

            return min_item