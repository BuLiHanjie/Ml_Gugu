import numpy as np


class Node:
    def __init__(self, value, priority=None):
        self.value = value
        self.min_val, self.max_val = value, value
        self.priority = priority if priority is not None else np.random.rand()
        self.son = [None, None]
        self.father = None
        self.count = 1
        self.total_count = 1
        pass

    def set_left_son(self, v):
        self.son[0] = v

    def set_right_son(self, v):
        self.son[1] = v

    def set_father(self, v):
        self.father = v

    def get_father(self):
        return self.father

    def get_son(self, p):
        return self.son[p]

    def rotate(self):
        father = self.father
        if father is None:
            raise Exception('error in tree rotate')
        p = 0 if father.son[0] == self else 1
        son = self.son[p ^ 1]
        father.son[p] = son
        self.son[p ^ 1] = father
        if father.father is not None:
            index = 0 if father.father.son[0] == father else 1
            father.father.son[index] = self
        self.father = father.father
        if son is not None:
            son.father = father
        father.father = self
        # father.update()
        # self.update()

    def update(self):
        s = self.count
        min_val, max_val = self.value, self.value
        for _s in self.son:
            if _s is None:
                continue
            s += _s.total_count
            min_val = min(min_val, _s.min_val)
            max_val = max(max_val, _s.max_val)
        self.total_count = s
        self.min_val = min_val
        self.max_val = max_val

    def can_delete(self):
        if self.son[0] is not None and self.son[1] is not None:
            return False
        return True

    def delete(self):
        father = self.father
        p = -1
        if father is not None:
            p = 0 if father.son[0] == self else 1
        p1 = -1
        if self.son[0] is not None:
            p1 = 0
        if self.son[1] is not None:
            p1 = 1
        if p != -1:
            father.son[p] = self.son[p1] if p1 != -1 else None
        if p1 != -1:
            self.son[p1].father = father

    def to_string(self):
        return self.value, \
               'father: {}'.format(self.father.value if self.father is not None else None), \
               'priority: {}'.format(self.priority), \
               'count: {}'.format(self.count), \
               'total_count: {}'.format(self.total_count), \
               'min_val: {}'.format(self.min_val), \
               'max_val: {}'.format(self.max_val), \
               'left son: {}'.format(self.son[0].value if self.son[0] is not None else None), \
               'right son: {}'.format(self.son[1].value if self.son[1] is not None else None)


class SegmentTree:

    def __init__(self, ids):
        self.ids = ids
        for i in range(1, len(ids)):
            if ids[i] <= ids[i - 1]:
                self.ids = np.unique(ids)
                break


class Treap:

    def __init__(self):
        self.root = Node(1e10, 10)

    def add(self, value):
        def insert(root, value):
            if root.value == value:
                root.count += 1
                root.total_count += 1
                return
            p = 0 if value < root.value else 1
            if root.son[p] is None:
                root.son[p] = Node(value)
                root.son[p].father = root
                root.total_count += 1
            else:
                insert(root.son[p], value)
            if root.son[p].priority > root.priority:
                # p1, p2 = root, root.son[p]
                # print(p1.to_string(), p2.to_string())
                root.son[p].rotate()
                # print(p1.to_string(), p2.to_string())
                root.update()
                root.father.update()
                if root.father.father is None:
                    self.root = root.father
            else:
                root.son[p].update()
                root.update()

        if self.root is None:
            self.root = Node(value)
        else:
            insert(self.root, value)
            if self.root.father is not None:
                self.root = self.root.father

    def delete(self, value):
        def del_node(root):
            if root.can_delete():
                root.delete()
                del root
                return
            p = 0 if root.son[0].priority > root.son[1].priority else 1
            son = root.son[p]
            son.rotate()
            del_node(root)
            son.update()
            if son.father is None:
                self.root = son

        def del_count(root, value):
            if root.value == value:
                root.count -= 1
                root.total_count -= 1
                if root.count == 0:
                    del_node(root)
                return
            p = 0 if value < root.value else 1
            if root.son[p] is None:
                raise Exception('value {} not is not found in tree'.format(value))
            son = root.son[p]
            del_count(son, value)
            root.update()

        del_count(self.root, value)
        if self.root.father is not None:
            self.root = self.root.father
        if self.root.count == 0:
            self.root = None

    def less_equal_count(self, value):
        def le_count(root, value):
            if root is None or root.min_val > value:
                return 0
            if root.max_val <= value:
                return root.total_count
            s = root.count if root.value <= value else 0
            for _son in root.son:
                s += le_count(_son, value)
            return s

        return le_count(self.root, value)

    def print(self):
        def pr(root):
            print(root.to_string())
            for v in root.son:
                if v is not None:
                    pr(v)

        if self.root is not None:
            pr(self.root)
        else:
            print('empty tree')


if __name__ == '__main__':
    tree = Treap()
    print('=' * 10)
    tree.print()
    tree.add(100)
    print('=' * 10)
    tree.print()
    tree.delete(100)
    print('=' * 10)
    tree.print()
    tree.add(87)
    print('=' * 10)
    tree.print()
    tree.add(83)
    print('=' * 10)
    tree.print()
    tree.delete(87)
    print('=' * 10)
    tree.print()
    tree.add(91)
    tree.delete(83)
    print('=' * 10)
    tree.print()
