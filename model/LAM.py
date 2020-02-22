import numpy as np
import math
class Tree(object):
    def __init__(self):
        self.parent = None
        self.num_children = 0
        self.children = list()

    def add_child(self,child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)

    def size(self):
        if getattr(self,'_size'):
            return self._size
        count = 1
        for i in range(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def depth(self):
        if getattr(self,'_depth'):
            return self._depth
        count = 0
        if self.num_children>0:
            for i in range(self.num_children):
                child_depth = self.children[i].depth()
                if child_depth>count:
                    count = child_depth
            count += 1
        self.depth = count
        return self._depth

    def __iter__(self):
        yield self
        for c in self.children:
            for x in c:
                yield x

def head_to_tree(head, tokens, len_, prune, subj_pos, obj_pos):
    '''
    Convert a sequence of head indexes into a tree object.将头部索引序列转换为树对象
    '''
    tokens = tokens[:len_].tolist()
    head = head[:len_].tolist()
    root = None
    if prune < 0:
        nodes = [Tree() for _ in head]
        for i in range(len(nodes)):
            h = head[i] #表示此node的头节点
            nodes[i].idx = i
            nodes[i].dist = -1
            if h == 0:
                root = nodes[i]
            else:
                nodes[h-1].add_child(nodes[i])
    else:
        # find dependency path\
        subj_pos = [i for i in range(len_) if subj_pos[i] == 0]  # 找到subj_pos的下标
        obj_pos = [i for i in range(len_) if obj_pos[i] == 0]  # 找到obj_pos的下标
        cas = None #存放subj\obj的每个pos的公共祖先，交集
        subj_ancestors = set(subj_pos) #存放subj的所有pos的祖先

        for s in subj_pos:
            h = head[s]  # 前者关系是h
            tmp = [s] #暂存当前pos祖先
            while h > 0 and len(tmp) <= len_:  # 一直寻找前者关系直到前者关系是0，root
                tmp += [h - 1]
                subj_ancestors.add(h - 1)
                h = head[h - 1]

            if cas is None:
                cas = set(tmp)  #
            else:
                cas.intersection_update(tmp)  # 更新一个集合，使其与另一个集合相交

        obj_ancestors = set(obj_pos)
        for o in obj_pos:
            h = head[o]
            tmp = [o]
            while h > 0 and len(tmp) <= len_:
                tmp += [h - 1]
                obj_ancestors.add(h - 1)
                h = head[h - 1]
            cas.intersection_update(tmp)

        # find lowest common ancestor寻找最低共同祖先
        if len(cas) == 1:
            lca = list(cas)[0]
        else:
            child_count = {k: 0 for k in cas}
            for ca in cas:
                if head[ca] > 0 and head[ca] - 1 in cas:
                    child_count[head[ca] - 1] += 1

            # the LCA has no child in the CA set
            for ca in cas:
                if child_count[ca] == 0:
                    lca = ca
                    break

        path_nodes = subj_ancestors.union(obj_ancestors).difference(cas)  # union：并集、difference：表示差集，路径节点为subj\obj的所有祖先，出去共同祖先+上最低祖先
        try:
            path_nodes.add(lca)
        except:
            print('a')

        # compute distance to path_nodes
        dist = [-1 if i not in path_nodes else 0 for i in range(len_)]

        for i in range(len_):  # 遍历每一个节点，对距离小于<0的节点，即不在path_node上的节点进行操作，进一步计算获得该node的dist
            if dist[i] < 0:
                stack = [i]#存放当前节点的祖先（路径），知道寻找到的祖先已经是根节点（与path无关了），或者祖先出现在path中（与path有关）
                while stack[-1] >= 0 and stack[-1] not in path_nodes and len(stack)<=len_:
                    stack.append(head[stack[-1]] - 1) #此处报错

                if stack[-1] in path_nodes:#祖先出现在path中（与path有关）
                    for d, j in enumerate(reversed(stack)):  # reversed返回给定序列值的反向迭代器
                        dist[j] = d
                else:  # 没有到path_node的路径
                    for j in stack:#遍历stack中每个节点
                        if j >= 0 and dist[j] < 0:
                            dist[j] = int(1e4)  # aka infinity无穷远

        highest_node = lca
        nodes = [Tree() if dist[i] <= prune else None for i in range(len_)]  # 为到root的dist小于等于prune的节点定义Tree，否则为None

        for i in range(len(nodes)):  # 为每个树节点添加parent和child
            if nodes[i] is None:
                continue
            h = head[i]  # 找head.parent
            nodes[i].idx = i  # 此节点的下标
            nodes[i].dist = dist[i]  # 此节点到path_node的距离
            if h > 0 and i != highest_node:  # 两个条件都是保证此节点不是根节点，
                assert nodes[h - 1] is not None
                nodes[h - 1].add_child(nodes[i])  # 为先节点增加child，本节点添加parent

        root = nodes[highest_node]

    assert root is not None
    return root


def tree_to_adj(sent_len, tree, directed=True, self_loop=False):#是否有向，是否自循环
    #Convert a tree object to an (numpy) adjacency matrix.
    ret = np.zeros((sent_len,sent_len),dtype=np.float32)
    adj = ret
    k = 1
    queue = [tree] #存放依次遍历的节点
    idx = []
    while len(queue) > 0:
        t, queue = queue[0], queue[1:] #t获取当前queue的第一个节点
        idx += [t.idx]
        for c in t.children:#定义adj矩阵中有联系的节点，元素定义为1（搜寻当前节点的子节点，及相邻节点）
            ret = get_adjElement(t, c, ret, k)
        queue += t.children #将当前节点的子节点放入queue中
    if not directed:
        ret = ret + ret.T
    if self_loop:
        for i in idx:
            ret[i,i] = 1
    return ret

#递归find节点高度
def get_adjElement(rootNode, otherNode , ret, k):#遍历当前节点下的所有子节点(otherNode表示)，k表示rootNode和otherNode之间的距离
    adj = ret
    adj[rootNode.idx, otherNode.idx] = 1 / math.e**(k-1)
    if otherNode.num_children != 0:#otherNode是叶子节点
        k = k+1
        for c in otherNode.children:
            adj = get_adjElement(rootNode, c, adj, k)

    return adj




def tree_to_dist(sent_len, tree):
    ret = -1 * np.ones(sent_len, dtype=np.int64)

    for node in tree:
        ret[node.idx] = node.dist

    return ret