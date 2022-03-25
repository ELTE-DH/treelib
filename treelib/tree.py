#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tree structure in `treelib`.

The :class:`Tree` object defines the tree-like structure based on :class:`Node` objects.
A new tree can be created from scratch without any parameter or a shallow/deep copy of another tree.
When deep=True, a deepcopy operation is performed on feeding tree parameter and more memory
is required to create the tree.
"""

import json
import uuid
from copy import deepcopy
from itertools import chain
from functools import partial

from io import StringIO

from .exceptions import *
from .node import Node


class Tree:
    """
    Tree objects are made of Node(s) stored in nodes dictionary.
    """

    #: ROOT, DEPTH, WIDTH, ZIGZAG constants :
    (ROOT, DEPTH, WIDTH, ZIGZAG) = list(range(4))
    node_class = Node

    def __init__(self, tree=None, deep: bool = False, node_class=None, tree_id=None):
        """
        Initiate a new tree or copy another tree with a shallow or deep copy.
        """

        # Initialize self._set_identifier
        if tree_id is None:
            tree_id = str(uuid.uuid1())
        self.tree_id = tree_id

        if node_class is not None:
            assert issubclass(node_class, Node)
            self.node_class = node_class

        #: Dict of nodes in a tree: {node identifier (nid): node_instance}.
        self.nodes = {}

        #: Get or set the node identifier (nid) of the root. This attribute can be accessed and modified
        #:  with ``.`` and ``=`` operator respectively.
        self.root = None

        if tree is not None:
            self.root = tree.root
            for nid, node in tree.nodes.items():
                new_node = deepcopy(node) if deep else node
                self.nodes[nid] = new_node
                if tree.tree_id != self.tree_id:
                    new_node.clone_pointers(tree.tree_id, self.tree_id)

        # Render characters
        self._dt = {
            'ascii': ('|', '|-- ', '+-- '),
            'ascii-ex': ('\u2502', '\u251c\u2500\u2500 ', '\u2514\u2500\u2500 '),
            'ascii-exr': ('\u2502', '\u251c\u2500\u2500 ', '\u2570\u2500\u2500 '),
            'ascii-em': ('\u2551', '\u2560\u2550\u2550 ', '\u255a\u2550\u2550 '),
            'ascii-emv': ('\u2551', '\u255f\u2500\u2500 ', '\u2559\u2500\u2500 '),
            'ascii-emh': ('\u2502', '\u255e\u2550\u2550 ', '\u2558\u2550\u2550 '),
        }

    def _clone(self, tree_id=None, with_tree=False, deep=False):
        """
        Clone current instance, with or without tree.

        Method intended to be overloaded, to avoid rewriting whole "subtree" and "remove_subtree" methods when
        inheriting from Tree.
        >>> class TreeWithComposition(Tree):
        >>>     def __init__(self, tree_description, tree=None, deep=False, tree_id=None):
        >>>         self.tree_description = tree_description
        >>>         super().__init__(tree=tree, deep=deep, tree_id=tree_id)
        >>>
        >>>     def _clone(self, tree_id=None, with_tree=False, deep=False):
        >>>         return TreeWithComposition(
        >>>             tree_id=tree_id,
        >>>             deep=deep,
        >>>             tree=self if with_tree else None,
        >>>             tree_description=self.tree_description
        >>>         )
        >>> my_custom_tree = TreeWithComposition(tree_description='smart tree')
        >>> subtree = my_custom_tree.subtree()
        >>> subtree.tree_description
        'smart tree'
        """
        return self.__class__(tree_id=tree_id, tree=self if with_tree else None, deep=deep)

    def all_nodes(self):
        """
        Returns all nodes in an iterator.
        """
        return self.nodes.values()

    def size(self, level=None):
        """
        Get the number of nodes of the whole tree if @level is not given.
        Otherwise, the total number of nodes at specific level is returned.
        0 is returned if the tree is not deep enough.

        @param level The level number in the tree. It must be greater or equal to 0.
        """
        if level is None:
            return len(self.nodes)
        else:
            return sum(1 for node in self.nodes.values() if self.level(node.nid) == level)

    def _check_nodeid_in_tree(self, nid):
        if nid not in self.nodes:
            raise NodeIDAbsentError(f'Node ({nid}) is not in the tree!')

    def filter_nodes(self, func):
        """
        Filters all nodes by function.

        :param func: is passed one node as an argument and that node is included if function returns true,
        :return: a filter iterator of the node in python 3 or a list of the nodes in python 2.
        """
        return filter(func, self.nodes.values())

    def _get_nid(self, node):
        """
        Get the node ID for the given node or pass node ID (the inverse of get_node, used internally)
        """
        if isinstance(node, self.node_class):
            nid = node.nid
        else:
            nid = node
        return nid

    def get_node(self, nid):
        """
        Get the object of the node with ID of ``nid``.

        An alternative way is using getitem ('[]') operation on the tree. But small difference exists between them:
        ``get_node()`` will return None if ``nid`` is absent, whereas '[]' will raise ``KeyError``.
        """
        return self.nodes.get(nid, None)

    def __getitem__(self, nid):
        """
        Return nodes[nid]
        """
        try:
            return self.nodes[nid]
        except KeyError:
            raise NodeIDAbsentError(f'Node ({nid}) is not in the tree!')

    def __len__(self):
        """
        Return len(nodes)
        """
        return len(self.nodes)

    def __contains__(self, node):
        nid = self._get_nid(node)  # TODO Check node's other properties!
        return nid in self.nodes

    def is_ancestor(self, ancestor, grandchild):
        """
        Check if the @ancestor the preceding nodes of @grandchild.

        :param ancestor: the node identifier (nid) or Node instance
        :param grandchild: the node identifier (nid) or Node instance
        :return: True or False
        """
        ancestor_nid = self._get_nid(ancestor)
        grandchild_nid = self._get_nid(grandchild)

        child = grandchild_nid
        parent = self.nodes[grandchild_nid].predecessor(self.tree_id)
        while parent is not None:
            if parent == ancestor_nid:
                return True
            else:
                child = self.nodes[child].predecessor(self.tree_id)  # TODO do we need this two?
                parent = self.nodes[parent].predecessor(self.tree_id)
        return False

    def parent(self, node, level=-1):
        """
        For a given node or node ID (nid), get parent node object at a given level.
        If no level is provided or level equals -1, the parent node is returned.
        If level equals 0 or nid equals root node.
        """
        nid = self._get_nid(node)
        self._check_nodeid_in_tree(nid)

        ascendant = self.nodes[nid].predecessor(self.tree_id)
        if level == -1:  # parent
            if ascendant is None or ascendant not in self.nodes:
                return None
            else:
                return self.nodes[ascendant]  # parent
        elif level == 0 or nid == self.root:  # root
            return self.nodes[nid]  # root -> root

        descendant = self.nodes[nid]
        if level >= self.level(descendant.nid):
            raise InvalidLevelNumber(f'Descendant level (level {self.level(descendant.nid)}) '
                                     f'must be greater than its parent\'s level (level {level})!')

        ascendant_level = self.level(ascendant)
        # Ascend to the appropriate level
        while ascendant is not None:
            if ascendant_level == level:
                return self.nodes[ascendant]
            else:
                descendant = ascendant
                ascendant = self.nodes[descendant].predecessor
                ascendant_level = self.level(ascendant)

    def children(self, node, lookup_nodes=False):
        """
        Return the children (IDs or nodes) list of node or node ID (nid).
        Empty list is returned if the ``node`` does not exist in the tree
        """
        if node is None:
            raise OSError('First parameter cannot be None!')  # TODO this is wierd!
        nid = self._get_nid(node)
        self._check_nodeid_in_tree(nid)

        children = self.nodes[nid].successors(self.tree_id)
        if lookup_nodes:
            children = [self.nodes[i] for i in children]
        return children

    def siblings(self, node, lookup_nodes=False):
        """
        Return the siblings of given @node or node ID.

        If @node is root or there are no siblings, an empty list is returned.
        """
        nid = self._get_nid(node)
        if nid == self.root:
            return []

        pid = self.parent(nid)
        parents_children_ids = self.children(pid)

        if lookup_nodes:
            siblings = [self.nodes[i] for i in parents_children_ids if i != nid]
        else:
            siblings = [i for i in parents_children_ids if i != nid]

        return siblings

    def level(self, node, filter_fun=lambda x: True):
        """
        Get the node level in this tree.
        The level is an integer starting with '0' at the root.
        In other words, the root lives at level '0';
        @filter params is added to calculate level passing exclusive nodes.
        """
        return sum(1 for _ in self.rsearch(node, filter_fun)) - 1

    def depth(self, node=None, filter_fun=lambda x: True):  # TODO merge with level?
        """
        Get the maximum level of this tree or the level of the given node.

        @param node Node instance or identifier (nid)
        @param filter_fun A function to filter the considered leaves when computing depth
        @return int
        @throw NodeIDAbsentError
        """
        if node is None:
            # Get maximum level of this tree
            ret_level = max(self.level(leaf.nid) for leaf in self.leaves(filter_fun=filter_fun))
        elif filter_fun(node):
            # Get level of the given node
            ret_level = self.level(node)
        else:
            raise NodeIDAbsentError('The given node is fitered out!')

        return ret_level

    def rsearch(self, node, filter_fun=lambda x: True):  # TODO expand_tree and __get? What is the difference?
        """
        Traverse the tree branch along the branch from nid to its ancestors (until root).

        :param node: node or node ID
        :param filter_fun: the function of one variable to act on the :class:`Node` object.
        """
        nid = self._get_nid(node)
        if nid is None:
            return
        self._check_nodeid_in_tree(nid)

        current = nid
        while current is not None:
            if filter_fun(self.nodes[current]):
                yield current
            # subtree() hasn't updated the predecessor
            if self.root != current:
                current = self.nodes[current].predecessor(self.tree_id)
            else:
                current = None

    def expand_tree(self, node=None, mode=DEPTH, filter_fun=lambda x: True, key=lambda x: x, reverse=False):
        """
        Python generator to traverse the tree (or a subtree) with optional node filtering and sorting.

        :param node: Node or node ID (nid) from which tree traversal will start. If None tree root will be used.
        :param mode: Traversal mode, may be either DEPTH, WIDTH or ZIGZAG
        :param filter_fun: the @filter_fun function is performed on Node object during traversing.
            The traversing will NOT visit those nodes (and their subrtree) which does not pass filtering.
        :param key: the @key and @reverse are present to sort nodes at each level.
            If @key is None the function returns nodes in original insertion order.
        :param reverse: if True reverse ordering.
        :return: Node IDs that satisfy the conditions in the defined order.
        :rtype: generator object
        """
        nid = self._get_nid(node)
        if nid is None:
            nid = self.root
        self._check_nodeid_in_tree(nid)

        curr_node = self.nodes[nid]
        if not filter_fun(curr_node):  # subtree filtered out
            return

        yield nid  # yield current node ID  # TODO lookup if needed!
        queue = [self.nodes[i] for i in curr_node.successors(self.tree_id) if filter_fun(self.nodes[i])]
        if mode in {self.DEPTH, self.WIDTH}:
            # Set sort key fun
            key_fun = self._create_sort_fun(key, reverse)

            # Set tree traversal order
            if mode is self.DEPTH:
                order_fun = partial(lambda x, y: chain(x, y))  # depth-first
            else:
                order_fun = partial(lambda x, y: chain(y, x))  # width-first

            queue = list(key_fun(queue))  # sort
            while len(queue) > 0:
                cn = queue[0]
                yield cn.nid  # TODO lookup if needed!

                expansion = list(key_fun(self.nodes[i] for i in cn.successors(self.tree_id)
                                         if filter_fun(self.nodes[i])))  # sort
                queue = list(order_fun(expansion, queue[1:]))  # step

        elif mode == self.ZIGZAG:
            stack_fw = []
            queue.reverse()
            stack = stack_bw = queue
            forward = False
            while len(stack) > 0:
                expansion = [self.nodes[i] for i in stack[0].successors(self.tree_id) if filter_fun(self.nodes[i])]
                yield stack.pop(0).nid  # TODO lookup if needed!
                if forward:
                    expansion.reverse()
                    stack_bw = expansion + stack_bw
                else:
                    stack_fw = expansion + stack_fw
                if len(stack) == 0:
                    forward = not forward
                    if forward:
                        stack = stack_fw
                    else:
                        stack = stack_bw
        else:
            raise ValueError(f'Traversal mode ({mode}) is not supported!')

    def leaves(self, nid=None, filter_fun=lambda x: True):
        """
        Get leaves of the whole tree or a subtree.
        """
        if nid is None:
            node_it = self.nodes.values()  # all nodes
        else:
            node_it = (self.nodes[node] for node in self.expand_tree(nid))  # subtree  # TODO lookup in expand_tree!

        leaves = [node for node in node_it if node.is_leaf(self.tree_id) and filter_fun(node)]

        return leaves

    def paths_to_leaves(self, filter_fun=lambda x: True):
        """
        Use this function to get the identifiers allowing to go from the root node to each leaf.

        :return: a list of list of identifiers, root being not omitted.

        For example:

        .. code-block:: python

            Harry
            |___ Bill
            |___ Jane
            |    |___ Diane
            |         |___ George
            |              |___ Jill
            |         |___ Mary
            |    |___ Mark

        Expected result:

        .. code-block:: python

            [('harry', 'jane', 'diane', 'mary'),
             ('harry', 'jane', 'mark'),
             ('harry', 'jane', 'diane', 'george', 'jill'),
             ('harry', 'bill')]

        """
        res = []

        for leaf in self.leaves():
            node_ids = [nid for nid in self.rsearch(leaf, filter_fun)]
            node_ids.reverse()
            res.append(tuple(node_ids))

        return res

    @staticmethod
    def _create_sort_fun(key, reverse):
        if key is None:
            if reverse:
                key_fun = reversed
            else:
                key_fun = partial(lambda x: x)  # Do not sort at all!
        else:
            key_fun = partial(sorted, key=key, reverse=reverse)
        return key_fun

    # MODIFYING FUNCTIONS ----------------------------------------------------------------------------------------------
    def add_node(self, node, parent=None):
        """
        Add a new node object to the tree and make the parent as the root by default.

        The 'node' parameter refers to an instance of Class::Node.
        """
        if not isinstance(node, self.node_class):
            raise OSError(f'First parameter must be object of {self.node_class} !')

        nid = node.nid
        if nid in self.nodes:
            raise DuplicatedNodeIdError(f'Cannot create node with ID {nid} !')

        pid = self._get_nid(parent)  # None or node identifier (nid) # TODO can this be None?

        if pid is None:  # Adding root node
            if self.root is not None:
                raise MultipleRootError('A tree takes one root merely!')
            else:
                self.root = nid
        elif pid not in self.nodes:
            raise NodeIDAbsentError(f'Parent node ({pid}) is not in the tree!')
        else:  # pid is not None and pid in self.nodes -> Updating non-root node's parent
            self.nodes[pid].update_successors(nid, self.node_class.ADD, tree_id=self.tree_id)

        self.nodes[nid] = node
        self.nodes[nid].set_predecessor(pid, self.tree_id)
        node.set_initial_tree_id(self.tree_id)

    def create_node(self, tag=None, nid=None, parent=None, data=None):
        """
        Create a child node for given @parent node. If node identifier (``nid``) is absent,
         a UUID will be generated automatically.
        """
        node = self.node_class(tag=tag, nid=nid, data=data)
        self.add_node(node, parent)
        return node

    def subtree(self, nid, tree_id=None, deep=False):
        """
        Return a shallow COPY of subtree with nid being the new root.
        If nid is None, return an empty tree.

        This line creates a deep copy of the entire tree.
        """
        st = self._clone(tree_id)
        if nid is None:
            return st  # TODO deepcopy if deep true!
        self._check_nodeid_in_tree(nid)

        st.root = nid
        for node_n in self.expand_tree(nid):
            st.nodes.update({self.nodes[node_n].nid: self.nodes[node_n]})
            # Define nodes parent/children in this tree
            # All pointers are the same as copied tree, except the root
            st[node_n].clone_pointers(self.tree_id, st.tree_id)
            if node_n == nid:
                # Reset root parent for the new tree
                st[node_n].set_predecessor(None, st.tree_id)

        if deep:
            st = Tree(nid, deep=True)

        return st

    def link_past_node(self, nid):
        """
        Delete a node by linking past it.

        For example, if we have `a -> b -> c` and delete node b, we are left
        with `a -> c`.
        """
        self._check_nodeid_in_tree(nid)
        if self.root == nid:
            raise LinkPastRootNodeError('Cannot link past the root node, delete it with remove_node()!')
        # Get the parent of the node we are linking past
        parent_node = self.nodes[self.nodes[nid].predecessor(self.tree_id)]
        # Set the children of the node to the parent
        for child in self.nodes[nid].successors(self.tree_id):
            self.nodes[child].set_predecessor(parent_node.nid, self.tree_id)
        # Link the children to the parent

        for curr_nid in self.nodes[nid].successors(self.tree_id):
            parent_node.update_successors(curr_nid, tree_id=self.tree_id)
        # Delete the node
        parent_node.update_successors(nid, mode=parent_node.DELETE, tree_id=self.tree_id)
        del self.nodes[nid]

    def move_node(self, source, destination):
        """
        Move node @source from its parent to another parent @destination.
        """
        if source not in self.nodes or destination not in self.nodes:
            raise NodeIDAbsentError
        elif self.is_ancestor(source, destination):
            raise LoopError

        parent = self.nodes[source].predecessor(self.tree_id)
        if parent is not None:
            self.nodes[parent].update_successors(source, self.node_class.DELETE, tree_id=self.tree_id)
        if destination is not None:
            self.nodes[destination].update_successors(source, self.node_class.ADD, tree_id=self.tree_id)
        self.nodes[source].set_predecessor(destination, self.tree_id)

    def merge(self, nid, new_tree, deep=False):
        """
        Patch @new_tree on current tree by pasting new_tree root children on current tree @nid node.

        Consider the following tree:
        >>> current = Tree()
        ...
        >>> current.show()
        root
        ├── A
        └── B
        >>> new_tree.show()
        root2
        ├── C
        └── D
            └── D1
        Merging new_tree on B node:
        >>>current.merge('B', new_tree)
        >>>current.show()
        root
        ├── A
        └── B
            ├── C
            └── D
                └── D1

        Note: if current tree is empty and nid is None, the new_tree root will be used as root on current tree.
         In all other cases new_tree root is not pasted.
        """
        if new_tree.root is None:
            return

        if nid is None:
            if self.root is None:
                new_tree_root = new_tree[new_tree.root]
                self.add_node(new_tree_root)
                nid = new_tree.root
            else:
                raise ValueError('Must define "nid" under a root which the new tree is merged!')
        for child_nid in new_tree.children(new_tree.root):
            self.paste(nid=nid, new_tree=new_tree.subtree(child_nid), deep=deep)

    def paste(self, nid, new_tree, deep=False):
        """
        Paste a @new_tree to the original one by linking the root of new tree to given node (nid).
        Add @deep for the deep copy of pasted tree.
        """
        assert isinstance(new_tree, Tree)

        if new_tree.root is None:
            return

        if nid is None:
            raise ValueError('Must define "nid" under which new tree is pasted!')
        elif nid not in self.nodes:
            raise NodeIDAbsentError(f'Node ({nid}) is not in the tree!')

        set_joint = set(new_tree.nodes) & set(self.nodes)  # joint keys
        if len(set_joint) > 0:
            raise ValueError(f'Duplicated nodes {list(map(str, set_joint))} exists.')

        for cid, node in new_tree.nodes.items():
            if deep:
                node = deepcopy(new_tree[node])
            self.nodes.update({cid: node})
            node.clone_pointers(new_tree.tree_id, self.tree_id)

        self.nodes[new_tree.root].set_predecessor(nid, self.tree_id)
        if nid is not None:
            self.nodes[nid].update_successors(new_tree.root, self.node_class.ADD, tree_id=self.tree_id)

    def remove_node(self, nid):
        """
        Remove a node indicated by 'nid' with all its successors. Return the number of removed nodes.
        """
        self._check_nodeid_in_tree(nid)

        parent = self.nodes[nid].predecessor(self.tree_id)

        # Remove node and its children
        removed = list(self.expand_tree(nid))

        for curr_nid in removed:
            if curr_nid == self.root:
                self.root = None
            self.nodes[curr_nid].set_predecessor(None, self.tree_id)
            if curr_nid is not None:
                for cid in self.nodes[curr_nid].successors(self.tree_id):
                    self.nodes[curr_nid].update_successors(cid, self.node_class.DELETE, tree_id=self.tree_id)

        # Update parent info
        if parent is not None:
            self.nodes[parent].update_successors(nid, self.node_class.DELETE, tree_id=self.tree_id)
        self.nodes[nid].set_predecessor(None, self.tree_id)

        for curr_nid in removed:
            self.nodes.pop(curr_nid)
        return len(removed)

    def remove_subtree(self, nid, tree_id=None):
        """
        Get a subtree with ``nid`` being the root. If nid is None, an empty tree is returned.

        For the original tree, this method is similar to`remove_node(self,nid)`,
         because given node and its children are removed from the original tree in both methods.
        For the returned value and performance, these two methods are
        different:

            * `remove_node` returns the number of deleted nodes;
            * `remove_subtree` returns a subtree of deleted nodes;

        You are always suggested to use `remove_node` if your only to delete nodes from a tree,
         as the other one need memory allocation to store the new tree.

        :return: a :class:`Tree` object.
        """
        st = self._clone(tree_id)
        if nid is None:
            return st

        self._check_nodeid_in_tree(nid)
        st.root = nid

        # In original tree, the removed nid will be unreferenced from its parents children
        parent = self.nodes[nid].predecessor(self.tree_id)

        removed = list(self.expand_tree(nid))
        for id_ in removed:
            if id_ == self.root:
                self.root = None
            st.nodes.update({id_: self.nodes.pop(id_)})
            st[id_].clone_pointers(self.tree_id, st.tree_id)
            st[id_].reset_pointers(self.tree_id)
            if id_ == nid:
                st[id_].set_predecessor(None, st.tree_id)
        if parent is not None:
            self.nodes[parent].update_successors(nid, self.node_class.DELETE, tree_id=self.tree_id)
        return st

    def update_node(self, nid_to_update, **attrs):
        """
        Update node's attributes.

        :param nid_to_update: the identifier (nid) of modified node
        :param attrs: attribute pairs recognized by Node object
        :return: None
        """
        self._check_nodeid_in_tree(nid_to_update)
        cn = self.nodes[nid_to_update]

        if 'nid' in attrs:  # However, identifier (nid) cannot be None it is better to check containment before poping
            identifier_val = attrs.pop('nid')
            # Updating node id meets following constraints:
            # * Update node identifier (nid) property
            cn = self.nodes.pop(nid_to_update)
            setattr(cn, 'nid', identifier_val)
            # * Update tree registration of var nodes
            self.nodes[identifier_val] = cn

            # * Update parent's followers
            pred_id = cn.predecessor(self.tree_id)
            if pred_id is not None:
                self.nodes[pred_id].update_successors(nid_to_update, self.node_class.REPLACE, identifier_val,
                                                      tree_id=self.tree_id)

            # * Update children's parents
            for fp in cn.successors(self.tree_id):
                self.nodes[fp].set_predecessor(identifier_val, self.tree_id)

            # * Update tree root if necessary
            if self.root == nid_to_update:
                self.root = identifier_val

        for attr, val in attrs.items():
            setattr(cn, attr, val)

    # PRINT RELATED FUNCTIONS ------------------------------------------------------------------------------------------
    def __get(self, nid, level, filter_fun, key, reverse, line_type):
        # Set sort key and reversing if needed
        key_fun = self._create_sort_fun(key, reverse)

        # Set line types
        line_elems = self._dt.get(line_type)
        if line_elems is None:
            raise ValueError(f'Undefined line type ({line_type})! Must choose from {set(self._dt)}!')

        return self.__get_iter(nid, level, filter_fun, key_fun, *line_elems, [])

    def __get_iter(self, nid, level, filter_fun, sort_fun, dt_vertical_line, dt_line_box, dt_line_corner, is_last):
        if nid is None:
            nid = self.root

        node = self.nodes[nid]

        if level == self.ROOT:
            yield '', node
        else:
            lines = []
            for curr_is_last in is_last[0:-1]:
                if curr_is_last:
                    line = ' ' * 4
                else:
                    line = dt_vertical_line + ' ' * 3
                lines.append(line)

            if is_last[-1]:
                lines.append(dt_line_corner)
            else:
                lines.append(dt_line_box)

            yield ''.join(lines), node

        if filter_fun(node):
            children = [self.nodes[i] for i in node.successors(self.tree_id) if filter_fun(self.nodes[i])]
            idxlast = len(children) - 1

            for idx, child in enumerate(sort_fun(children)):
                is_last.append(idx == idxlast)
                yield from self.__get_iter(child.nid, level + 1, filter_fun, sort_fun,
                                           dt_vertical_line, dt_line_box, dt_line_corner, is_last)
                is_last.pop()

    def __print_backend(self, nid=None, level=ROOT, filter_fun=lambda x: True, key=lambda x: x, reverse=False,
                        line_type='ascii-ex', get_label_fun=lambda node: node.tag):
        """
        Another implementation of printing tree using Stack Print tree structure in hierarchy style.
        The @key @reverse is present to sort node at each level.

        For example:

        .. code-block:: bash

            Root
            |___ C01
            |    |___ C11
            |         |___ C111
            |         |___ C112
            |___ C02
            |___ C03
            |    |___ C31

        A more elegant way to achieve this function using Stack structure, for constructing the Nodes Stack
         push and pop nodes with additional level info.
        """
        # Iter with func
        for pre, node in self.__get(nid, level, filter_fun, key, reverse, line_type):
            label = get_label_fun(node)
            yield f'{pre}{label}'

    def __str__(self):
        return self.show()

    def show(self, nid=None, level=ROOT, filter_fun=lambda x: True, key=lambda x: x, reverse=False,
             line_type='ascii-ex', get_label_fun=lambda node: node.tag):
        """
        Print the tree structure in hierarchy style.

        You have three ways to output your tree data, i.e., stdout with ``show()``,
        plain text file with ``save2file()``, and json string with ``to_json()``. The
        former two use the same backend to generate a string of tree structure in a
        text graph.

        You can also specify the ``line_type`` parameter, such as 'ascii' (default), 'ascii-ex',
         'ascii-exr', 'ascii-em', 'ascii-emv', 'ascii-emh') to the change graphical form.

        :param nid: the reference node to start expanding.
        :param level: the node level in the tree (root as level 0).
        :param filter_fun: the function of one variable to act on the :class:`Node` object.
            When this parameter is specified, the traversing will not continue to following
            children of node whose condition does not pass the filter.
        :param key: the ``key`` param for sorting :class:`Node` objects in the same level.
        :param reverse: the ``reverse`` param for sorting :class:`Node` objects in the same level.
        :param line_type:
        :param get_label_fun: A function to define how to print labels
        :return: None
        """
        try:
            reader = self.__print_backend(nid, level, filter_fun, key, reverse, line_type, get_label_fun)
        except NodeIDAbsentError:
            reader = ['Tree is empty']  # TODO what to do here?

        return '\n'.join(reader) + '\n'

    def save2file(self, filename, nid=None, level=ROOT, filter_fun=lambda x: True, key=lambda x: x, reverse=False,
                  line_type='ascii-ex', get_label_fun=lambda node: node.tag):
        """
        Save the tree into file for offline analysis.
        """
        with open(filename, 'w', encoding='UTF-8') as fh:
            fh.writelines(f'{line}\n' for line in
                          self.__print_backend(nid, level, filter_fun, key, reverse, line_type, get_label_fun))

    def to_dict(self, nid=None, key=lambda x: x, sort=True, reverse=False, with_data=False):
        """
        Transform the whole tree into a dict.
        """

        if nid is None:
            nid = self.root
        ntag = self.nodes[nid].tag
        tree_dict = {ntag: {'children': []}}
        if with_data:
            tree_dict[ntag]['data'] = self.nodes[nid].data

        queue = [self.nodes[i] for i in self.nodes[nid].successors(self.tree_id)]
        if sort:
            queue.sort(key=key, reverse=reverse)

        for elem in queue:
            dict_form = self.to_dict(elem.nid, with_data=with_data, sort=sort, reverse=reverse)
            tree_dict[ntag]['children'].append(dict_form)
        if len(tree_dict[ntag]['children']) == 0:
            tree_dict = self.nodes[nid].tag if not with_data else {ntag: {'data': self.nodes[nid].data}}
        return tree_dict

    def to_json(self, with_data=False, sort=True, reverse=False):
        """
        To format the tree in JSON format.
        """
        return json.dumps(self.to_dict(with_data=with_data, sort=sort, reverse=reverse))

    def to_graphviz(self, filename=None, shape='circle', graph='digraph'):
        """
        Exports the tree in the dot format of the graphviz software.
        """
        nodes, connections = [], []
        if len(self.nodes) > 0:

            for n in self.expand_tree(mode=self.WIDTH):
                nid = self.nodes[n].nid
                state = f'"{nid}" [label="{self.nodes[n].tag}", shape={shape}]'
                nodes.append(state)

                for cid in self.children(nid):
                    connections.append(f'"{nid}" -> "{cid}"')

        # Write nodes and connections to dot format
        is_plain_file = filename is not None
        if is_plain_file:
            f = open(filename, 'w', encoding='UTF-8')
        else:
            f = StringIO()

        f.write(f'{graph} tree {{\n')
        f.writelines(f'\t{n}\n' for n in nodes)

        if len(connections) > 0:
            f.write('\n')

        f.writelines(f'\t{c}\n' for c in connections)
        f.write('}')

        if not is_plain_file:
            print(f.getvalue())
        else:
            f.close()
