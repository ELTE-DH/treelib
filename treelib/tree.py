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

        # Initialize self.tree_id
        if tree_id is None:
            tree_id = str(uuid.uuid1())
        self.tree_id = tree_id

        if node_class is not None:
            if not issubclass(node_class, Node):
                raise TypeError('node_class should be type of Node or sublcass of Node !')
            self.node_class = node_class

        #: Dict of nodes in a tree: {node ID (nid): node_instance}.
        self.nodes = {}

        #: Get or set the node ID (nid) of the root. This attribute can be accessed and modified
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

    # HELPER FUNCTIONS -------------------------------------------------------------------------------------------------
    def _clone(self, tree_id=None, with_tree=False, deep=False):
        """
        Clone current instance, with or without tree.

        Method intended to be overloaded, to avoid rewriting whole ``subtree`` and ``pop_subtree`` methods when
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

    def _get_nid(self, node):
        """
        Get the node ID (nid) for the given Node instance or node ID (the inverse of ``get_node``, used internally)
        """
        if isinstance(node, self.node_class):
            nid = node.nid
        else:
            nid = node

        if nid not in self.nodes:
            raise NodeIDAbsentError(f'Node ({nid}) is not in the tree!')

        return nid

    def _get_node(self, node):
        """
        Get the Node instance for the given Node instance or node ID (similar to ``get_node``, used internally)
        """
        if isinstance(node, self.node_class):
            curr_node = self.nodes.get(node.nid)  # Node is given as parameter and is not found in the tree
            if curr_node != node:
                raise NodeIDAbsentError(f'Node ({node}) is not in the tree!')
        else:
            curr_node = self.nodes.get(node)  # Node ID (nid) is given as parameter
            if curr_node is None:
                raise NodeIDAbsentError(f'Node ({node}) is not in the tree!')

        return curr_node

    # SIMPLE READER FUNCTIONS ------------------------------------------------------------------------------------------
    def size(self, level=None):
        """
        Get the number of nodes in the whole tree if ``level`` is not given.
        Otherwise, the total number of nodes at specific level is returned.
        0 is returned if too high level is specified and the tree is not deep enough.

        :param level: The level number in the tree. It must be greater or equal to 0, None to return len(tree).
        """
        if level is None:
            return len(self.nodes)
        elif level == 0:
            return 1  # On the root level it is trivially only one node
        else:
            return sum(1 for node in self.nodes.values() if self.level(node.nid) == level)

    def __getitem__(self, nid):
        """
        Return a Node instance for a node ID (nid) if the tree contains it
        """
        try:
            return self.nodes[nid]
        except KeyError:
            raise NodeIDAbsentError(f'Node ({nid}) is not in the tree!')

    def __len__(self):
        """
        Return the number of nodes (node IDs (nid)) in the tree
        """
        return len(self.nodes)

    def __contains__(self, node):
        if isinstance(node, self.node_class):
            nid = node.nid
            return self.nodes.get(nid) == node  # Only True if Node instances are equal, node ID (nid) is not enough!
        else:
            return node in self.nodes  # Check if node ID (nid) is in the tree

    def is_ancestor(self, ancestor, child):
        """
        Check if the ``ancestor`` the preceding nodes of ``child``.

        :param ancestor: the node ID (nid) or Node instance
        :param child: the node ID (nid) or Node instance
        :return: True or False
        """
        ancestor_node = self._get_node(ancestor)
        ancestor_nid = ancestor_node.nid
        child_node = self._get_node(child)

        parent_nid = child_node.predecessor(self.tree_id)
        while parent_nid is not None:  # If parent is None we are at the root node
            parent_node = self.nodes[parent_nid]
            if parent_nid == ancestor_nid and parent_node == ancestor_node:
                return True
            else:
                parent_nid = parent_node.predecessor(self.tree_id)  # The parent of the parent

        return False

    def level(self, node, filter_fun=lambda x: True):  # TODO demote level to private function!
        """
        Get the Node instance or node ID level in this tree.
        The level is an integer starting with '0' at the root.
        In other words, the root lives at level '0';
        ``filter_fun`` param is added to calculate level only if the given node is not in a fitered subtree.
        """
        return sum(1 for _ in self.busearch(node, filter_fun)) - 1

    def depth(self, node=None, filter_fun=lambda x: True):
        """
        Get the maximum level of this tree or the level of the given Node instance or node ID.

        :param node: Node instance or identifier (nid)
        :param filter_fun: A function to filter the subtrees when computing depth for leaves
        :return int:
        :throw NodeIDAbsentError:
        """
        if node is None:  # Get the maximum level of this tree
            ret_level = max(self.level(leaf.nid, filter_fun=filter_fun) for leaf in self.leaves())
        else:  # Get level of the given node
            ret_level = self.level(node, filter_fun=filter_fun)

        return ret_level

    # Node returing READER FUNCTIONS -----------------------------------------------------------------------------------
    # TODO from here may be implement deep copy of returned elems? Maybe Lookup too?
    def all_nodes(self):
        """
        Returns all Node instances in an iterator.
        """
        return self.nodes.values()

    def get_node(self, nid):
        """
        Get the the Node instance with node ID ``nid``.

        An alternative way is using ``__getitem__`` ('[]') operation on the tree.
        But ``get_node()`` will return None if ``nid`` is absent, whereas '[]' will raise ``KeyError``.
        """
        return self.nodes.get(nid)

    def filter_nodes(self, func):
        """
        Filters all Node instances by the given function.

        :param func: All Node instances are passed and those will be included where the function returns True.
        :return: a filter iterator of the node
        """
        return filter(func, self.nodes.values())

    def parent(self, node, level=-1):
        """
        For a given Node instance or node ID (nid), get parent Node instance at a given level.
        If no level is provided or level equals -1, the parent Node instance is returned.
        If level equals 0 the root Node instance is returned.
        if the given Node instance or node ID (nid) equals root node None is returned.
        NodeIDAbsentError exception is thrown if the ``node`` does not exist in the tree.
        """
        nid = self._get_nid(node)

        if nid == self.root:  # Root node for every level -> None
            return None
        elif level == 0:  # Root level of non-root element
            return self.nodes[self.root]  # Root Node instance

        tree_id = self.tree_id
        ancestor = self.nodes[nid].predecessor(tree_id)  # Direct parent nid (None for root node)
        if level == -1:  # Direct parent is required
            return self.nodes[ancestor]  # Parent Node instance (root node where parent is None is already handled)
        elif level >= self.level(nid):
            raise InvalidLevelNumber(f'The given node\'s level ({self.level(nid)}) must be greater than its '
                                     f'parent\'s level (level {level})!')

        # Ascend to the appropriate level
        while ancestor is not None:
            ascendant_level = self.level(ancestor)
            if ascendant_level == level:
                return self.nodes[ancestor]
            else:
                ancestor = self.nodes[ancestor].predecessor(tree_id)  # Get the parent of the current node

    def children(self, node, filter_fun=lambda x: True, lookup_nodes=False):
        """
        Return the direct children (IDs or Node instance) list of the given Node instance or node ID (nid).
        NodeIDAbsentError exception is thrown if the ``node`` does not exist in the tree.
        """
        nid = self._get_nid(node)

        for child_nid in self.nodes[nid].successors(self.tree_id):  # TODO node or nid as prefered parameter?
            child_node = self.nodes[child_nid]
            if filter_fun(child_node):
                if lookup_nodes:
                    yield child_node
                else:
                    yield child_nid

    def siblings(self, node, lookup_nodes=False):
        """
        Return the siblings of given ``node`` or node ID.
        If ``node`` is root or there are no siblings, an empty list is returned.
        NodeIDAbsentError exception is thrown if the ``node`` does not exist in the tree.
        """
        nid = self._get_nid(node)
        if nid == self.root:
            return []

        direct_parent_id = self.nodes[nid].predecessor(self.tree_id)
        parents_children_ids = self.nodes[direct_parent_id].successors(self.tree_id)
        siblings_it = (i for i in parents_children_ids if i != nid)

        if lookup_nodes:
            siblings = [self.nodes[i] for i in siblings_it]
        else:
            siblings = list(siblings_it)

        return siblings

    def leaves(self, node=None, filter_fun=lambda x: True):
        """
        Get leaves of the whole tree (if node is None) or a subtree (if node is a node ID or Node instance).
        If fiter_fun is set leaves in the filtered subtree are not considered.
        """
        if node is not None:  # Leaves for a specific subtree
            nid = self._get_nid(node)
        else:
            nid = self.root  # All leaves

        subtree_node_it = (self.nodes[node] for node in self.expand_tree(nid))  # TODO lookup and filter in expand_tree!

        leaves = [node for node in subtree_node_it if node.is_leaf(self.tree_id) and filter_fun(node)]

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

            [('harry', 'bill'),
             ('harry', 'jane', 'diane', 'george', 'jill'),
             ('harry', 'jane', 'diane', 'mary'),
             ('harry', 'jane', 'mark'),
             ]

        """
        res = []

        for leaf in self.leaves():
            node_ids = [nid for nid in self.busearch(leaf, filter_fun)]
            node_ids.reverse()
            res.append(tuple(node_ids))

        return res

    def busearch(self, node, filter_fun=lambda x: True, lookup_node=False):
        """
        Traverse the tree branch bottom-up along the branch from nid to its ancestors (until root).

        :param node: node or node ID
        :param filter_fun: the function of one variable to act on the :class:`Node` object.
        :param lookup_node: return Node instances or node IDs (nids)
        """
        # TODO expand_tree and __get? What is the difference?
        if node is None:  # We are at root there is no way up
            return

        if lookup_node:
            lookup_node_fun = partial(lambda x: x)
        else:
            lookup_node_fun = partial(lambda x: x.nid)

        current_node = self._get_node(node)
        current_nid = current_node.nid
        while current_nid is not None:
            if filter_fun(current_node):
                yield lookup_node_fun(current_node)
            if self.root != current_nid:
                current_nid = current_node.predecessor(self.tree_id)  # Parent of current node
                current_node = self.nodes[current_nid]
            else:
                current_nid = None

    def expand_tree(self, node=None, mode=DEPTH, filter_fun=lambda x: True, key=lambda x: x, reverse=False,
                    lookup_nodes=False):
        """
        Python generator to traverse the tree (or a subtree) with optional node filtering and sorting.

        :param node: Node or node ID (nid) from which tree traversal will start. If None tree root will be used.
        :param mode: Traversal mode, may be either ``DEPTH``, ``WIDTH`` or ``ZIGZAG``
        :param filter_fun: the ``filter_fun`` function is performed on Node object during traversing.
            The traversing will NOT visit those nodes (and their subrtree) which does not pass filtering.
        :param key: the ``key`` and ``reverse`` are present to sort nodes at each level.
            If ``key`` is None the function returns nodes in original insertion order.
        :param reverse: if True reverse ordering.
        :param lookup_nodes: return Nodes or just the node IDs (nid)
        :return: Node IDs that satisfy the conditions in the defined order.
        :rtype: generator object
        """
        if node is None:
            nid = self.root
            curr_node = self.nodes[self.root]
        else:
            nid = self._get_nid(node)
            curr_node = self.nodes[nid]  # TODO maybe there is better solution.

        if not filter_fun(curr_node):  # subtree filtered out
            return

        if lookup_nodes:
            yield node
            lookup_nodes_fun = partial(lambda x: x)
        else:
            yield nid  # yield current node ID
            lookup_nodes_fun = partial(lambda x: x.nid)

        # filtered_successors = partial(lambda node: ,)

        queue = self.children(curr_node, filter_fun, lookup_nodes=True)
        if mode in {self.DEPTH, self.WIDTH}:
            # Set sort key fun
            key_fun = self._create_sort_fun(key, reverse)

            # Set tree traversal order
            if mode is self.DEPTH:
                order_fun = partial(lambda x, y: chain(x, y))  # depth-first
            else:
                order_fun = partial(lambda x, y: chain(y, x))  # width-first

            queue = list(key_fun(queue))  # Sort children
            while len(queue) > 0:
                cn = queue[0]
                yield lookup_nodes_fun(cn)

                expansion = list(key_fun(self.children(cn, filter_fun, lookup_nodes=True)))  # Sort children
                queue = list(order_fun(expansion, queue[1:]))  # Step

        elif mode == self.ZIGZAG:
            stack_fw = []
            queue = list(queue)
            queue.reverse()
            stack = stack_bw = queue
            forward = False
            while len(stack) > 0:
                curr_node = lookup_nodes_fun(stack.pop(0))
                yield curr_node
                expansion = list(self.children(curr_node, filter_fun, lookup_nodes=True))
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

    # MODIFYING FUNCTIONS ----------------------------------------------------------------------------------------------
    def create_node(self, tag=None, nid=None, data=None, parent=None):
        """
        Create a child node for given ``parent`` node. If node identifier (``nid``) is absent,
         a UUID will be generated automatically.
        """
        node = self.node_class(tag=tag, nid=nid, data=data)
        self.add_node(node, parent)
        return node

    def add_node(self, node, parent=None):  # TODO default factory ?
        """
        Add a new node object to the tree and make the parent as the root by default.

        The 'node' parameter refers to an instance of Class::Node.
        """
        if not isinstance(node, self.node_class):
            raise TypeError(f'First parameter must be object of {self.node_class} !')

        nid = node.nid
        if nid in self.nodes:
            raise DuplicatedNodeIdError(f'Cannot create node with ID {nid} !')

        if parent is not None:  # Parent can be None!
            pid = self._get_nid(parent)
        else:
            pid = None

        if pid is None:  # Adding root node
            if self.root is not None:
                raise MultipleRootError('A tree takes one root merely!')
            else:
                self.root = nid
        elif pid not in self.nodes:
            raise NodeIDAbsentError(f'Parent node ({pid}) is not in the tree!')
        else:  # pid is not None and pid in self.nodes -> Updating non-root node's parent
            self.nodes[pid].update_successors(nid, self.node_class.ADD, tree_id=self.tree_id)

        node.set_predecessor(pid, self.tree_id)
        node.set_initial_tree_id(self.tree_id)
        self.nodes[nid] = node

    def update_node(self, node_to_update, **attrs):
        """
        Update node's attributes.

        :param node_to_update: the identifier (nid) of modified node
        :param attrs: attribute pairs recognized by Node object (Beware, attributes unconditionally updated!)
        :return: None
        """
        nid_to_update = self._get_nid(node_to_update)
        cn = self.nodes[nid_to_update]

        new_identifier_val = attrs.pop('nid', None)
        if new_identifier_val is not None:
            # Updating node id meets following constraints:
            # * Update node ID (nid) property
            cn = self.nodes.pop(nid_to_update)
            cn.nid = new_identifier_val

            # * Update tree registration of nodes
            self.nodes[new_identifier_val] = cn

            # * Update parent's followers
            parent_nid = cn.predecessor(self.tree_id)
            if parent_nid is not None:
                self.nodes[parent_nid].update_successors(nid_to_update, self.node_class.REPLACE, new_identifier_val,
                                                         tree_id=self.tree_id)

            # * Update children's parents
            for fp in cn.successors(self.tree_id):
                self.nodes[fp].set_predecessor(new_identifier_val, self.tree_id)

            # * Update tree root if necessary
            if self.root == nid_to_update:
                self.root = new_identifier_val

        for attr, val in attrs.items():
            setattr(cn, attr, val)  # Potentially dangerous operation!

    def move_node(self, source, destination):
        """
        Move node ID or Node instance ``source`` (with its subree) from its parent to another parent ``destination``.
        """
        source_nid = self._get_nid(source)
        destination_nid = self._get_nid(destination)

        # Source node can not be root, but destination can be!
        if self.is_ancestor(source, destination):  # Double check! TODO
            raise LoopError(f'Source ({source_nid}) node ID is an ancestor of '
                            f'destination ({destination_nid}) node ID in the tree!')

        # Update old and new parents
        # Parent can not be None as it woluld mean source node is the root which case is already handled by LoopError
        parent = self.nodes[source].predecessor(self.tree_id)
        self.nodes[parent].update_successors(source, self.node_class.DELETE, tree_id=self.tree_id)
        self.nodes[destination].update_successors(source, self.node_class.ADD, tree_id=self.tree_id)
        # Add new parent for source
        self.nodes[source].set_predecessor(destination, self.tree_id)

    def remove_node(self, node):
        """
        Delete a node by linking past it.

        For example, if we have `a -> node -> c` and delete node ``node``, we are left with `a -> c`.
        """
        nid = self._get_nid(node)

        if self.root == nid:
            raise LinkPastRootNodeError('Cannot link past the root node, delete it with remove_subtree()!')
        # Get the parent of the node we are linking past
        parent_node = self.nodes[self.nodes[nid].predecessor(self.tree_id)]
        # Set the children of the node to the parent
        for child in self.nodes[nid].successors(self.tree_id):
            self.nodes[child].set_predecessor(parent_node.nid, self.tree_id)
        # Move the children from the current node to the parent node
        for curr_nid in self.nodes[nid].successors(self.tree_id):
            parent_node.update_successors(curr_nid, tree_id=self.tree_id)
        # Delete the node from the parent and from the nodes registry
        parent_node.update_successors(nid, mode=self.node_class.DELETE, tree_id=self.tree_id)
        del self.nodes[nid]

    def subtree(self, node, tree_id=None, deep=False):  # TODO implement deep copy!
        """
        Return a shallow or deep COPY of subtree with nid being the new root.
        If nid is None, return an empty tree.

        This line creates a deep copy of the entire tree.
        """
        st = self._clone(tree_id)
        if node is None:
            return st  # TODO deepcopy if deep true!
        nid = self._get_nid(node)

        st.root = nid
        for node_n_nid in self.expand_tree(nid):
            node_n = self.nodes[node_n_nid]
            st.nodes[node_n_nid] = node_n
            # Define nodes parent/children in this tree
            # All pointers are the same as copied tree, except the root
            st.nodes[node_n_nid].clone_pointers(self.tree_id, st.tree_id)
            if node_n_nid == nid:
                # Reset root parent for the new tree
                st.nodes[node_n_nid].set_predecessor(None, st.tree_id)

        if deep:
            st = Tree(nid, deep=True)

        return st

    def remove_subtree(self, node):
        """
        Remove a node ID or Node instance indicated by 'node' with all its successors.
        Return the number of removed nodes.
        """
        nid = self._get_nid(node)

        if nid == self.root:
            # This tree will be empty, but the nodes may occur in other trees, so we meed to update each separately!
            self.root = None
        else:
            # Update parent info (if nid is not root)
            # In the original tree, the removed nid will be unreferenced from its parents children
            parent = self.nodes[nid].predecessor(self.tree_id)
            self.nodes[parent].update_successors(nid, self.node_class.DELETE, tree_id=self.tree_id)

        self.nodes[nid].set_predecessor(None, self.tree_id)

        # Remove node and its children (We must generate the list of nodes to be removed before modifying the tree!)
        nids_removed = list(self.expand_tree(nid))
        for curr_nid in nids_removed:
            # Remove node (from this tree)
            curr_node = self.nodes.pop(curr_nid)
            # Remove parent (from this tree)
            curr_node.set_predecessor(None, self.tree_id)
            # Remove children (from this tree)
            for cid in curr_node.successors(self.tree_id):
                curr_node.update_successors(cid, self.node_class.DELETE, tree_id=self.tree_id)

        return len(nids_removed)

    def pop_subtree(self, node, tree_id=None):
        """
        Get a subtree with ``node`` being the root. If nid is None, an empty tree is returned.

        For the original tree, this method is similar to`remove_subtree(self,nid)`,
         because given node and its children are removed from the original tree in both methods.
        For the returned value and performance, these two methods are
        different:

            * `remove_subtree` returns the number of deleted nodes;
            * `pop_subtree` returns a subtree of deleted nodes;

        You are always suggested to use `remove_subtree` if your only to delete nodes from a tree,
         as the other one need memory allocation to store the new tree.

        :return: a :class:`Tree` object.
        """
        nid = self._get_nid(node)

        st = self._clone(tree_id)  # TODO this why?
        if nid is None:
            return st

        # TODO this is duplicate!
        if nid == self.root:
            # This tree will be empty, but the nodes may occur in other trees, so we meed to update each separately!
            self.root = None
        else:
            # Update parent info (if nid is not root)
            # In the original tree, the removed nid will be unreferenced from its parents children
            parent = self.nodes[nid].predecessor(self.tree_id)
            self.nodes[parent].update_successors(nid, self.node_class.DELETE, tree_id=self.tree_id)

        nids_removed = list(self.expand_tree(nid))
        for curr_nid in nids_removed:
            # Remove node (from this tree)
            curr_node = self.nodes.pop(curr_nid)
            # Clone pointers to the new tree
            curr_node.clone_pointers(self.tree_id, st.tree_id)
            # Because we reuse the Node instance, we must clean all reference to the old tree!
            curr_node.set_initial_tree_id(st.tree_id)  # TODO hack!
            curr_node.delete_pointers(self.tree_id)
            # Add the prepared node to the new tree
            st.nodes[curr_nid] = curr_node

        # Set the parent of the root in the new tree!
        st.nodes[nid].set_predecessor(None, st.tree_id)
        st.root = nid

        return st

    def merge(self, nid, new_tree, deep=False):
        """
        Patch ``new_tree`` on current tree by pasting new_tree root children on current tree ``nid`` node.

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

        Note: if current tree is empty and nid is None, the ``new_tree`` root will be used as root on current tree.
         In all other cases ``new_tree`` root is not pasted.
        """
        if new_tree.root is None:
            return

        if nid is None:
            if self.root is None:
                new_tree_root = new_tree.nodes[new_tree.root]
                self.add_node(new_tree_root)
                nid = new_tree.root
            else:
                raise ValueError('Must define "nid" under a root which the new tree is merged!')

        for child_nid in new_tree.children(new_tree.root):
            self.paste(nid, new_tree.subtree(child_nid), deep)

    def paste(self, nid, new_tree, deep=False):
        """
        Paste a ``new_tree`` to the original one by linking the root of new tree to given node (nid).
        Add ``deep`` for the deep copy of pasted tree.
        """
        if not isinstance(new_tree, self.__class__):
            raise TypeError(f'new_tree is expected to be {self.__class__} not {type(new_tree)} !')
        elif new_tree.root is None:  # Nothing to do!
            return
        elif nid is None:
            raise ValueError('Must define "nid" under which new tree is pasted!')
        elif nid not in self.nodes:
            raise NodeIDAbsentError(f'Node ({nid}) is not in the tree!')

        set_joint = set(new_tree.nodes) & set(self.nodes)  # Joint keys
        if len(set_joint) > 0:
            raise ValueError(f'Duplicated nodes {sorted(set_joint)} exists.')

        for cid, node in new_tree.nodes.items():
            if deep:
                node = deepcopy(new_tree[node])
            node.clone_pointers(new_tree.tree_id, self.tree_id)
            self.nodes[cid] = node

        self.nodes[new_tree.root].set_predecessor(nid, self.tree_id)
        if nid is not None:
            self.nodes[nid].update_successors(new_tree.root, self.node_class.ADD, tree_id=self.tree_id)

    # PRINT RELATED FUNCTIONS ------------------------------------------------------------------------------------------
    def __get(self, node, level, filter_fun, key, reverse, line_type):
        # Set sort key and reversing if needed
        key_fun = self._create_sort_fun(key, reverse)

        # Set line types
        line_elems = self._dt.get(line_type)
        if line_elems is None:
            raise ValueError(f'Undefined line type ({line_type})! Must choose from {set(self._dt)}!')

        if node is None:
            nid = self.root
        else:
            nid = self._get_nid(node)

        init_node = self.nodes[nid]
        if filter_fun(init_node):
            return self.__get_recursive(init_node, level, filter_fun, key_fun, [], *line_elems)

    def __get_recursive(self, node, level, filter_fun, sort_fun, is_last,
                        dt_vertical_line, dt_line_box, dt_line_corner):
        # Format current level
        lines = []
        if level > self.ROOT:
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

        # Do deeper!
        children = list(self.children(node, filter_fun, lookup_nodes=True))
        idxlast = len(children) - 1

        for idx, child in enumerate(sort_fun(children)):
            is_last.append(idx == idxlast)
            yield from self.__get_recursive(child, level + 1, filter_fun, sort_fun, is_last,
                                            dt_vertical_line, dt_line_box, dt_line_corner)
            is_last.pop()

    def __print_backend(self, nid=None, level=ROOT, filter_fun=lambda x: True, key=lambda x: x, reverse=False,
                        line_type='ascii-ex', get_label_fun=lambda node: node.tag, record_end=''):
        """
        Another implementation of printing tree using Stack Print tree structure in hierarchy style.
        The ``key`` ``reverse`` is present to sort node at each level.

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
            yield f'{pre}{label}{record_end}'

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
        reader = self.__print_backend(nid, level, filter_fun, key, reverse, line_type, get_label_fun)

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

        queue = list(self.children(nid, lookup_nodes=True))
        if sort:
            queue.sort(key=key, reverse=reverse)

        for elem in queue:
            # TODO recursive!
            dict_form = self.to_dict(elem.nid, with_data=with_data, sort=sort, reverse=reverse)
            tree_dict[ntag]['children'].append(dict_form)

        if len(tree_dict[ntag]['children']) == 0:
            if not with_data:
                tree_dict = self.nodes[nid].tag
            else:
                tree_dict = {ntag: {'data': self.nodes[nid].data}}

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

            for nid in self.expand_tree(mode=self.WIDTH):
                state = f'"{nid}" [label="{self.nodes[nid].tag}", shape={shape}]'
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
