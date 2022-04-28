#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Node structure in treelib.

A :class:`Node` object contains basic properties such as node identifier (nid),
node tag, parent node, children nodes etc., and some operations for a node.
"""

import copy
import uuid
from collections import defaultdict
from typing import Any, Hashable


class Node:
    """
    Nodes are elementary objects that are stored in the `_nodes` dictionary of a Tree.
    Use `data` attribute to store node-specific data.
    """

    def __init__(self, tag: Hashable = None, nid: Hashable = None, data: Any = None):
        """
        Create a new Node object to be placed inside a Tree object.
        """

        #: If given as a parameter, must be unique (tuple of parent nodes recommended)
        if nid is None:
            nid = str(uuid.uuid1())
        self._identifier = nid

        #: None or something else
        #: If None, self._identifier will be set to the tree_id's value.
        # The readable node name for humans. This attribute can be accessed and
        #  modified with ``.`` and ``=`` operator respectively.
        if tag is None:
            self.tag = self._identifier
        else:
            self.tag = tag

        #: Identifier (nid) of the parent's node :
        self._predecessor = {}
        #: Identifier(s) (nid(s)) of the children's node(s) :
        self._successors = defaultdict(list)

        #: User payload associated with this node.
        self.data = data

    def __lt__(self, other):
        return self.tag < other.tag

    def predecessor(self, tree_id):
        """
        The parent ID of a node in a given tree.
        """
        return self._predecessor[tree_id]

    def set_predecessor(self, nid, tree_id):
        """
        Set the value of `_predecessor`.
        """
        self._predecessor[tree_id] = nid

    def successors(self, tree_id):
        """
        With a getting operator, a list of IDs of node's children is obtained.
        With a setting operator, the value can be list, set, or dict.
        For list or set, it is converted to a list type by the package; for dict, the keys are treated as the node IDs.
        """
        return self._successors[tree_id]

    def set_successors(self, value, tree_id):
        """
        Set the value of `_successors`.
        """
        setter_lookup = {
            'NoneType': lambda x: list(),
            'list': lambda x: x,
            'dict': lambda x: list(x.keys()),
            'set': lambda x: list(x)
        }

        t = value.__class__.__name__
        f_setter = setter_lookup.get(t)
        if f_setter is None:
            raise NotImplementedError(f'Unsupported value type {t}!')
        self._successors[tree_id] = f_setter(value)

    def add_successor(self, nid, tree_id):
        self.successors(tree_id).append(nid)

    def remove_successor(self, nid, tree_id):
        self.successors(tree_id).remove(nid)  # Raises ValueError if nid is not in the list of successors

    def replace_successor(self, nid, tree_id, replace):
        ind = self.successors(tree_id).index(nid)
        self.successors(tree_id)[ind] = replace

    @property
    def nid(self):
        """
        The unique ID of a node within the scope of a tree. This attribute can be accessed and modified with
         ``.`` and ``=`` operator respectively.
        """
        return self._identifier

    @nid.setter
    def nid(self, value):
        """
        Set the value of `_identifier`.
        """
        if value is not None:
            self._identifier = value
        else:
            raise ValueError('Node ID can not be None!')

    def clone_pointers(self, former_tree_id, new_tree_id):
        former_bpointer = self.predecessor(former_tree_id)
        self.set_predecessor(former_bpointer, new_tree_id)
        former_fpointer = self.successors(former_tree_id)
        # fpointer is a list and would be copied by reference without using deepcopy
        self.set_successors(copy.deepcopy(former_fpointer), new_tree_id)

    def delete_pointers(self, tree_id):
        self._predecessor.pop(tree_id, None)
        self._successors.pop(tree_id, None)

    def is_leaf(self, tree_id):
        """
        Return true if current node has no children.
        """
        return len(self.successors(tree_id)) == 0

    def is_root(self, tree_id):
        """
        Return true if self has no parent, i.e. as root.
        """
        return self.predecessor(tree_id) is None

    def is_in_other_trees(self, tree_id):
        """
        Return true if node has predecessor or successors in other trees
        """
        return len(self._predecessor.keys() - {tree_id}) > 0 or len(self._successors.keys() - {tree_id}) > 0

    def __repr__(self):
        name = self.__class__.__name__
        kwargs = [
            f'tag={self.tag}',
            f'nid={self.nid}',
            f'data={self.data}',
        ]
        return f'{name}({", ".join(kwargs)})'
