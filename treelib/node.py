#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Node structure in treelib.

A :class:`Node` object contains basic properties such as node identifier,
node tag, parent node, children nodes etc., and some operations for a node.
"""

import copy
import uuid
from collections import defaultdict
from typing import Any, Hashable

from .exceptions import NodePropertyError


class Node:
    """
    Nodes are elementary objects that are stored in the `_nodes` dictionary of a Tree.
    Use `data` attribute to store node-specific data.
    """

    #: Mode constants for routine `update_fpointer()`.
    (ADD, DELETE, INSERT, REPLACE) = list(range(4))

    def __init__(self, tag: Hashable = None, identifier: Hashable = None, expanded: bool = True, data: Any = None):
        """
        Create a new Node object to be placed inside a Tree object.
        """

        #: If given as a parameter, must be unique (tuple of parent nodes recommended)
        if identifier is None:
            identifier = str(uuid.uuid1())
        self._identifier = identifier

        #: None or something else
        #: If None, self._identifier will be set to the identifier's value.
        # The readable node name for humans. This attribute can be accessed and
        #  modified with ``.`` and ``=`` operator respectively.
        if tag is None:
            self.tag = self._identifier
        else:
            self.tag = tag

        #: Boolean
        self.expanded: bool = expanded

        #: Identifier of the parent's node :
        self._predecessor = {}
        #: Identifier(s) of the children's node(s) :
        self._successors = defaultdict(list)

        #: User payload associated with this node.
        self.data = data

        # For retro-compatibility on bpointer/fpointer
        self._initial_tree_id = None

    def __lt__(self, other):
        return self.tag < other.tag

    def set_initial_tree_id(self, tree_id):
        if self._initial_tree_id is None:
            self._initial_tree_id = tree_id

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

    def set_successors(self, value, tree_id=None):
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

    def _manipulator_append(self, nid, tree_id, _=None, __=None):
        self.successors(tree_id).append(nid)

    def _manipulator_delete(self, nid, tree_id, _=None, __=None):
        if nid in self.successors(tree_id):
            self.successors(tree_id).remove(nid)
        else:
            ValueError(f'Nid {nid} wasn\'t present in fpointer!')

    def _manipulator_replace(self, nid, tree_id, mode=None, replace=None):
        if replace is None:
            raise NodePropertyError(f'Argument "repalce" should be provided when mode is {mode}!')
        ind = self.successors(tree_id).index(nid)
        self.successors(tree_id)[ind] = replace

    def update_successors(self, nid, mode=ADD, replace=None, tree_id=None):
        """
        Update the children list with different modes: addition (Node.ADD or Node.INSERT) and deletion (Node.DELETE).
        """
        if nid is None:
            return

        manipulator_lookup = {
            self.ADD: self._manipulator_append,
            self.DELETE: self._manipulator_delete,
            self.INSERT: self._manipulator_append,  # Removed deprecated value
            self.REPLACE: self._manipulator_replace
        }

        f_name = manipulator_lookup.get(mode)
        if f_name is None:
            raise NotImplementedError(f'Unsupported node updating mode {mode}!')
        f_name(nid, tree_id, mode, replace)

    @property
    def identifier(self):
        """
        The unique ID of a node within the scope of a tree. This attribute can be accessed and modified with
         ``.`` and ``=`` operator respectively.
        """
        return self._identifier

    @identifier.setter
    def identifier(self, value):
        """
        Set the value of `_identifier`.
        """
        if value is not None:
            self._identifier = value
        else:
            ValueError('Node ID can not be None!')

    def clone_pointers(self, former_tree_id, new_tree_id):
        former_bpointer = self.predecessor(former_tree_id)
        self.set_predecessor(former_bpointer, new_tree_id)
        former_fpointer = self.successors(former_tree_id)
        # fpointer is a list and would be copied by reference without using deepcopy
        self.set_successors(copy.deepcopy(former_fpointer), tree_id=new_tree_id)

    def reset_pointers(self, tree_id):
        self.set_predecessor(None, tree_id)
        self.set_successors([], tree_id=tree_id)

    def is_leaf(self, tree_id=None):
        """
        Return true if current node has no children.
        """
        if tree_id is None:
            tree_id = self._initial_tree_id

        return len(self.successors(tree_id)) == 0

    def is_root(self, tree_id=None):
        """
        Return true if self has no parent, i.e. as root.
        """
        if tree_id is None:
            tree_id = self._initial_tree_id

        return self.predecessor(tree_id) is None

    def __repr__(self):
        name = self.__class__.__name__
        kwargs = [
            f'tag={self.tag}',
            f'identifier={self.identifier}',
            f'data={self.data}',
        ]
        return f'{name}({", ".join(kwargs)})'
