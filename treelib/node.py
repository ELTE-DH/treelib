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
from typing import Any, Union, Hashable, MutableMapping, List


class Node:
    """
    Nodes are elementary objects that are stored in the `_nodes` dictionary of a Tree.
    Use `data` attribute to store node-specific data.
    """

    def __init__(self, tag: Hashable = None, nid: Hashable = None, data: Any = None):
        """
        Create a new Node object to be placed inside a Tree object.
        """

        #: If given as a parameter, must be unique (tuple of parent nodes recommended).
        if nid is None:
            nid = str(uuid.uuid1())  # Generate a UUID from a host ID, sequence number, and the current time.
        self._identifier: Hashable = nid

        #: None or something else
        #: If None, self._identifier will be set to the tree_id's value.
        # The readable node name for humans. This attribute can be accessed and
        #  modified with ``.`` and ``=`` operator respectively.
        if tag is None:
            self.tag: Any = self._identifier
        else:
            self.tag: Any = tag

        #: Identifier (nid) of the parent's node for every tree_id the node is in (tree_id -> nid mapping).
        self._predecessor: MutableMapping[Hashable, Hashable] = {}
        #: Identifier(s) (nid(s)) of the children's node(s) for every tree_id (tree_id -> nid list mapping).
        self._successors: MutableMapping[Hashable, List] = defaultdict(list)

        #: User payload associated with this node for every tree_id the node is in (same payload for every tree).
        self.data: Any = data

    def __lt__(self, other):
        return self.tag < other.tag

    def predecessor(self, tree_id: Hashable) -> Hashable:
        """
        The parent ID of a node in a given tree.
        """
        return self._predecessor[tree_id]

    def set_predecessor(self, nid: Union[Hashable, None], tree_id: Hashable) -> None:
        """
        Set the value of `_predecessor`.
        """
        if nid is not None and not isinstance(nid, Hashable):
            raise TypeError(f'Node ID must be NoneType or Hashable not {type(nid)}!')

        self._predecessor[tree_id] = nid

    def remove_predecessor(self, tree_id: Hashable) -> None:
        self._predecessor.pop(tree_id)  # Raises ValueError if tree_id is not in the dict of predecessors.

    def successors(self, tree_id: Hashable) -> List:
        """
        With a getting operator, a list of IDs of node's children is obtained.
        """
        return self._successors[tree_id]

    def set_successors(self, value: List, tree_id: Hashable) -> None:
        """
        Set the value of `_successors`.
        With a setting operator, the value must be list and must be converted by the user.
        """
        if not isinstance(value, List):
            raise NotImplementedError(f'Unsupported value type {type(value)}!')

        self._successors[tree_id] = value

    def remove_successors(self,  tree_id: Hashable):
        self._successors.pop(tree_id)  # Raises ValueError if tree_id is not in the dict of successors.

    def add_successor(self, nid: Hashable, tree_id: Hashable) -> None:
        self._successors[tree_id].append(nid)

    def remove_successor(self, nid: Hashable, tree_id: Hashable) -> None:
        self._successors[tree_id].remove(nid)  # Raises ValueError if nid is not in the list of successors.

    def replace_successor(self, nid: Hashable, tree_id: Hashable, replace: Hashable) -> None:
        ind = self._successors[tree_id].index(nid)
        self._successors[tree_id][ind] = replace

    @property
    def nid(self) -> Hashable:
        """
        The unique ID of a node within the scope of a tree. This attribute can be accessed and modified with
         ``.`` and ``=`` operator respectively.
        """
        return self._identifier

    @nid.setter
    def nid(self, value: Hashable) -> None:
        """
        Set the value of `_identifier`.
        """
        if value is None:
            raise ValueError('Node ID can not be None!')

        self._identifier = value

    # TODO remove reference to the old tree if move is True else is_in_multiple_trees() will not work!
    def clone_pointers(self, former_tree_id: Hashable, new_tree_id: Hashable, move=False) -> None:
        """
        Copy Node to another tree.
        """
        # 1. Get the parent of the current node in the old tree.
        former_bpointer = self._predecessor[former_tree_id]
        # 2. Set the parent of the current node in the new tree.
        self._predecessor[new_tree_id] = former_bpointer
        # 3. Get the children of the current node in the old tree.
        former_fpointer = self._successors[former_tree_id]
        # 3. Set the children of the current node in the new tree.
        # fpointer is a list and without using deepcopy it would be copied by reference
        # and would mess up the two trees upon modification (i.e. both would be changed at the same time).
        self._successors[new_tree_id] = copy.deepcopy(former_fpointer)

    def delete_pointers(self, tree_id: Hashable) -> None:
        self._predecessor.pop(tree_id, None)
        self._successors.pop(tree_id, None)

    def is_leaf(self, tree_id: Hashable) -> bool:
        """
        Return true if current node has no children.
        """
        return len(self._successors[tree_id]) == 0

    def is_root(self, tree_id: Hashable) -> bool:
        """
        Return true if self has no parent, i.e. as root.
        """
        return self._predecessor[tree_id] is None

    def is_in_multiple_trees(self) -> bool:
        """
        Return true if node has predecessor or successors in multiple trees
        """
        return len(self._predecessor.keys()) > 1 or len(self._successors.keys()) > 1

    def __repr__(self) -> str:
        name = self.__class__.__name__
        kwargs = [
            f'tag={self.tag}',
            f'nid={self._identifier}',
            f'data={self.data}',
        ]
        return f'{name}({", ".join(kwargs)})'
