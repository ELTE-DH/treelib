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

    def __init__(self, tree=None, deep: bool = False, node_class=None, identifier=None):
        """
        Initiate a new tree or copy another tree with a shallow or deep copy.
        """

        # Initialize self._set_identifier
        if identifier is None:
            identifier = str(uuid.uuid1())
        self.identifier = identifier

        if node_class is not None:
            assert issubclass(node_class, Node)
            self.node_class = node_class

        #: Dict of nodes in a tree: {identifier: node_instance}.
        self.nodes = {}

        #: Get or set the identifier of the root. This attribute can be accessed and modified
        #:  with ``.`` and ``=`` operator respectively.
        self.root = None

        if tree is not None:
            self.root = tree.root
            for nid, node in tree.nodes.items():
                new_node = deepcopy(node) if deep else node
                self.nodes[nid] = new_node
                if tree.identifier != self.identifier:
                    new_node.clone_pointers(tree.identifier, self.identifier)

        # Render characters
        self._dt = {
            'ascii': ('|', '|-- ', '+-- '),
            'ascii-ex': ('\u2502', '\u251c\u2500\u2500 ', '\u2514\u2500\u2500 '),
            'ascii-exr': ('\u2502', '\u251c\u2500\u2500 ', '\u2570\u2500\u2500 '),
            'ascii-em': ('\u2551', '\u2560\u2550\u2550 ', '\u255a\u2550\u2550 '),
            'ascii-emv': ('\u2551', '\u255f\u2500\u2500 ', '\u2559\u2500\u2500 '),
            'ascii-emh': ('\u2502', '\u255e\u2550\u2550 ', '\u2558\u2550\u2550 '),
        }

    def _clone(self, identifier=None, with_tree=False, deep=False):
        """
        Clone current instance, with or without tree.

        Method intended to be overloaded, to avoid rewriting whole "subtree" and "remove_subtree" methods when
        inheriting from Tree.
        >>> class TreeWithComposition(Tree):
        >>>     def __init__(self, tree_description, tree=None, deep=False, identifier=None):
        >>>         self.tree_description = tree_description
        >>>         super().__init__(tree=tree, deep=deep, identifier=identifier)
        >>>
        >>>     def _clone(self, identifier=None, with_tree=False, deep=False):
        >>>         return TreeWithComposition(
        >>>             identifier=identifier,
        >>>             deep=deep,
        >>>             tree=self if with_tree else None,
        >>>             tree_description=self.tree_description
        >>>         )
        >>> my_custom_tree = TreeWithComposition(tree_description='smart tree')
        >>> subtree = my_custom_tree.subtree()
        >>> subtree.tree_description
        'smart tree'
        """
        return self.__class__(identifier=identifier, tree=self if with_tree else None, deep=deep)

    def all_nodes(self):
        """
        Returns all nodes in an iterator.
        """
        return self.nodes.values()

    def get_node(self, nid):
        """
        Get the object of the node with ID of ``nid``.

        An alternative way is using '[]' operation on the tree. But small difference exists between them:
        ``get_node()`` will return None if ``nid`` is absent, whereas '[]' will raise ``KeyError``.
        """
        if nid is not None:  # In case nid were a valid node identifier
            return self.nodes.get(nid, None)

    def __getitem__(self, key):
        """
        Return nodes[key]
        """
        try:
            return self.nodes[key]
        except KeyError:
            raise NodeIDAbsentError(f'Node \'{key}\' is not in the tree!')

    def __len__(self):
        """
        Return len(nodes)
        """
        return len(self.nodes)

    def __contains__(self, identifier):
        return identifier in self.nodes

    def add_node(self, node, parent=None):
        """
        Add a new node object to the tree and make the parent as the root by default.

        The 'node' parameter refers to an instance of Class::Node.
        """
        if not isinstance(node, self.node_class):
            raise OSError(f'First parameter must be object of {self.node_class}!')
        elif node.identifier in self.nodes:
            raise DuplicatedNodeIdError(f'Cannot create node with ID \'{node.identifier}\'!')

        if isinstance(parent, self.node_class):
            pid = parent.identifier
        else:
            pid = parent

        if pid is None:
            if self.root is not None:
                raise MultipleRootError('A tree takes one root merely!')
            else:
                self.root = node.identifier
        elif pid not in self.nodes:
            raise NodeIDAbsentError(f'Parent node \'{pid}\' is not in the tree!')

        self.nodes.update({node.identifier: node})
        if pid is not None:
            self[pid].update_successors(node.identifier, self.node_class.ADD, tree_id=self.identifier)
        self[node.identifier].set_predecessor(pid, self.identifier)
        node.set_initial_tree_id(self.identifier)

    def ancestor(self, nid, level=None):
        """
        For a given id, get ancestor node object at a given level.
        If no level is provided, the parent node is returned.
        """
        if nid not in self.nodes:
            raise NodeIDAbsentError('Node \'{nid}\' is not in the tree!')

        descendant = self[nid]
        ascendant = self[nid].predecessor(self.identifier)
        ascendant_level = self.level(ascendant)

        if level is None:
            return ascendant  # parent
        elif nid == self.root:
            return self[nid]  # root -> root
        elif level >= self.level(descendant.identifier):
            raise InvalidLevelNumber(f'Descendant level (level {self.level(descendant.identifier)}) '
                                     f'must be greater than its ancestor\'s level (level {level})!')

        # Ascend to the appropriate level
        while ascendant is not None:
            if ascendant_level == level:
                return self[ascendant]
            else:
                descendant = ascendant
                ascendant = self[descendant].predecessor
                ascendant_level = self.level(ascendant)

    def children(self, nid):
        """
        Return the children (Node) list of nid.
        Empty list is returned if nid does not exist
        """
        return [self[i] for i in self.is_branch(nid)]

    def create_node(self, tag=None, identifier=None, parent=None, data=None):
        """
        Create a child node for given @parent node. If ``identifier`` is absent, a UUID will be generated automatically.
        """
        node = self.node_class(tag=tag, identifier=identifier, data=data)
        self.add_node(node, parent)
        return node

    def level(self, nid, filter_fun=lambda x: True):
        """
        Get the node level in this tree.
        The level is an integer starting with '0' at the root.
        In other words, the root lives at level '0';

        Update: @filter params is added to calculate level passing
        exclusive nodes.
        """
        return sum(1 for _ in self.rsearch(nid, filter_fun)) - 1

    def depth(self, node=None):
        """
        Get the maximum level of this tree or the level of the given node.

        @param node Node instance or identifier
        @return int
        @throw NodeIDAbsentError
        """
        if node is None:
            # Get maximum level of this tree
            ret_level = max(self.level(leaf.identifier) for leaf in self.leaves())
        else:
            # Get level of the given node
            if not isinstance(node, self.node_class):
                nid = node
            else:
                nid = node.identifier
            if nid not in self.nodes:
                raise NodeIDAbsentError(f'Node \'{nid}\' is not in the tree!')
            ret_level = self.level(nid)

        return ret_level

    def expand_tree(self, nid=None, mode=DEPTH, filter_fun=lambda x: True, key=None, reverse=False, sorting=True):
        """
        Python generator to traverse the tree (or a subtree) with optional node filtering and sorting.

        Loosely based on an algorithm from 'Essential LISP' by John R. Anderson,
        Albert T. Corbett, and Brian J. Reiser, page 239-241.

        :param nid: Node identifier from which tree traversal will start.
            If None tree root will be used
        :param mode: Traversal mode, may be either DEPTH, WIDTH or ZIGZAG
        :param filter_fun: the @filter_fun function is performed on Node object during
            traversing. In this manner, the traversing will NOT visit the node
            whose condition does not pass the filter and its children.
        :param key: the @key and @reverse are present to sort nodes at each
            level. If @key is None sorting is performed on node tag.
        :param reverse: if True reverse sorting
        :param sorting: if True perform node sorting, if False return
            nodes in original insertion order. In latter case @key and
            @reverse parameters are ignored.
        :return: Node IDs that satisfy the conditions
        :rtype: generator object
        """
        if nid is None:
            nid = self.root
        if nid not in self.nodes:
            raise NodeIDAbsentError('Node \'{nid}\' is not in the tree!')

        if filter_fun(self[nid]):
            yield nid
            queue = [self[i] for i in self[nid].successors(self.identifier) if filter_fun(self[i])]
            if mode in {self.DEPTH, self.WIDTH}:
                if sorting:
                    queue.sort(key=key, reverse=reverse)
                while len(queue) > 0:
                    yield queue[0].identifier
                    expansion = [self[i] for i in queue[0].successors(self.identifier) if filter_fun(self[i])]
                    if sorting:
                        expansion.sort(key=key, reverse=reverse)
                    if mode is self.DEPTH:
                        queue = expansion + queue[1:]  # depth-first
                    elif mode is self.WIDTH:
                        queue = queue[1:] + expansion  # width-first

            elif mode is self.ZIGZAG:
                stack_fw = []
                queue.reverse()
                stack = stack_bw = queue
                direction = False
                while len(stack) > 0:
                    expansion = [self[i] for i in stack[0].successors(self.identifier) if filter_fun(self[i])]
                    yield stack.pop(0).identifier
                    if direction:
                        expansion.reverse()
                        stack_bw = expansion + stack_bw
                    else:
                        stack_fw = expansion + stack_fw
                    if len(stack) == 0:
                        direction = not direction
                        if direction:
                            stack = stack_fw
                        else:
                            stack = stack_bw
            else:
                raise ValueError(f'Traversal mode \'{mode}\' is not supported!')

    def filter_nodes(self, func):
        """
        Filters all nodes by function.

        :param func: is passed one node as an argument and that node is included if function returns true,
        :return: a filter iterator of the node in python 3 or a list of the nodes in python 2.
        """
        return filter(func, self.nodes.values())

    def is_branch(self, nid):
        """
        Return the children (ID) list of nid.
        Empty list is returned if nid does not exist
        """
        if nid is None:
            raise OSError('First parameter cannot be None!')
        if nid not in self.nodes:
            raise NodeIDAbsentError(f'Node \'{nid}\' is not in the tree!')

        try:
            fpointer = self[nid].successors(self.identifier)
        except KeyError:
            fpointer = []
        return fpointer

    def leaves(self, nid=None):
        """
        Get leaves of the whole tree or a subtree.
        """
        leaves = []
        if nid is None:
            for node in self.nodes.values():
                if node.is_leaf(self.identifier):
                    leaves.append(node)
        else:
            for node in self.expand_tree(nid):
                if self[node].is_leaf(self.identifier):
                    leaves.append(self[node])
        return leaves

    def link_past_node(self, nid):
        """
        Delete a node by linking past it.

        For example, if we have `a -> b -> c` and delete node b, we are left
        with `a -> c`.
        """
        if nid not in self.nodes:
            raise NodeIDAbsentError(f'Node \'{nid}\' is not in the tree!')
        if self.root == nid:
            raise LinkPastRootNodeError('Cannot link past the root node, delete it with remove_node()!')
        # Get the parent of the node we are linking past
        parent_node = self[self[nid].predecessor(self.identifier)]
        # Set the children of the node to the parent
        for child in self[nid].successors(self.identifier):
            self[child].set_predecessor(parent_node.identifier, self.identifier)
        # Link the children to the parent

        for identifier in self[nid].successors(self.identifier):
            parent_node.update_successors(identifier, tree_id=self.identifier)
        # Delete the node
        parent_node.update_successors(nid, mode=parent_node.DELETE, tree_id=self.identifier)
        del self.nodes[nid]

    def move_node(self, source, destination):
        """
        Move node @source from its parent to another parent @destination.
        """
        if source not in self.nodes or destination not in self.nodes:
            raise NodeIDAbsentError
        elif self.is_ancestor(source, destination):
            raise LoopError

        parent = self[source].predecessor(self.identifier)
        if parent is not None:
            self[parent].update_successors(source, self.node_class.DELETE, tree_id=self.identifier)
        if destination is not None:
            self[destination].update_successors(source, self.node_class.ADD, tree_id=self.identifier)
        self[source].set_predecessor(destination, self.identifier)

    def is_ancestor(self, ancestor, grandchild):
        """
        Check if the @ancestor the preceding nodes of @grandchild.

        :param ancestor: the node identifier
        :param grandchild: the node identifier
        :return: True or False
        """
        child = grandchild
        parent = self[grandchild].predecessor(self.identifier)
        while parent is not None:
            if parent == ancestor:
                return True
            else:
                child = self[child].predecessor(self.identifier)
                parent = self[child].predecessor(self.identifier)
        return False

    def parent(self, nid):
        """
        Get parent :class:`Node` object of given id.
        """
        if nid not in self.nodes:
            raise NodeIDAbsentError('Node \'{nid}\' is not in the tree!')

        pid = self[nid].predecessor(self.identifier)
        if pid is None or pid not in self.nodes:
            return None

        return self[pid]

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
        for child in new_tree.children(new_tree.root):
            self.paste(nid=nid, new_tree=new_tree.subtree(child.identifier), deep=deep)

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
            raise NodeIDAbsentError(f'Node \'{nid}\' is not in the tree!')

        set_joint = set(new_tree.nodes) & set(self.nodes)  # joint keys
        if len(set_joint) > 0:
            raise ValueError(f'Duplicated nodes {list(map(str, set_joint))} exists.')

        for cid, node in new_tree.nodes.items():
            if deep:
                node = deepcopy(new_tree[node])
            self.nodes.update({cid: node})
            node.clone_pointers(new_tree.identifier, self.identifier)

        self[new_tree.root].set_predecessor(nid, self.identifier)
        if nid is not None:
            self[nid].update_successors(new_tree.root, self.node_class.ADD, tree_id=self.identifier)

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
            node_ids = [nid for nid in self.rsearch(leaf.identifier, filter_fun)]
            node_ids.reverse()
            res.append(tuple(node_ids))

        return res

    def remove_node(self, identifier):
        """
        Remove a node indicated by 'identifier' with all its successors. Return the number of removed nodes.
        """
        if identifier not in self.nodes:
            raise NodeIDAbsentError(f'Node \'{identifier}\' is not in the tree!')

        parent = self[identifier].predecessor(self.identifier)

        # Remove node and its children
        removed = list(self.expand_tree(identifier))

        for nid in removed:
            if nid == self.root:
                self.root = None
            self[nid].set_predecessor(None, self.identifier)
            if nid is not None:
                for cid in self[nid].successors(self.identifier):
                    self[nid].update_successors(cid, self.node_class.DELETE, tree_id=self.identifier)

        # Update parent info
        if parent is not None:
            self[parent].update_successors(identifier, self.node_class.DELETE, tree_id=self.identifier)
        self[identifier].set_predecessor(None, self.identifier)

        for nid in removed:
            self.nodes.pop(nid)
        return len(removed)

    def remove_subtree(self, nid, identifier=None):
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
        st = self._clone(identifier)
        if nid is None:
            return st

        if nid not in self.nodes:
            raise NodeIDAbsentError(f'Node \'{nid}\' is not in the tree!')
        st.root = nid

        # In original tree, the removed nid will be unreferenced from its parents children
        parent = self[nid].predecessor(self.identifier)

        removed = list(self.expand_tree(nid))
        for id_ in removed:
            if id_ == self.root:
                self.root = None
            st.nodes.update({id_: self.nodes.pop(id_)})
            st[id_].clone_pointers(self.identifier, st.identifier)
            st[id_].reset_pointers(self.identifier)
            if id_ == nid:
                st[id_].set_predecessor(None, st.identifier)
        if parent is not None:
            self[parent].update_successors(nid, self.node_class.DELETE, tree_id=self.identifier)
        return st

    def rsearch(self, nid, filter_fun=lambda x: True):
        """
        Traverse the tree branch along the branch from nid to its ancestors (until root).

        :param nid: node ID
        :param filter_fun: the function of one variable to act on the :class:`Node` object.
        """
        if nid is None:
            return
        if nid not in self.nodes:
            raise NodeIDAbsentError(f'Node \'{nid}\' is not in the tree!')

        current = nid
        while current is not None:
            if filter_fun(self[current]):
                yield current
            # subtree() hasn't updated the predecessor
            if self.root != current:
                current = self[current].predecessor(self.identifier)
            else:
                current = None

    def siblings(self, nid):
        """
        Return the siblings of given @nid.

        If @nid is root or there are no siblings, an empty list is returned.
        """
        siblings = []

        if nid != self.root:
            pid = self[nid].predecessor(self.identifier)
            siblings = [self[i] for i in self[pid].successors(self.identifier) if i != nid]

        return siblings

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
            return sum(1 for node in self.nodes.values() if self.level(node.identifier) == level)

    def subtree(self, nid, identifier=None, deep=False):
        """
        Return a shallow COPY of subtree with nid being the new root.
        If nid is None, return an empty tree.

        This line creates a deep copy of the entire tree.
        """
        st = self._clone(identifier)
        if nid is None:
            return st  # TODO deepcopy if deep true!
        if nid not in self.nodes:
            raise NodeIDAbsentError(f'Node \'{nid}\' is not in the tree!')

        st.root = nid
        for node_n in self.expand_tree(nid):
            st.nodes.update({self[node_n].identifier: self[node_n]})
            # Define nodes parent/children in this tree
            # All pointers are the same as copied tree, except the root
            st[node_n].clone_pointers(self.identifier, st.identifier)
            if node_n == nid:
                # Reset root parent for the new tree
                st[node_n].set_predecessor(None, st.identifier)

        if deep:
            st = Tree(nid, deep=True)

        return st

    def update_node(self, nid, **attrs):
        """
        Update node's attributes.

        :param nid: the identifier of modified node
        :param attrs: attribute pairs recognized by Node object
        :return: None
        """
        cn = self[nid]  # TODO check if nid is None?

        if 'identifier' in attrs:  # However, identifier cannot be None it is better to check containment before poping
            identifier_val = attrs.pop('identifier')
            # Updating node id meets following constraints:
            # * Update node identifier property
            cn = self.nodes.pop(nid)
            setattr(cn, 'identifier', identifier_val)
            # * Update tree registration of var nodes
            self.nodes[identifier_val] = cn

            # * Update parent's followers
            pred_id = cn.predecessor(self.identifier)
            if pred_id is not None:
                self[pred_id].update_successors(nid, self.node_class.REPLACE, identifier_val, tree_id=self.identifier)

            # * Update children's parents
            for fp in cn.successors(self.identifier):
                self[fp].set_predecessor(identifier_val, self.identifier)

            # * Update tree root if necessary
            if self.root == nid:
                self.root = identifier_val

        for attr, val in attrs.items():
            setattr(cn, attr, val)

    def __get(self, nid, level, filter_fun, key, reverse, line_type):
        # Set sort key and reversing if needed
        if key is None:
            if reverse:
                key_fun = partial(reversed)
            else:
                key_fun = partial(lambda x: x)  # Do not sort at all!
        else:
            key_fun = partial(sorted, key=key, reverse=reverse)

        # Set line types
        line_elems = self._dt.get(line_type)
        if line_elems is None:
            raise ValueError(f'Undefined line type \'{line_type}\' must choose from {set(self._dt)}')

        return self.__get_iter(nid, level, filter_fun, key_fun, *line_elems, [])

    def __get_iter(self, nid, level, filter_fun, sort_fun, dt_vertical_line, dt_line_box, dt_line_corner, is_last):
        if nid is None:
            nid = self.root

        node = self[nid]

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
            children = [self[i] for i in node.successors(self.identifier) if filter_fun(self[i])]
            idxlast = len(children) - 1

            for idx, child in enumerate(sort_fun(children)):
                is_last.append(idx == idxlast)
                yield from self.__get_iter(child.identifier, level + 1, filter_fun, sort_fun,
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
        ntag = self[nid].tag
        tree_dict = {ntag: {'children': []}}
        if with_data:
            tree_dict[ntag]['data'] = self[nid].data

        queue = [self[i] for i in self[nid].successors(self.identifier)]
        if sort:
            queue.sort(key=key, reverse=reverse)

        for elem in queue:
            dict_form = self.to_dict(elem.identifier, with_data=with_data, sort=sort, reverse=reverse)
            tree_dict[ntag]['children'].append(dict_form)
        if len(tree_dict[ntag]['children']) == 0:
            tree_dict = self[nid].tag if not with_data else {ntag: {'data': self[nid].data}}
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
                nid = self[n].identifier
                state = f'"{nid}" [label="{self[n].tag}", shape={shape}]'
                nodes.append(state)

                for c in self.children(nid):
                    cid = c.identifier
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
