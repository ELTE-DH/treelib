#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tree structure in `treelib`.

The :class:`Tree` object defines the tree-like structure based on :class:`Node` objects.
A new tree can be created from scratch without any parameter or a shallow/deep copy of another tree.
When deep=True, a deepcopy operation is performed on feeding tree parameter and more memory
is required to create the tree.
"""

import uuid
from copy import deepcopy
from itertools import chain
from functools import partial
from typing import Any, Union, Hashable, MutableMapping, List, Tuple

from .exceptions import *
from .node import Node


class Tree:
    """
    Tree objects are made of Node(s) stored in nodes dictionary.
    """

    #: ROOT, DEPTH, WIDTH, ZIGZAG constants :
    DEPTH, WIDTH, ZIGZAG = range(3)
    node_class = Node

    def __init__(self, tree=None, deep: bool = False, node_class=None, tree_id: Union[Hashable, None] = None):
        """
        Initiate a new tree or copy another tree with a shallow or deep copy.
        """

        # Initialize self.tree_id
        if tree_id is None:
            tree_id = str(uuid.uuid1())  # Generate a UUID from a host ID, sequence number, and the current time.
        self.tree_id: Hashable = tree_id

        if node_class is not None:
            if not issubclass(node_class, Node):
                raise TypeError('node_class should be type of Node or sublcass of Node !')
            self.node_class = node_class

        #: Dict of nodes in a tree: {node ID (nid): node_instance}.
        self.nodes = {}

        #: Get or set the node ID (nid) of the root. This attribute can be accessed and modified
        #:  with ``.`` and ``=`` operator respectively.
        self.root: Union[Hashable, None] = None

        if tree is not None:
            self.root = tree.root
            for nid, node in tree.nodes.items():
                new_node = deepcopy(node) if deep else node
                self.nodes[nid] = new_node
                if tree.tree_id != self.tree_id:
                    new_node.clone_pointers(tree.tree_id, self.tree_id)

        # Render characters
        self._dt: MutableMapping[str, Tuple[str, str, str]] = {
            'ascii': ('|', '|-- ', '+-- '),
            'ascii-ex': ('\u2502', '\u251c\u2500\u2500 ', '\u2514\u2500\u2500 '),
            'ascii-exr': ('\u2502', '\u251c\u2500\u2500 ', '\u2570\u2500\u2500 '),
            'ascii-em': ('\u2551', '\u2560\u2550\u2550 ', '\u255a\u2550\u2550 '),
            'ascii-emv': ('\u2551', '\u255f\u2500\u2500 ', '\u2559\u2500\u2500 '),
            'ascii-emh': ('\u2502', '\u255e\u2550\u2550 ', '\u2558\u2550\u2550 '),
        }

    # HELPER FUNCTIONS -------------------------------------------------------------------------------------------------
    def _clone(self, tree_id: Union[Hashable, None] = None, deep: bool = False):
        """
        Clone current instance, without the nodes of the tree. Used internally for copying possible other attributes.
        This method intended to be overloaded, by providing a uniform signature to handle differences
         compared to the base class.
        Therefore rewriting whole ``subtree`` and ``pop_subtree`` methods is not necessary when inheriting from Tree.

        For example:
        >>> class TreeWithDescription(Tree):
        >>>     def __init__(self, tree_description: str,
        >>>                  tree=None, deep: bool = False, node_class=None, tree_id: Union[Hashable, None] = None):
        >>>         # Implement (deep) copying of new attributes.
        >>>         self.tree_description = deepcopy(tree_description) if deep else tree_descriptio
        >>>         super().__init__(tree, deep, node_class, tree_id)
        >>>
        >>>     def _clone(self, tree_id: Union[Hashable, None] = None, deep: bool = False):
        >>>         # Return a new instance of the tree object with the new attributes, but provide a uniform signature.
        >>>         return TreeWithDescription(self.tree_description,
        >>>                                    deep=deep,
        >>>                                    tree_id=tree_id,
        >>>                                    )
        >>> my_custom_tree = TreeWithDescription(tree_description='smart tree')
        >>> subtree = my_custom_tree.subtree()  # subtree() does not deal with the extra tree_description attribute.
        >>> subtree.tree_description
        'smart tree'
        """
        return self.__class__(tree_id=tree_id)

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

    @staticmethod
    def _get_lookup_nodes_fun(lookup_nodes):
        if lookup_nodes:
            lookup_nodes_fun = partial(lambda x: x)
        else:
            lookup_nodes_fun = partial(lambda x: x.nid)

        return lookup_nodes_fun

    def _get_nid(self, node):
        """
        Get the node ID (nid) for the given Node instance or node ID (the inverse of ``get_node``, used internally).
        """
        if isinstance(node, self.node_class):
            nid = node.nid

            # Sanity check, only for debuging: The given node equals with the one looked up with nid!
            assert self.nodes[nid] == node
        else:
            nid = node

        if nid not in self.nodes:
            raise NodeIDAbsentError(f'Node ({nid}) is not in the tree!')

        return nid

    def _get_node(self, node):
        """
        Get the Node instance for the given Node instance or node ID (similar to ``get_node``, used internally).
        """
        if isinstance(node, self.node_class):
            # Sanity check, only for debuging: Node is given as parameter and is not found in the tree.
            assert self.nodes.get(node.nid) == node  # None -> AssertionError as node must be self.node_class!
            curr_node = node
        else:
            curr_node = self.nodes.get(node)  # Node ID (nid) is given as parameter,
            if curr_node is None:
                raise NodeIDAbsentError(f'Node ({node}) is not in the tree!')

        return curr_node

    # SIMPLE READER FUNCTIONS ------------------------------------------------------------------------------------------
    def size(self, level: int = None) -> int:
        """
        Get the number of nodes in the whole tree if ``level`` is not given.
        Otherwise, the total number of nodes at specific level is returned.
        0 is returned if too high level is specified and the tree is not deep enough.

        :param level: The level number in the tree. It must be greater or equal to 0, None to return len(tree).
        """
        if level is None:
            return len(self.nodes)  # Same as __len__().
        elif level == 0:
            return 1  # On the root level it is trivially only one node.
        elif level > 0:
            return sum(1 for node in self.nodes.values() if self.depth(node.nid) == level)
        else:
            raise InvalidLevelNumber(f'Level cannot be negative ({level})!')

    def __len__(self) -> int:
        """
        Return the number of nodes (node IDs (nid)) in the tree.
        """
        return len(self.nodes)

    def __getitem__(self, nid: Hashable):
        """
        Return a Node instance for a node ID (nid) if the tree contains it else raises NodeIDAbsentError.
        """
        try:
            return self.nodes[nid]
        except KeyError:
            raise NodeIDAbsentError(f'Node ({nid}) is not in the tree!')

    def __contains__(self, node) -> bool:
        if isinstance(node, self.node_class):
            nid = node.nid
            return self.nodes.get(nid) == node  # Only True if Node instances are equal, node ID (nid) is not enough!
        else:
            return node in self.nodes  # Check if node ID (nid) is in the tree.

    def is_ancestor(self, ancestor, child) -> bool:
        """
        Check if the ``ancestor`` the preceding nodes of ``child``.

        :param ancestor: The Node instance or node ID (nid).
        :param child: The Node instance or node ID (nid).
        :return: True or False
        """
        if self.root is not None:  # if Tree is not empty else -> False!
            ancestor_node = self._get_node(ancestor)  # If ancestor is None -> NodeIDAbsentError.
            ancestor_nid = ancestor_node.nid
            child_node = self._get_node(child)  # If child is None -> NodeIDAbsentError.

            parent_nid = child_node.predecessor(self.tree_id)
            while parent_nid is not None:  # If parent is None we are at the root node -> False.
                parent_node = self.nodes[parent_nid]
                if parent_nid == ancestor_nid and parent_node == ancestor_node:
                    return True
                else:
                    parent_nid = parent_node.predecessor(self.tree_id)  # The parent of the current parent.

        return False

    def depth(self, node=None, filter_fun=lambda x: True) -> int:
        """
        Get the level for the given Node instance or node ID (nid) in the tree.
        If node is None get the maximum depth of this tree.
        If the tree is empty or the checked subtree (including the root) is filtered out returns NodeIDAbsentError.

        :param node: The Node instance or node ID (nid).
        :param filter_fun: a function with one parameter executed on a :class:`Node` object.
            When this parameter is specified, the traversing will NOT visit those nodes
            (and their children) which does not pass filtering (evaluates False).
        :return int: An integer (level) starting with 0 at the root. In other words, the root lives at level 0.
        :throw NodeIDAbsentError:
        """
        if node is None:  # Get the maximum level of this tree.
            level = max((sum(1 for _ in self.busearch(leaf.nid, filter_fun)) - 1 for leaf in self.leaves()), default=0)
        else:  # Get level of the given node
            level = sum(1 for _ in self.busearch(node, filter_fun)) - 1

        if level < 0:
            raise NodeIDAbsentError('All nodes are filtered out!')  # TODO Is this solution good?

        return level

    # Node returing READER FUNCTIONS -----------------------------------------------------------------------------------
    def get_node(self, nid: Hashable):
        """
        Get the the Node instance with node ID ``nid``.

        An alternative way is using ``__getitem__`` ('[]') operation on the tree.
        But ``get_node()`` will return None if ``nid`` is absent, whereas '[]' will raise ``KeyError``.
        """
        return self.nodes.get(nid)

    def get_nodes(self, filter_fun=None, lookup_nodes: bool = True):
        """
        Returns all Node instances in an iterator if filter function is not specified or None.
        Else traverses the tree top down and filters subtrees by the given function.

        :param filter_fun: a function with one parameter executed on a :class:`Node` object.
            When this parameter is specified, the traversing will NOT visit those nodes
            (and their children) which does not pass filtering (evaluates False).
        :param lookup_nodes: return Node instances (default) or node IDs (nids).
        :return: return an iterator of Node instances (default) or node IDs (nids).
        """
        if filter_fun is None:
            lookup_nodes_fun = self._get_lookup_nodes_fun(lookup_nodes)
            return (lookup_nodes_fun(curr_node) for curr_node in self.nodes.values())
        else:
            return self.expand_tree(self.root, filter_fun=filter_fun, lookup_nodes=lookup_nodes)

    def parent(self, node, level: int = -1, lookup_nodes: bool = True):
        """
        For a given Node instance or node ID (nid), get parent Node instance at a given level.
        Cornercases are evaluated in this order:
        - If level equals 0 the root Node instance is returned.
        - If the given Node instance or node ID (nid) equals root node None is returned.
        - If no level is provided or level equals -1, the parent Node instance is returned.
        NodeIDAbsentError exception is raised if the ``node`` does not exist in the tree or equals None.
        """
        nid = self._get_nid(node)  # If node is None -> NodeIDAbsentError.

        if lookup_nodes:
            lookup_nodes_fun = partial(lambda x: self.nodes[x])
        else:
            lookup_nodes_fun = partial(lambda x: x)

        if level == 0:  # Root level of any element -> Root Node instance or node ID (nid).
            return lookup_nodes_fun(self.root)
        elif nid == self.root:  # Root node for every non-root level -> None.
            return None

        tree_id = self.tree_id
        ancestor = self.nodes[nid].predecessor(tree_id)  # Direct parent nid (None for root node is alread handled).
        if level == -1:  # Direct parent is required.
            return lookup_nodes_fun(ancestor)  # Parent Node instance (root node where parent is None already handled).
        elif level >= self.depth(nid):  # Root node is already handled, so depth(nid) cannot be <= 0.
            raise InvalidLevelNumber(f'The given node\'s level ({self.depth(nid)}) must be greater than its '
                                     f'parent\'s level ({level})!')

        # Ascend to the appropriate level.
        while ancestor is not None:
            ascendant_level = self.depth(ancestor)
            if ascendant_level == level:
                return lookup_nodes_fun(ancestor)
            else:
                ancestor = self.nodes[ancestor].predecessor(tree_id)  # Get the parent of the current node.

    def children(self, node, filter_fun=lambda x: True, lookup_nodes: bool = True):
        """
        Return the iterator of direct children (IDs or Node instance) of the given Node instance or node ID (nid).
        If there are no children or all the children are filtered out return empty iterator.
        NodeIDAbsentError exception is thrown if the ``node`` does not exist in the tree or equals None.
        """

        if lookup_nodes:  # Store decision on lookup for later use.
            lookup_nodes_fun = partial(lambda x, y: y)
        else:
            lookup_nodes_fun = partial(lambda x, y: x)

        curr_node = self._get_node(node)  # If node is None -> NodeIDAbsentError.
        for child_nid in curr_node.successors(self.tree_id):
            child_node = self.nodes[child_nid]
            if filter_fun(child_node):
                yield lookup_nodes_fun(child_nid, child_node)

    def siblings(self, node, filter_fun=lambda x: True, lookup_nodes: bool = True):
        """
        Return the iterator of siblings of the given Node instance or node ID (nid) ``node``.
        If ``node`` is root (there are no siblings) or all of them are filtered, empty iterator is returned.
        NodeIDAbsentError exception is thrown if the ``node`` does not exist in the tree or equals None.
        """
        nid = self._get_nid(node)  # If node is None -> NodeIDAbsentError.
        if nid == self.root:
            return

        direct_parent_id = self.nodes[nid].predecessor(self.tree_id)
        parents_children_ids = self.nodes[direct_parent_id].successors(self.tree_id)
        if filter_fun is None and not lookup_nodes:
            siblings_it = (sid for sid in parents_children_ids if sid != nid)  # All sibling IDs without hassle.
        else:
            # Must lookup nodes either way.
            sibling_nodes_it = (self.nodes[i] for i in parents_children_ids if i != nid)
            lookup_nodes_fun = self._get_lookup_nodes_fun(lookup_nodes)
            siblings_it = (lookup_nodes_fun(i) for i in sibling_nodes_it if filter_fun(i))

        yield from siblings_it

    def leaves(self, node=None, filter_fun=None, lookup_nodes: bool = True):
        """
        Get the iterator of leaves of the whole tree (if node is None)
         or a subtree (if node is a Node instance or node ID (nid)).
        If fiter_fun is set leaves in the filtered subtree are not considered.
        If tree is empty (i.e. it has no root node) empty iterator is returned.
        """
        if node is not None:  # Leaves for a specific subtree.
            nid = self._get_nid(node)
        elif self.root is None:  # Empty tree -> empty iterator.
            return
        else:
            nid = self.root  # Whole tree -> all leaves.

        lookup_nodes_fun = self._get_lookup_nodes_fun(lookup_nodes)

        if filter_fun is None or nid == self.root:
            subtree_node_it = self.nodes.values()  # No filter. from root node -> All nodes are considered.
        else:  # Subtree or filtered tree!
            subtree_node_it = (node for node in self.expand_tree(nid, filter_fun=filter_fun, lookup_nodes=True))

        yield from (lookup_nodes_fun(node) for node in subtree_node_it if node.is_leaf(self.tree_id))

    def paths_to_leaves(self, node=None, filter_fun=lambda x: True):
        """
        Get the identifiers allowing to go from the given node to each leaf.

        :return: an iterator of tuples of identifiers, root is included into the path.

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

        for leaf in self.leaves(node):
            node_ids = [nid for nid in self.busearch(leaf, filter_fun, lookup_nodes=False)]
            node_ids.reverse()
            yield tuple(node_ids)

    def busearch(self, node, filter_fun=lambda x: True, lookup_nodes: bool = True):
        """
        Traverse the tree branch bottom-up along the branch from a given Node instance or node ID (nid)
         to its ancestors until root.
        NodeIDAbsentError exception is thrown if the ``node`` does not exist in the tree or equals None.

        :param node: The Node instance or node ID (nid).
        :param filter_fun: a function with one parameter executed on a :class:`Node` object.
            When this parameter is specified, the traversing will NOT visit those nodes
            (and their children) which does not pass filtering (evaluates False).
        :param lookup_nodes: return Node instances (default) or node IDs (nids).
        """
        if lookup_nodes:  # Store decision on lookup for later use.
            lookup_nodes_fun = partial(lambda x, y: y)
        else:
            lookup_nodes_fun = partial(lambda x, y: x)

        ret_node_list = []
        current_node = self._get_node(node)  # If node is None -> NodeIDAbsentError.
        current_nid = current_node.nid
        while current_nid is not None:
            if filter_fun(current_node):
                ret_node_list.append(lookup_nodes_fun(current_nid, current_node))
            else:
                return  # Subtree filtered out -> empty iterator.
            if self.root != current_nid:
                current_nid = current_node.predecessor(self.tree_id)  # Parent of current node.
                current_node = self.nodes[current_nid]
            else:
                current_nid = None

        yield from ret_node_list

    def expand_tree(self, node=None, mode: int = DEPTH, filter_fun=lambda x: True, key=lambda x: x,
                    reverse: bool = False, lookup_nodes: bool = False):
        """
        Python generator to traverse the tree (or a subtree) top-down with optional node filtering and sorting.

        :param node: The Node instance or node ID (nid) from which tree traversal will start.
             If None tree root will be used.
        :param mode: Traversal mode, may be either ``DEPTH``, ``WIDTH`` or ``ZIGZAG``.
        :param filter_fun: a function with one parameter executed on a :class:`Node` object.
            When this parameter is specified, the traversing will NOT visit those nodes
            (and their children) which does not pass filtering (evaluates False).
        :param key: the ``key`` and ``reverse`` are present to sort nodes at each level.
            If ``key`` is None the function returns nodes in original insertion order.
        :param reverse: if True reverse ordering.
        :param lookup_nodes: return Node instances (default) or node IDs (nids).
        :return: Node IDs that satisfy the conditions in the defined order.
        :rtype: generator object.
        """
        # TODO expand_tree and _print_backend and to_dict? What is the difference?
        if node is not None:
            current_node = self._get_node(node)
        elif self.root is None:
            return
        else:  # node is None -> Traverse the full tree from root.
            current_node = self.nodes[self.root]

        if not filter_fun(current_node):  # Subtree filtered out.  # TODO should not filter the parameter node! Why?
            return

        lookup_nodes_fun = self._get_lookup_nodes_fun(lookup_nodes)

        yield lookup_nodes_fun(current_node)  # Yield current Node or node ID (nid).

        queue = self.children(current_node, filter_fun)
        if mode in {self.DEPTH, self.WIDTH}:
            # Set sort key fun
            sort_fun = self._create_sort_fun(key, reverse)

            # Set tree traversal order.
            if mode is self.DEPTH:
                order_fun = partial(lambda x, y: chain(x, y))  # Depth-first.
            else:
                order_fun = partial(lambda x, y: chain(y, x))  # Width-first.

            queue = list(sort_fun(queue))  # Sort children.
            while len(queue) > 0:
                cn = queue[0]
                yield lookup_nodes_fun(cn)

                expansion = list(sort_fun(self.children(cn, filter_fun)))  # Sort children.
                queue = list(order_fun(expansion, queue[1:]))  # Step.

        elif mode == self.ZIGZAG:
            # Zigzag traversal means starting from left to right, then right to left for the next level
            # and then again left to right and so on in an alternate manner.
            stack_fw = []
            queue = list(queue)
            queue.reverse()
            stack = stack_bw = queue
            forward = False
            while len(stack) > 0:
                current_node = stack.pop(0)
                yield lookup_nodes_fun(current_node)
                expansion = list(self.children(current_node, filter_fun))
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

    # MODIFYING FUNCTIONS (Node) ---------------------------------------------------------------------------------------
    def create_node(self, tag: Any = None, nid: Hashable = None, data: Any = None, parent: Union[Hashable, None] = None,
                    update: bool = True):
        """
        Create a child node for given ``parent`` node and add it to the tree.
         If node identifier (``nid``) is absent, a UUID will be generated automatically.
        """
        node = self.node_class(tag=tag, nid=nid, data=data)
        self.add_node(node, parent, update)
        return node

    def add_node(self, node, parent=None, update: bool = True) -> None:
        """
        Add a previously created Node object to the tree and register its parent (root by default).
        Update ``node`` if update is True and it is in the tree replace the node object
         else raise DuplicatedNodeIDError.

        The ``node`` parameter refers to an instance of Class::Node.
        """
        if not isinstance(node, self.node_class):
            raise TypeError(f'First parameter must be object of {self.node_class} !')

        nid = node.nid
        if nid in self.nodes and not update:
            raise DuplicatedNodeIDError(f'Cannot create node with ID {nid} !')
        elif nid not in self.nodes:  # Regardless of update.
            # Register parent and parent's children if not updtaing existing node.
            if parent is not None:  # Parent can be None!
                pid = self._get_nid(parent)  # Implicitly check if parent is in the tree.
            else:
                pid = None

            if pid is None:  # Adding root node.
                if self.root is not None:
                    raise MultipleRootError('A tree takes one root merely!')
                else:
                    self.root = nid
            else:  # pid is not None and pid in self.nodes -> Updating non-root node's parent.
                self.nodes[pid].add_successor(nid, self.tree_id)

            node.set_predecessor(pid, self.tree_id)

        self.nodes[nid] = node

    def update_node(self, node_to_update, **attrs) -> None:
        """
        Update node's attributes.

        :param node_to_update: The Node instance or node ID (nid) of the node to be updated.
        :param attrs: attribute pairs recognized by Node object (Beware, attributes unconditionally updated!)
        :return: None
        """
        cn = self._get_node(node_to_update)

        new_identifier_val = attrs.pop('nid', None)
        if new_identifier_val is not None:
            # Updating node id meets following constraints:
            # * Check if the new node ID (nid) is not a duplicate.
            if new_identifier_val in self.nodes:
                raise DuplicatedNodeIDError(f'Cannot create node with ID {new_identifier_val} !')
            # * Update node ID (nid) property.
            if cn.is_in_multiple_trees():
                raise NodePropertyError('Cannot update node ID as node is in multiple trees!')
            nid_to_update = cn.nid
            cn.nid = new_identifier_val

            # * Update tree registration of nodes.
            self.nodes.pop(nid_to_update)
            self.nodes[new_identifier_val] = cn

            # * Update tree root if necessary.
            if self.root == nid_to_update:
                self.root = new_identifier_val
            else:  # * Update parent's child name registry only if non-root element is updated!
                parent_nid = cn.predecessor(self.tree_id)
                if parent_nid is not None:
                    self.nodes[parent_nid].replace_successor(nid_to_update, self.tree_id, new_identifier_val)

            # * Update children's parents registry.
            for fp in cn.successors(self.tree_id):
                self.nodes[fp].set_predecessor(new_identifier_val, self.tree_id)

        for attr, val in attrs.items():
            setattr(cn, attr, val)  # Potentially dangerous operation!

    def move_node(self, source, destination) -> None:
        """
        Move node ID or Node instance ``source`` (with its subree) from its parent to another parent ``destination``.
        """
        source_nid = self._get_nid(source)
        destination_nid = self._get_nid(destination)

        # Source node can not be root (all nodes has root as acncestor), but destination can be!
        if self.is_ancestor(source, destination):
            raise LoopError(f'Source ({source_nid}) node ID is an ancestor of '
                            f'destination ({destination_nid}) node ID in the tree!')

        # Update old and new parents:
        # 1. Get parent node (Parent can not be None as it woluld mean source node is the root
        #  which case is already handled by LoopError).
        parent_nid = self.nodes[source_nid].predecessor(self.tree_id)
        # 2. Remove source from parent.
        self.nodes[parent_nid].remove_successor(source_nid, self.tree_id)
        # 3. Register source as a child of destination.
        self.nodes[destination_nid].add_successor(source_nid, self.tree_id)
        # 4. Register new parent (destination) for source.
        self.nodes[source_nid].set_predecessor(destination_nid, self.tree_id)

    def remove_node(self, node) -> None:
        """
        Delete a node by linking past it.

        For example, if we have `a -> node -> c` and delete node ``node``, we are left with `a -> c`.
        """
        nid = self._get_nid(node)

        if self.root == nid:
            raise LinkPastRootNodeError('Cannot link past the root node, delete it with remove_subtree()!')
        # Get the parent of the node we are linking past
        parent_node_nid = self.nodes[nid].predecessor(self.tree_id)
        parent_node = self.nodes[parent_node_nid]

        # Delete the node from the parent and from the nodes registry
        node_to_be_removed = self.nodes.pop(nid)
        parent_node.remove_successor(nid, self.tree_id)
        # Link parent with children
        for child in node_to_be_removed.successors(self.tree_id):
            # Set the children of the node to the parent
            self.nodes[child].set_predecessor(parent_node_nid, self.tree_id)
            # Move the children from the current node to the parent node
            parent_node.add_successor(child, self.tree_id)

    # MODIFYING FUNCTIONS (subtree) ------------------------------------------------------------------------------------
    def subtree(self, node, tree_id: Union[Hashable, None] = None, deep: bool = False):
        """
        Return a shallow or deep copy of subtree with ``node`` being the new root.

        :param node: The Node instance or node ID (nid) that will be the new root.
        :param tree_id: The ID of the new tree.
        :param deep: Indicates if the elements of the tree will be deep copied or not.
        :return: A new Tree instance.
        """
        new_root_nid = self._get_nid(node)

        # Create a new tree without nodes, but with possible other attributes (deep) copied.
        st = self._clone(tree_id, deep=deep)
        actual_tree_id = st.tree_id  # If tree_id is None the actual tree_id is generated!

        # Set root.
        st.root = new_root_nid
        # Copy nodes.
        for node_n_nid in self.expand_tree(new_root_nid):
            node_n = self.nodes[node_n_nid]
            st.nodes[node_n_nid] = deepcopy(node_n) if deep else node_n
            # Define nodes parent/children in this tree.
            st.nodes[node_n_nid].clone_pointers(self.tree_id, actual_tree_id)

        # All pointers are the same, except the root. The parent of the root for the new tree must be None!
        st.nodes[new_root_nid].set_predecessor(None, actual_tree_id)

        return st

    def remove_subtree(self, node) -> int:
        """
        Remove a Node instance or node ID (nid) indicated by ``node`` with all its successors.
        Return the number of removed nodes.
        """
        nid = self._get_nid(node)

        # TODO when removing a node all reference for the actual tree must be explicitly deleted
        #  else is_in_multiple_trees() will not work!
        if nid == self.root:
            # This tree will be empty, but the nodes may occur in other trees, so we meed to update each separately!
            self.root = None
        else:
            # Update parent info (if nid is not root):
            # In the original tree, the removed nid will be unreferenced from its parents children.
            parent = self.nodes[nid].predecessor(self.tree_id)
            self.nodes[parent].remove_successor(nid, self.tree_id)
            self.nodes[nid].set_predecessor(None, self.tree_id)

        # Remove node and its children
        # We must generate the full list of nodes to be removed before modifying the tree!
        nids_removed = list(self.expand_tree(nid))
        for curr_nid in nids_removed:
            # Remove node (from this tree)
            curr_node = self.nodes.pop(curr_nid)
            # Remove parent (from this tree)
            curr_node.set_predecessor(None, self.tree_id)
            # Remove children (from this tree)
            for cid in curr_node.successors(self.tree_id):
                curr_node.remove_successor(cid, self.tree_id)

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
            self.nodes[parent].remove_successor(nid, self.tree_id)

        nids_removed = list(self.expand_tree(nid))
        for curr_nid in nids_removed:
            # Remove node (from this tree)
            curr_node = self.nodes.pop(curr_nid)
            # Clone pointers to the new tree
            curr_node.clone_pointers(self.tree_id, st.tree_id)
            # Because we reuse the Node instance, we must clean all reference to the old tree!
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

        for child_nid in new_tree.children(new_tree.root, lookup_nodes=False):
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
            self.nodes[nid].add_successor(new_tree.root, self.tree_id)

    # PRINT RELATED FUNCTIONS ------------------------------------------------------------------------------------------
    def __str__(self):
        return self.show()

    def show(self, node=None, level=0, filter_fun=lambda x: True, key=lambda x: x, reverse=False,
             line_type='ascii-ex', get_label_fun=lambda node: node.tag, record_end='\n'):
        """
        Another implementation of printing tree using Stack Print tree structure in hierarchy style.
        Return the tree structure as string in hierarchy style.

        :param node: the reference Node instance or node ID (nid) to start expanding.
        :param level: the node level in the tree (root as level 0).
        :param filter_fun: a function with one parameter executed on a :class:`Node` object.
            When this parameter is specified, the traversing will NOT visit those nodes
            (and their children) which does not pass filtering (evaluates False).
        :param key: the ``key`` param for sorting :class:`Node` objects in the same level.
        :param reverse: the ``reverse`` param for sorting :class:`Node` objects in the same level.
        :param line_type: such as 'ascii' (default), 'ascii-ex', 'ascii-exr', 'ascii-em', 'ascii-emv', 'ascii-emh'
            to the change graphical form.
        :param get_label_fun: A function to define how to print labels
        :param record_end: The ending character for each record (e.g. newline)
        :return: None

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
        """
        return ''.join(self.show_iter(node, level, filter_fun, key, reverse, line_type, get_label_fun, record_end))

    def show_iter(self, node=None, level=0, filter_fun=lambda x: True, key=lambda x: x, reverse=False,
                  line_type='ascii-ex', get_label_fun=lambda node: node.tag, record_end=''):
        """
        Same as show(), but returns an iterator.
        """
        hack = True  # TODO hack!
        for pre, curr_node in self._print_backend(node, level, filter_fun, key, reverse, line_type):
            hack = False
            label = get_label_fun(curr_node)
            yield f'{pre}{label}{record_end}'
        if hack:
            yield f'{self.__class__.__name__}()'

    def _print_backend(self, node, level, filter_fun, key, reverse, line_type):
        """
        Set up depth-fist search

        A more elegant way to achieve this function using Stack structure, for constructing the Nodes Stack
         push and pop nodes with additional level info.
        """
        if node is not None:
            current_node = self._get_node(node)
        elif self.root is None:
            return
        else:
            current_node = self.nodes[self.root]

        if not filter_fun(current_node):  # TODO should not filter the parameter node!
            return

        # Set sort key and reversing if needed
        sort_fun = self._create_sort_fun(key, reverse)

        # Set line types
        line_elems = self._dt.get(line_type)
        if line_elems is None:
            raise ValueError(f'Undefined line type ({line_type})! Must choose from {set(self._dt)}!')

        yield from self._print_backend_recursive(current_node, level, filter_fun, sort_fun, [], *line_elems)

    def _print_backend_recursive(self, node, level, filter_fun, sort_fun, is_last,
                                 dt_vertical_line, dt_line_box, dt_line_corner):
        # Format current level
        lines = []
        if level > 0:
            for curr_is_last in is_last[:-1]:
                if curr_is_last:
                    line = ' ' * 4
                else:
                    line = dt_vertical_line + ' ' * 3
                lines.append(line)

            if is_last[-1]:
                lines.append(dt_line_corner)
            else:
                lines.append(dt_line_box)

        yield ''.join(lines), node  # Yield the current level

        # Go deeper!
        children = list(self.children(node, filter_fun))
        idxlast = len(children) - 1

        for idx, child in enumerate(sort_fun(children)):
            is_last.append(idx == idxlast)
            yield from self._print_backend_recursive(child, level + 1, filter_fun, sort_fun, is_last,
                                                     dt_vertical_line, dt_line_box, dt_line_corner)
            is_last.pop()

    def to_dict(self, node=None, filter_fun=lambda x: True, key=lambda x: x, reverse=False, with_data=False) -> dict:
        """
        Transform the whole tree into a dict.
        """
        if node is not None:
            current_node = self._get_node(node)
        elif self.root is None:
            return {}
        else:
            current_node = self.nodes[self.root]

        if not filter_fun(current_node):   # TODO should not filter the parameter node!
            return {}

        # Set sort key and reversing if needed
        sort_fun = self._create_sort_fun(key, reverse)

        ntag = current_node.tag
        tree_dict = {ntag: {'children': []}}
        if with_data:
            tree_dict[ntag]['data'] = current_node.data

        children = []
        for elem in sort_fun(self.children(current_node, filter_fun)):
            dict_form = self.to_dict(elem, filter_fun, key, reverse, with_data)  # TODO recursive!
            children.append(dict_form)

        tree_dict[ntag]['children'] = children

        if len(children) == 0:  # Handle leaves
            if not with_data:
                tree_dict = current_node.tag
            else:
                tree_dict = {ntag: {'data': current_node.data}}

        return tree_dict

    def to_graphviz(self, node=None, filter_fun=lambda x: True, key=lambda x: x, reverse=False,
                    shape='circle', graph='digraph'):
        """
        Exports the tree in the dot format of the graphviz software as string generator.
        """

        yield f'{graph} tree {{\n'

        if self.root is not None:

            connections = []
            for curr_node in self.expand_tree(node, self.WIDTH, filter_fun, key, reverse, lookup_nodes=True):
                nid = curr_node.nid
                state = f'"{nid}" [label="{curr_node.tag}", shape={shape}]'
                yield f'\t{state}\n'  # Yield nodes

                for cid in self.children(curr_node, lookup_nodes=False):  # Accumulate connections
                    connections.append(f'"{nid}" -> "{cid}"')

            if len(connections) > 0:  # Write connections after all nodes have been written
                yield '\n'
                yield from (f'\t{c}\n' for c in connections)

        yield '}\n'
