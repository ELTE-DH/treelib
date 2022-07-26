#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

import os

import unittest
from treelib import Tree, Node
from treelib.tree import NodeIDAbsentError, LoopError, DuplicatedNodeIDError


class TreeCase(unittest.TestCase):
    def setUp(self):
        """
        Hárry
        ├── Jane
        │   └── Diane
        └── Bill
            └── George
        """
        tree = Tree(tree_id='tree 1')
        tree.create_node('Hárry', 'hárry')
        tree.create_node('Jane', 'jane', parent='hárry')
        tree.create_node('Bill', 'bill', parent='hárry')
        tree.create_node('Diane', 'diane', parent='jane')
        tree.create_node('George', 'george', parent='bill')

        self.tree = tree
        self.copytree = Tree(self.tree, deep=True)

    @staticmethod
    def get_t1():
        """
        root
        ├── A
        │   └── A1
        └── B
        """
        t = Tree(tree_id='t1')
        t.create_node(tag='root', nid='r')
        t.create_node(tag='A', nid='a', parent='r')
        t.create_node(tag='B', nid='b', parent='r')
        t.create_node(tag='A1', nid='a1', parent='a')
        return t

    @staticmethod
    def get_t2():
        """
        root2
        ├── C
        └── D
            └── D1
        """
        t = Tree(tree_id='t2')
        t.create_node(tag='root2', nid='r2')
        t.create_node(tag='C', nid='c', parent='r2')
        t.create_node(tag='D', nid='d', parent='r2')
        t.create_node(tag='D1', nid='d1', parent='d')
        return t

    def test_tree(self):
        self.assertEqual(isinstance(self.tree, Tree), True)
        self.assertEqual(isinstance(self.copytree, Tree), True)

    def test_is_root(self):
        self.assertTrue(self.tree.nodes['hárry'].is_root('tree 1'))
        self.assertFalse(self.tree.nodes['jane'].is_root('tree 1'))

    def test_tree_wise_is_root(self):
        subtree = self.tree.subtree('jane', tree_id='subtree 2')
        # harry is root of tree 1 but not present in subtree 2
        self.assertTrue(self.tree.nodes['hárry'].is_root('tree 1'))
        self.assertNotIn('hárry', subtree.nodes)
        # jane is not root of tree 1 but is root of subtree 2
        self.assertFalse(self.tree.nodes['jane'].is_root('tree 1'))
        self.assertTrue(subtree.nodes['jane'].is_root('subtree 2'))

    def test_paths_to_leaves(self):
        paths = list(self.tree.paths_to_leaves())
        self.assertEqual(len(paths), 2)
        self.assertTrue(('hárry', 'jane', 'diane') in paths)
        self.assertTrue(('hárry', 'bill', 'george') in paths)

    def test_nodes(self):
        self.assertEqual(len(self.tree.nodes), 5)
        self.assertEqual(len(list(self.tree.get_nodes())), 5)
        self.assertEqual(self.tree.size(), 5)
        self.assertEqual(self.tree.get_node('jane').tag, 'Jane')
        self.assertEqual('jane' in self.tree, True)
        self.assertEqual('alien' in self.tree, False)
        self.tree.create_node('Alien', 'alien', parent='jane')
        self.assertEqual('alien' in self.tree, True)
        self.tree.remove_subtree('alien')
        self.assertEqual(len(self.tree), 5)

    def test_getitem(self):
        """
        Nodes can be accessed via getitem.
        """
        for node_id in self.tree.nodes:
            try:
                self.tree[node_id]
            except NodeIDAbsentError:
                self.fail('Node access should be possible via getitem.')
        try:
            self.tree['root']
        except NodeIDAbsentError:
            pass
        else:
            self.fail('There should be no default fallback value for getitem.')

    def test_parent(self):
        for nid in self.tree.nodes:
            if nid == self.tree.root:
                self.assertEqual(self.tree.parent(nid), None)
            else:
                self.assertEqual(self.tree.parent(nid) in self.tree.get_nodes(), True)

    def test_ancestor(self):
        for nid in self.tree.nodes:
            if nid == self.tree.root:
                self.assertEqual(self.tree.parent(nid), None)
            else:
                for level in range(self.tree.depth(nid) - 1, 0, -1):
                    self.assertEqual(self.tree.parent(nid, level=level) in self.tree.get_nodes(), True)

    def test_children(self):
        for nid in self.tree.nodes:
            children = list(self.tree.children(nid, lookup_nodes=False))
            for child in children:
                self.assertEqual(self.tree[child] in self.tree.get_nodes(), True)
            children = list(self.tree.children(nid))
            for child in children:
                self.assertEqual(child in self.tree.get_nodes(), True)
        try:
            list(self.tree.children('alien'))
        except NodeIDAbsentError:
            pass
        else:
            self.fail('The absent node should be declaimed.')

    def test_remove_node(self):
        self.tree.create_node('Jill', 'jill', parent='george')
        self.tree.create_node('Mark', 'mark', parent='jill')
        self.assertEqual(self.tree.remove_subtree('jill'), 2)
        self.assertEqual(self.tree.get_node('jill') is None, True)
        self.assertEqual(self.tree.get_node('mark') is None, True)

    def test_tree_wise_depth(self):
        # Try getting the level of this tree
        self.assertEqual(self.tree.depth(), 2)
        self.tree.create_node('Jill', 'jill', parent='george')
        self.assertEqual(self.tree.depth(), 3)
        self.tree.create_node('Mark', 'mark', parent='jill')
        self.assertEqual(self.tree.depth(), 4)

        # Try getting the level of the node
        """
        self.tree.show()
        Hárry
        |___ Bill
        |    |___ George
        |         |___ Jill
        |              |___ Mark
        |___ Jane
        |    |___ Diane
        """
        self.assertEqual(self.tree.depth(self.tree.get_node('mark')), 4)
        self.assertEqual(self.tree.depth(self.tree.get_node('jill')), 3)
        self.assertEqual(self.tree.depth(self.tree.get_node('george')), 2)
        self.assertEqual(self.tree.depth('jane'), 1)
        self.assertEqual(self.tree.depth('bill'), 1)
        self.assertEqual(self.tree.depth('hárry'), 0)

        # Try getting Exception
        node = Node('Test One', 'identifier 1')
        self.assertRaises(AssertionError, self.tree.depth, node)

        # Reset the test case
        self.tree.remove_subtree('jill')

    def test_leaves(self):
        leaves = list(self.tree.leaves())
        for nid in self.tree.expand_tree():
            self.assertEqual((self.tree[nid].is_leaf('tree 1')) == (self.tree[nid] in leaves), True)
        leaves = list(self.tree.leaves(node='jane'))
        for nid in self.tree.expand_tree(node='jane'):
            self.assertEqual(self.tree[nid].is_leaf('tree 1') == (self.tree[nid] in leaves), True)

    def test_tree_wise_leaves(self):
        leaves = list(self.tree.leaves())
        for nid in self.tree.expand_tree():
            self.assertEqual((self.tree[nid].is_leaf('tree 1')) == (self.tree[nid] in leaves), True)
        leaves = list(self.tree.leaves(node='jane'))
        for nid in self.tree.expand_tree(node='jane'):
            self.assertEqual(self.tree[nid].is_leaf('tree 1') == (self.tree[nid] in leaves), True)

    def test_link_past_node(self):
        self.tree.create_node('Jill', 'jill', parent='hárry')
        self.tree.create_node('Mark', 'mark', parent='jill')
        self.assertEqual('mark' not in self.tree.children('hárry', lookup_nodes=False), True)
        self.tree.remove_node('jill')
        self.assertEqual('mark' in self.tree.children('hárry', lookup_nodes=False), True)

    def test_expand_tree(self):
        # # Default config
        # Hárry
        #   |-- Jane
        #       |-- Diane
        #   |-- Bill
        #       |-- George
        # Traverse in depth first mode preserving insertion order
        nodes = [nid for nid in self.tree.expand_tree(key=None)]
        self.assertEqual(nodes, [u'h\xe1rry', u'jane', u'diane', u'bill', u'george'])
        self.assertEqual(len(nodes), 5)

        # By default traverse depth first and sort child nodes by node tag
        nodes = [nid for nid in self.tree.expand_tree()]
        self.assertEqual(nodes, [u'h\xe1rry', u'bill', u'george', u'jane', u'diane'])
        self.assertEqual(len(nodes), 5)

        # # Expanding from specific node
        nodes = [nid for nid in self.tree.expand_tree(node='bill')]
        self.assertEqual(nodes, [u'bill', u'george'])
        self.assertEqual(len(nodes), 2)

        # # Changing into width mode preserving insertion order
        nodes = [nid for nid in self.tree.expand_tree(mode=Tree.WIDTH, key=None)]
        self.assertEqual(nodes, [u'h\xe1rry', u'jane', u'bill', u'diane', u'george'])
        self.assertEqual(len(nodes), 5)

        # Breadth first mode, child nodes sorting by tag
        nodes = [nid for nid in self.tree.expand_tree(mode=Tree.WIDTH)]
        self.assertEqual(nodes, [u'h\xe1rry', u'bill', u'jane',  u'george', u'diane'])
        self.assertEqual(len(nodes), 5)

        # # Expanding by filters
        # Stops at root
        nodes = [nid for nid in self.tree.expand_tree(filter_fun=lambda x: x.tag == 'Bill')]
        self.assertEqual(len(nodes), 0)
        nodes = [nid for nid in self.tree.expand_tree(filter_fun=lambda x: x.tag != 'Bill')]
        self.assertEqual(nodes, [u'h\xe1rry', u'jane', u'diane'])
        self.assertEqual(len(nodes), 3)

    def test_move_node(self):
        diane_parent = self.tree.parent('diane')
        self.tree.move_node('diane', 'bill')
        self.assertEqual('diane' in self.tree.children('bill', lookup_nodes=False), True)
        self.tree.move_node('diane', diane_parent.nid)

    def test_paste_tree(self):
        new_tree = Tree()
        new_tree.create_node('Jill', 'jill')
        new_tree.create_node('Mark', 'mark', parent='jill')
        self.tree.paste_subtree('jane', new_tree)
        self.assertEqual('jill' in self.tree.children('jane', lookup_nodes=False), True)

        self.assertEqual(self.tree.show(), """Hárry
├── Bill
│   └── George
└── Jane
    ├── Diane
    └── Jill
        └── Mark
""")
        self.tree.remove_subtree('jill')
        self.assertNotIn('jill', self.tree.nodes.keys())
        self.assertNotIn('mark', self.tree.nodes.keys())
        self.tree.show()
        self.assertEqual(self.tree.show(), """Hárry
├── Bill
│   └── George
└── Jane
    └── Diane
""")

    def test_merge(self):

        # Merge on empty initial tree
        t1 = Tree(tree_id='t1')
        t2 = self.get_t2()
        # t1.merge_subtree(node=None, other_tree=t2)

        self.assertEqual(t2.tree_id, 't2')
        self.assertEqual(t2.root, 'r2')
        self.assertEqual(set(t2.nodes.keys()), {'r2', 'c', 'd', 'd1'})
        self.assertEqual(t2.show(), """root2
├── C
└── D
    └── D1
""")

        # Merge empty other_tree (on root)
        t1 = self.get_t1()
        t2 = Tree(tree_id='t2')
        t1.merge_subtree(node='r', other_tree=t2)

        self.assertEqual(t1.tree_id, 't1')
        self.assertEqual(t1.root, 'r')
        self.assertEqual(set(t1.nodes.keys()), {'r', 'a', 'a1', 'b'})
        self.assertEqual(t1.show(), """root
├── A
│   └── A1
└── B
""")

        # Merge at root
        t1 = self.get_t1()
        t2 = self.get_t2()
        t1.merge_subtree(node='r', other_tree=t2)

        self.assertEqual(t1.tree_id, 't1')
        self.assertEqual(t1.root, 'r')
        self.assertNotIn('r2', t1.nodes.keys())
        self.assertEqual(set(t1.nodes.keys()), {'r', 'a', 'a1', 'b', 'c', 'd', 'd1'})
        self.assertEqual(t1.show(), """root
├── A
│   └── A1
├── B
├── C
└── D
    └── D1
""")

        # Merge on node
        t1 = self.get_t1()
        t2 = self.get_t2()
        t1.merge_subtree(node='b', other_tree=t2)
        self.assertEqual(t1.tree_id, 't1')
        self.assertEqual(t1.root, 'r')
        self.assertNotIn('r2', t1.nodes.keys())
        self.assertEqual(set(t1.nodes.keys()), {'r', 'a', 'a1', 'b', 'c', 'd', 'd1'})
        self.assertEqual(t1.show(), """root
├── A
│   └── A1
└── B
    ├── C
    └── D
        └── D1
""")

    def test_paste(self):

        # Paste under root
        t1 = self.get_t1()
        t2 = self.get_t2()
        t1.paste_subtree(node='r', other_tree=t2)
        self.assertEqual(t1.tree_id, 't1')
        self.assertEqual(t1.root, 'r')
        self.assertEqual(t1.parent('r2').nid, 'r')
        self.assertEqual(set(t1.nodes.keys()), {'r', 'r2', 'a', 'a1', 'b', 'c', 'd', 'd1'})
        self.assertEqual(t1.show(), """root
├── A
│   └── A1
├── B
└── root2
    ├── C
    └── D
        └── D1
""")

        # Paste under non-existing node
        t1 = self.get_t1()
        t2 = self.get_t2()
        with self.assertRaises(NodeIDAbsentError) as e:
            t1.paste_subtree(node='not_existing', other_tree=t2)
        self.assertEqual(e.exception.args[0], 'Node (not_existing) is not in the tree!')

        # Paste under None nid
        t1 = self.get_t1()
        t2 = self.get_t2()
        with self.assertRaises(ValueError) as e:
            t1.paste_subtree(node=None, other_tree=t2)
        self.assertEqual(e.exception.args[0], 'Use subtree() instead of pasting into an empty tree!')

        # Paste under node
        t1 = self.get_t1()
        t2 = self.get_t2()
        t1.paste_subtree(node='b', other_tree=t2)
        self.assertEqual(t1.tree_id, 't1')
        self.assertEqual(t1.root, 'r')
        self.assertEqual(t1.parent('b').nid, 'r')
        self.assertEqual(set(t1.nodes.keys()), {'r', 'a', 'a1', 'b', 'c', 'd', 'd1', 'r2'})
        self.assertEqual(t1.show(), """root
├── A
│   └── A1
└── B
    └── root2
        ├── C
        └── D
            └── D1
""")
        # Paste empty other_tree (under root)
        t1 = self.get_t1()
        t2 = Tree(tree_id='t2')
        t1.paste_subtree(node='r', other_tree=t2)

        self.assertEqual(t1.tree_id, 't1')
        self.assertEqual(t1.root, 'r')
        self.assertEqual(set(t1.nodes.keys()), {'r', 'a', 'a1', 'b'})
        self.assertEqual(t1.show(), """root
├── A
│   └── A1
└── B
""")

    def test_rsearch(self):
        for nid in ['hárry', 'jane', 'diane']:
            self.assertEqual(nid in self.tree.busearch('diane', lookup_nodes=False), True)

    def test_subtree(self):
        subtree_copy = self.tree.subtree('jane', 'subtree', deep=True)
        self.assertEqual(subtree_copy.parent('jane') is None, True)
        subtree_copy['jane'].tag = 'Sweeti'
        self.assertEqual(self.tree['jane'].tag == 'Jane', True)
        self.assertEqual(subtree_copy.depth('diane'), 1)
        self.assertEqual(subtree_copy.depth('jane'), 0)
        self.assertEqual(self.tree.depth('jane'), 1)

    def test_remove_subtree(self):
        subtree_shallow = self.tree.pop_subtree('jane')
        self.assertEqual('diane' in subtree_shallow.children('jane', lookup_nodes=False), True)
        self.assertEqual('jane' not in self.tree.children('hárry', lookup_nodes=False), True)
        self.tree.paste_subtree('hárry', subtree_shallow)
        self.assertEqual('diane' in self.tree.children('jane', lookup_nodes=False), True)

    def test_remove_subtree_whole_tree(self):
        self.tree.pop_subtree('hárry')
        self.assertIsNone(self.tree.root)
        self.assertEqual(len(self.tree.nodes.keys()), 0)

    def test_siblings(self):
        self.assertEqual(len(list(self.tree.siblings('hárry'))) == 0, True)
        self.assertEqual(list(self.tree.siblings('jane'))[0].nid == 'bill', True)

    def test_tree_data(self):
        class Flower(object):
            def __init__(self, color):
                self.color = color
        self.tree.create_node('Jill', 'jill', parent='jane',
                              data=Flower('white'))
        self.assertEqual(self.tree['jill'].data.color, 'white')
        self.tree.remove_subtree('jill')

    # TODO implement with custom function
    """
    def test_show_data_property(self):
        other_tree = Tree()

        sys.stdout = open(os.devnull, 'w')  # Stops from printing to console

        try:
            other_tree.show()

            class Flower(object):
                def __init__(self, color):
                    self.color = color
            other_tree.create_node('Jill', 'jill', data=Flower('white'))
            other_tree.show(data_property='color')
        finally:
            sys.stdout.close()
            sys.stdout = sys.__stdout__  # Stops from printing to console
    """

    def test_depth(self):
        self.assertEqual(Tree().depth(), 0)
        self.assertEqual(self.tree.depth('hárry'),  0)
        depth = self.tree.depth()
        self.assertEqual(self.tree.depth('diane'),  depth)
        self.assertRaises(NodeIDAbsentError, self.tree.depth, 'diane', lambda x: x.nid != 'jane')

    def test_size(self):
        self.assertEqual(self.tree.size(level=2), 2)
        self.assertEqual(self.tree.size(level=1), 2)
        self.assertEqual(self.tree.size(level=0), 1)

    def test_print_backend(self):
        expected_result = """\
Hárry
├── Bill
│   └── George
└── Jane
    └── Diane
"""

        assert str(self.tree) == expected_result

    def test_show(self):
        sys.stdout = open(os.devnull, 'w')  # Stops from printing to console

        try:
            self.tree.show()
            Tree().show()  # Empty tree!
        finally:
            sys.stdout.close()
            sys.stdout = sys.__stdout__  # Stops from printing to console

    def tearDown(self):
        self.tree = None
        self.copytree = None

    def test_all_nodes(self):
        """
        tests: Tree.all_nodes_iter
        Added by: William Rusnack
        """
        new_tree = Tree()
        self.assertEqual(len(list(new_tree.get_nodes())), 0)
        nodes = list()
        nodes.append(new_tree.create_node('root_node'))
        nodes.append(new_tree.create_node('second', parent=new_tree.root))
        for nd in new_tree.get_nodes():
            self.assertTrue(nd in nodes)

    def test_filter_nodes(self):
        """
        tests: Tree.filter_nodes
        """
        new_tree = Tree(tree_id='tree 1')

        self.assertEqual(tuple(new_tree.get_nodes(lambda n: True)), ())

        nodes = list()
        nodes.append(new_tree.create_node('root_node'))
        nodes.append(new_tree.create_node('second', parent=new_tree.root))

        self.assertEqual(tuple(new_tree.get_nodes(lambda n: False)), ())
        self.assertEqual(tuple(new_tree.get_nodes(lambda n: n.is_root('tree 1'))), (nodes[0],))
        self.assertEqual(tuple(new_tree.get_nodes(lambda n: not n.is_root('tree 1'))), ())
        self.assertTrue(set(new_tree.get_nodes(lambda n: True)), set(nodes))

    @staticmethod
    def test_loop():
        tree = Tree()
        tree.create_node('a', 'a')
        tree.create_node('b', 'b', parent='a')
        tree.create_node('c', 'c', parent='b')
        tree.create_node('d', 'd', parent='c')
        try:
            tree.move_node('b', 'd')
        except LoopError:
            pass

    def test_modify_node_identifier_directly_failed(self):
        tree = Tree()
        tree.create_node('Harry', 'harry')
        tree.create_node('Jane', 'jane', parent='harry')
        n = tree.get_node('jane')
        self.assertTrue(n.nid == 'jane')

        # Failed to modify
        n.nid = 'xyz'
        self.assertTrue(tree.get_node('xyz') is None)
        self.assertTrue(tree.get_node('jane').nid == 'xyz')

    def test_modify_node_identifier_recursively(self):
        tree = Tree()
        tree.create_node('Harry', 'harry')
        tree.create_node('Jane', 'jane', parent='harry')
        n = tree.get_node('jane')
        self.assertTrue(n.nid == 'jane')

        # Success to modify
        tree.update_node(n.nid, nid='xyz')
        self.assertTrue(tree.get_node('jane') is None)
        self.assertTrue(tree.get_node('xyz').nid == 'xyz')

    def test_modify_node_identifier_root(self):
        tree = Tree(tree_id='tree 3')
        tree.create_node('Harry', 'harry')
        tree.create_node('Jane', 'jane', parent='harry')
        tree.update_node(tree['harry'].nid, nid='xyz', tag='XYZ')
        self.assertTrue(tree.root == 'xyz')
        self.assertTrue(tree['xyz'].tag == 'XYZ')
        self.assertEqual(tree.parent('jane').nid, 'xyz')

    def test_subclassing(self):
        class SubNode(Node):
            pass

        class SubTree(Tree):
            node_class = SubNode

        tree = SubTree()
        node = tree.create_node()
        self.assertTrue(isinstance(node, SubNode))

        self.assertTrue(isinstance(node, SubNode))

    def test_shallow_copy_hermetic_pointers(self):
        # tree 1
        # Hárry
        #   └── Jane
        #       └── Diane
        #   └── Bill
        #       └── George
        tree2 = self.tree.subtree(node='jane', tree_id='tree 2')
        # tree 2
        # Jane
        #   └── Diane

        # Check that in shallow copy, instances are the same
        self.assertIs(self.tree['jane'], tree2['jane'])
        self.assertEqual(self.tree['jane']._predecessor, {'tree 1': u'hárry', 'tree 2': None})
        self.assertEqual(dict(self.tree['jane']._successors), {'tree 1': ['diane'], 'tree 2': ['diane']})

        # When creating new node on subtree, check that it has no impact on initial tree
        tree2.create_node('Jill', 'jill', parent='diane')
        self.assertIn('jill', tree2)
        self.assertIn('jill', tree2.children('diane', lookup_nodes=False))
        self.assertNotIn('jill', self.tree)
        self.assertNotIn('jill', self.tree.children('diane', lookup_nodes=False))

    def test_paste_duplicate_nodes(self):
        t1 = Tree()
        t1.create_node(nid='A')
        t2 = Tree()
        t2.create_node(nid='A')
        t2.create_node(nid='B', parent='A')

        with self.assertRaises(DuplicatedNodeIDError) as e:
            t1.paste_subtree('A', t2)
        self.assertEqual(e.exception.args, ('Duplicated nodes [\'A\'] exists!',))

    def test_shallow_paste(self):
        t1 = Tree()
        n1 = t1.create_node(nid='A')

        t2 = Tree()
        n2 = t2.create_node(nid='B')

        t3 = Tree()
        n3 = t3.create_node(nid='C')

        t1.paste_subtree(n1.nid, t2)
        self.assertEqual(t1.to_dict(), {'A': {'children': ['B']}})
        t1.paste_subtree(n1.nid, t3)
        self.assertEqual(t1.to_dict(), {'A': {'children': ['B', 'C']}})

        self.assertEqual(t1.depth(n1.nid), 0)
        self.assertEqual(t1.depth(n2.nid), 1)
        self.assertEqual(t1.depth(n3.nid), 1)

    def test_root_removal(self):
        t = Tree()
        t.create_node(nid='root-A')
        self.assertEqual(len(t.nodes.keys()), 1)
        self.assertEqual(t.root, 'root-A')
        t.remove_subtree(node='root-A')
        self.assertEqual(len(t.nodes.keys()), 0)
        self.assertEqual(t.root, None)
        t.create_node(nid='root-B')
        self.assertEqual(len(t.nodes.keys()), 1)
        self.assertEqual(t.root, 'root-B')
