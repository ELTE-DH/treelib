import unittest

from collections import defaultdict
from treelib import Node


class NodeCase(unittest.TestCase):
    def setUp(self):
        self.node1 = Node('Test One', 'identifier 1')
        self.node2 = Node('Test Two', 'identifier 2')

    def test_initialization(self):
        self.assertEqual(self.node1.tag, 'Test One')
        self.assertEqual(self.node1.nid, 'identifier 1')

        self.assertEqual(self.node1._predecessor, {})
        self.assertEqual(self.node1._successors, defaultdict(list))
        self.assertEqual(self.node1.data, None)

        a = Node(tag=set())
        b = Node(nid=set())
        c = Node(data=set())

        d = Node()
        self.assertTrue((d._identifier is not None))
        self.assertTrue(isinstance(d._identifier, str))  # UUID1

        e = Node(42)
        self.assertTrue((e._identifier == 42))
        self.assertTrue(e.tag == e._identifier)

        f = Node(tag=42)
        self.assertTrue(e.tag == 42)
        self.assertTrue(e.tag != e._identifier)
        self.assertTrue(e.data is None)

        g = Node(data=dict())
        self.assertTrue(g.data == dict())

        self.assertTrue(self.node1 < self.node2)

    def test_predecessor(self):
        pass  # TODO Itt teszteljük az összes predecessor függvényt (predecessor, set_predecessor, remove_predecessor)


    def test_set_tag(self):
        self.node1.tag = 'Test 1'
        self.assertEqual(self.node1.tag, 'Test 1')
        self.node1.tag = 'Test One'

    def test_object_as_node_tag(self):
        node = Node(tag=(0, 1))
        self.assertEqual(node.tag, (0, 1))
        self.assertTrue(node.__repr__().startswith('Node'))

    def test_set_identifier(self):
        self.node1.nid = 'ID1'
        self.assertEqual(self.node1.nid, 'ID1')
        self.node1.nid = 'identifier 1'
        with self.assertRaises(TypeError):
            self.node1.nid = None

    def test_update_successors(self):
        self.node1.add_successor('identifier 2', 'tree 1')
        self.assertEqual(self.node1.successors('tree 1'), ['identifier 2'])
        self.assertEqual(self.node1._successors['tree 1'], ['identifier 2'])
        self.node1.set_successors([], 'tree 1')
        self.assertEqual(self.node1._successors['tree 1'], [])
        self.assertRaises(NotImplementedError, self.node1.set_successors, Exception, 'tree 1')

    def test_set_predecessor(self):
        self.node2.set_predecessor('identifier 1', 'tree 1')
        self.assertEqual(self.node2.predecessor('tree 1'), 'identifier 1')
        self.assertEqual(self.node2._predecessor['tree 1'], 'identifier 1')
        self.node2.set_predecessor(None, 'tree 1')
        self.assertEqual(self.node2.predecessor('tree 1'), None)
        self.assertRaises(TypeError, self.node2.set_predecessor, {}, 'tree 1')

    def test_set_is_leaf(self):
        self.node1.add_successor('identifier 2', 'tree 2')
        self.node2.set_predecessor('identifier 1', 'tree 1')
        self.assertEqual(self.node1.is_leaf('tree 2'), False)
        self.assertEqual(self.node2.is_leaf('tree 1'), True)

    def test_tree_wise_is_leaf(self):
        self.node1.add_successor('identifier 2', 'tree 1')
        self.node2.set_predecessor('identifier 1', 'tree 1')
        self.assertEqual(self.node1.is_leaf('tree 1'), False)
        self.assertEqual(self.node2.is_leaf('tree 1'), True)

    def test_unhashables(self):
        self.assertRaises(TypeError, self.node1.add_successor, set(), dict())
        with self.assertRaises(TypeError):
            self.node1.nid = None

    def test_predecessor_pop(self):
        self.node1._predecessor.pop('identifier 1', 'tree 1')
        self.node1._predecessor.pop('identifier 2', None)

    def test_successors_pop(self):
        self.node1._successors.pop('identifier 3', 'tree 3')
        self.node1._successors.pop('identifier 4', None)


    def test_data(self):

        class Flower(object):
            def __init__(self, color):
                self.color = color

            def __str__(self):
                return f'{self.color}'

        self.node1.data = Flower('red')
        self.assertEqual(self.node1.data.color, 'red')
