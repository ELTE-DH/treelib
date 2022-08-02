import unittest

from collections import defaultdict
from treelib import Node, Tree
from typing import Any, Union, Hashable, MutableMapping, List


class NodeCase(unittest.TestCase):
    def setUp(self):
        self.node1 = Node('Test One', 'identifier 1')
        self.node2 = Node('Test Two', 'identifier 2')
        self.node3 = Node('Test Three', 'identifier 3', [3, 5, 10])
        self.node4 = Node('Test Four', None , None)
        self.node5 = Node('Test Five', 'identifier 5')

    def test_initialization(self):
        self.assertEqual(self.node1.tag, 'Test One')
        self.assertEqual(self.node1.nid, 'identifier 1')
        self.assertEqual(self.node2.tag, 'Test Two')
        self.assertEqual(self.node2.nid, 'identifier 2')
        self.assertEqual(self.node3.tag, 'Test Three')
        self.assertEqual(self.node3.nid, 'identifier 3')
        self.assertEqual(self.node3.data, [3, 5, 10])

        self.assertFalse(self.node3.nid == self.node4.nid)
        self.assertTrue(self.node4.nid is not None)
        self.assertTrue(self.node4.data is None)

        a = Node(tag=set())
        self.assertTrue((a.tag is not None))
        self.assertTrue((a.tag == set()))
        self.assertFalse(isinstance(a.tag, Hashable))
        self.assertRaises(TypeError, isinstance(a.tag, set))

        b = Node(nid=set())
        self.assertTrue((b._identifier is not None))
        self.assertTrue((b._identifier == set()))
        self.assertFalse(isinstance(b._identifier, Hashable))
        self.assertRaises(TypeError, isinstance(b._identifier, set))

        c = Node(data=set('a'))
        self.assertTrue((c.data is not None))
        self.assertTrue((c.data == set('a')))
        self.assertTrue((c.data is not Any))

        d = Node()
        self.assertTrue((d._identifier is not None))
        self.assertTrue(isinstance(d._identifier, str))  # UUID1

        e = Node(nid=42)
        f = Node(tag=42)

        self.assertTrue((e._identifier == 42))
        self.assertTrue(e.tag == e._identifier)
        self.assertTrue(e._identifier != f._identifier)

        self.assertTrue(f.tag == 42)
        self.assertFalse(f.tag != e._identifier)
        self.assertTrue(f.data is None)

        g = Node(data=dict())
        self.assertTrue(g.data == dict())
        self.assertTrue(g.data is not Hashable)
        self.assertTrue(g._identifier is not None)

        self.assertEqual(self.node1._predecessor, {})
        self.assertTrue(self.node1._predecessor is not Hashable)
        self.assertTrue(self.node1._predecessor == {})
        self.assertEqual(self.node1._successors, defaultdict(list))
        self.assertFalse(self.node1._successors is Hashable)
        self.assertTrue(self.node1._successors == defaultdict(list))

        self.node1.data = ['b', 'c']
        self.assertTrue(self.node1.data is not None)

        self.assertEqual(self.node2.data, None)

        self.assertTrue(self.node1 < self.node2)
        self.assertFalse(self.node3 < self.node4)  # Ezt nem értem

    def test_predecessor(self):
        self.node1._predecessor[1] = 'test predecessor'
        self.node1._predecessor[2] = {'tree 2'}
        self.node1._predecessor[3] = None
        self.assertFalse(self.node1._predecessor[1] is Hashable)
        self.assertFalse(self.node1._predecessor[2] is Hashable)
        self.assertTrue(isinstance(self.node1._predecessor[1], str))
        self.assertEqual(self.node1._predecessor[1], 'test predecessor')
        self.assertIsNone(self.node1._predecessor[3])
        # TODO Itt teszteljük az összes predecessor függvényt (predecessor, set_predecessor, remove_predecessor)

    def test_set_predecessor(self):
        self.node1.set_predecessor('identifier 3', 'tree 3')
        self.assertEqual(self.node1.predecessor('tree 3'), 'identifier 3')
        self.assertEqual(self.node1._predecessor['tree 3'], 'identifier 3')
        self.node1.set_predecessor(None, 'tree 3')
        self.assertEqual(self.node1.predecessor('tree 3'), None)
        self.assertRaises(TypeError, self.node1.set_predecessor, {}, 'tree 3')
        self.node2.set_predecessor('identifier 4', 'tree 4')
        self.assertFalse(self.node2.set_predecessor('identifier 4', 'tree 4') is Hashable)
        self.assertEqual(self.node2.predecessor('tree 4'), 'identifier 4')
        self.assertEqual(self.node2._predecessor['tree 4'], 'identifier 4')
        self.assertTrue(self.node2._predecessor['tree 4'] is not Hashable)
        self.assertRaises(TypeError, self.node1.set_predecessor, [], 'tree 4')

        self.assertRaises(TypeError, self.node1.set_predecessor, 'identifier 5', [])

        self.node3.set_predecessor(tuple, 'tree 6')
        self.assertEqual(self.node3._predecessor['tree 6'], tuple)
        self.assertEqual(self.node3.predecessor('tree 6'), tuple)

        self.node5.set_predecessor('identifier 10', 'tree 10')

    def test_remove_predecessor(self):
        self.node5.set_predecessor('tree 10', 'identifier 10')
        self.node5.set_predecessor('tree 10', 'identifier 6')
        self.node5.set_predecessor('tree 5', 'identifier 9')
        self.node5.remove_predecessor('identifier 6')
        self.node5.remove_predecessor('identifier 9')
        self.assertEqual(self.node5._predecessor['identifier 10'], 'tree 10')
        self.assertRaises(TypeError, self.node5.remove_predecessor, 'identifier 7', 'tree 8')

    def test_successors(self):
        self.node3.successors('identifier 3')
        self.node2.successors('identifier 2')
        self.node1.successors(None)
        self.assertFalse(self.node3.successors('identifier 3') is Hashable)


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
