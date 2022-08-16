import unittest

from collections import defaultdict
from treelib import Node, Tree
from typing import Any, Union, Hashable, MutableMapping, List


class NodeCase(unittest.TestCase):
    def setUp(self):

        self.node1 = Node('Test One', 'identifier 1')
        self.node2 = Node('Test Two', 'identifier 2')
        self.node3 = Node('Test Three', 'identifier 3', [3, 5, 10])
        self.node4 = Node('Test Four')  # None, None TODO "Test nid=None, tag=None"
        self.node5 = Node('Test Five', None, (10, 2, 1))
        self.node6 = Node(None, 'identifier 6', '1,2,3')
        self.node7 = Node(None, None, None)

        # TODO itt kellene preparálni az osztályokat, amiket tesztelsz, ha csak egy mód van rá.

    def test_initialization(self):

        self.assertTrue(self.node3.tag == 'Test Three')
        self.assertTrue(self.node3.nid == 'identifier 3')
        self.assertTrue(self.node3.data == [3, 5, 10])

        self.assertTrue(self.node3.nid != self.node4.nid)

        self.assertTrue(self.node4.tag == 'Test Four')
        self.assertTrue(self.node4.nid is not None)
        self.assertTrue(self.node4.data is None)

        self.assertTrue(self.node5.tag == 'Test Five')
        self.assertTrue(self.node5.tag is not None)
        self.assertTrue(self.node5.data == (10, 2, 1))

        self.assertTrue(self.node6.tag is not None)
        self.assertTrue(self.node6.tag == self.node6.nid)
        self.assertTrue(self.node6.data == '1,2,3')

        self.assertTrue(self.node7.tag is not None)
        self.assertTrue(self.node7.nid is not None)
        self.assertTrue(self.node7.tag == self.node7.nid)
        self.assertTrue(self.node7.data is None)

        a = Node(tag=set())                # TODO ezeket a setup()-ba... -> Nem tudtam a setupba áttenni,
                                           # TODO mert akkor az itt elvégzett tesztek nem futnak le

        self.assertTrue((a.tag is not None))
        self.assertTrue((a.tag == set()))
        self.assertTrue(not isinstance(a.tag, Hashable))
        self.assertRaises(TypeError, isinstance(a.tag, set))

        b = Node(nid=set())
        self.assertTrue((b._identifier is not None))
        self.assertTrue((b._identifier == set()))
        self.assertFalse(isinstance(b._identifier, Hashable))
        self.assertRaises(TypeError, isinstance(b._identifier, set))

        c = Node(data=set('a'))
        self.assertTrue((c.data is not None))
        self.assertTrue((c.data == set('a')))
        self.assertTrue(isinstance(c.data, set))

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

        self.assertTrue(self.node1._predecessor == {})
        self.assertTrue(self.node1._predecessor is not Hashable)

        self.assertTrue(self.node1._successors == defaultdict(list))
        self.assertFalse(self.node1._successors is Hashable)

        self.node1.data = ['b', 'c']
        self.assertTrue(self.node1.data is not None)

        self.assertTrue(self.node2.data == None)

        self.assertTrue(self.node1 < self.node2)

    def test_set_tag(self):  # TODO ezt cifrázni pl nem hashable-vel
        self.node1.tag = 'Test 1'
        self.assertTrue(self.node1.tag == 'Test 1')
        self.node1.nid = []
        self.assertRaises(TypeError, isinstance(self.node1.nid, list))

        self.node2.tag = None
        self.assertTrue(self.node2.tag is not None)
        self.node2.nid = ()
        self.assertRaises(TypeError, isinstance(self.node2.nid, set))

        self.node3.tag = ['Test 3', 'Test Three']
        self.assertTrue(self.node3.tag is not Hashable)
        self.assertRaises(TypeError, isinstance(self.node3.tag, list))
        self.node3.nid = 'Test 3'
        self.assertTrue(self.node3.tag == 'Test 3')

        self.node4.tag = 'Test 4'
        self.node4.nid = None
        self.assertTrue(self.node4.tag == self.node4.nid)
        self.assertTrue(isinstance(self.node4.tag, Hashable))
        self.assertTrue(isinstance(self.node4.tag, str))

        self.node5.tag = None
        self.node5.nid = None
        self.assertTrue(self.node5.tag is not None)
        self.assertTrue(self.node5.nid is not None)
        self.assertTrue(self.node5.tag == self.node5.nid)
        with self.assertRaises(TypeError):
            self.node5.tag = None

    def test_set_identifier(self):
        self.node1.nid = 'ID1'
        self.assertTrue(self.node1.nid == 'ID1')
        self.node1.nid = 'identifier 1'
        with self.assertRaises(TypeError):
            self.node1.nid = None

        self.node2.nid = ['Test 2', 'Test Two']
        self.assertTrue(isinstance(self.node2.nid, list))
        self.assertRaises(TypeError, isinstance(self.node2.nid, list))
        self.assertRaises(TypeError, self.node2.nid is not Hashable)

        self.node3.nid = ['Test 2', 'Test Two']
        self.assertTrue(isinstance(self.node2.nid, list))
        self.assertRaises(TypeError, isinstance(self.node2.nid, list))
        self.assertRaises(TypeError, self.node2.nid is not Hashable)

        self.node4.nid = None
        self.assertTrue(isinstance(self.node4.nid, str))
        with self.assertRaises(TypeError):
            self.node1.nid = None

    def test_object_as_node_tag(self):
        node = Node(tag=(0, 1))
        self.assertTrue(node.tag == (0, 1))
        self.assertTrue(node.__repr__().startswith('Node'))

        node = Node(tag='Test 10')
        self.assertTrue(node.tag == 'Test 10')
        self.assertTrue(node.__repr__().startswith('Node'))

    def test_predecessor(self):
        self.node1._predecessor[1] = 'test predecessor'
        self.node1._predecessor[2] = {'tree 2'}
        self.node1._predecessor[3] = None
        self.assertTrue(isinstance(self.node1._predecessor[1], Hashable))
        self.assertFalse(isinstance(self.node1._predecessor[2], Hashable))
        with self.assertRaises(ValueError):
            self.node1._predecessor[3] = None  # TODO ennek itt működnie kellene, nem lehetne None
        self.assertEqual(self.node1._predecessor[1], 'test predecessor')

        self.node2._predecessor = {1: 'test one'}
        self.assertTrue(self.node2._predecessor is not Hashable)
        self.node2._predecessor[2] = 'add test two'
        self.assertTrue(isinstance(self.node2._predecessor[2], Hashable))
        self.node2._predecessor = self.node1
        self.assertTrue(self.node1._successors == self.node2)
        self.assertTrue(self.node1._successors is not None)

        self.node3._predecessor = None
        with self.assertRaises(ValueError):
            self.node3._predecessor = None

        # TODO Itt teszteljük az összes predecessor függvényt (predecessor, set_predecessor, remove_predecessor)

    def test_set_predecessor(self):
        self.node1.set_predecessor('identifier 3', 'tree 3')
        self.assertTrue(self.node1.predecessor('tree 3') == 'identifier 3')
        self.assertTrue(self.node1._predecessor['tree 3'] == 'identifier 3')

        self.node1.set_predecessor(None, 'tree 3')
        self.assertTrue(self.node1.predecessor('tree 3') == None)

        self.assertRaises(TypeError, self.node1.set_predecessor, {}, 'tree 3')

        self.node2.set_predecessor('identifier 4', 'tree 4')
        self.assertTrue(isinstance(self.node2.set_predecessor('identifier 4', 'tree 4'), Hashable))
        self.assertTrue(self.node2.predecessor('tree 4') == 'identifier 4')
        self.assertTrue(self.node2._predecessor['tree 4'] == 'identifier 4')

        self.assertRaises(TypeError, self.node1.set_predecessor, [], 'tree 4')
        self.assertRaises(TypeError, self.node1.set_predecessor, 'identifier 5', [])

        self.node3.set_predecessor(tuple, 'tree 6')
        self.assertTrue(self.node3._predecessor['tree 6'] == tuple)
        self.assertTrue(self.node3.predecessor('tree 6') == tuple)

        self.node5.set_predecessor(self.node3, 'tree 223')

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

    def test_set_successors(self):
        self.node4.set_successors([1, 2, 3], 'identifier 4')
        self.assertTrue(isinstance(self.node4.successors('identifier 4'), list))
        self.assertTrue(self.node4.successors('identifier 4') is not Hashable)
        self.assertRaises(NotImplementedError, self.node5.set_successors, {}, 'identifier 5')
        self.assertRaises(NotImplementedError, self.node5.set_successors, '', 'identifier 5')
        self.assertRaises(NotImplementedError, self.node5.set_successors, tuple, 'identifier 5')
        self.assertRaises(NotImplementedError, self.node5.set_successors, (), 'identifier 5')
        self.assertRaises(NotImplementedError, self.node1.set_successors, Exception, 'tree 1')

    def test_remove_successors(self):
        self.node4.set_successors([1, 2, 3], 'identifier 4')
        self.node4.set_successors([10, 20], 'identifier 6')
        self.node4.remove_successors('identifier 4')
        self.assertTrue(isinstance(self.node4.successors('identifier 6'), list))
        self.assertTrue(self.node4.successors('identifier 6') is not None)
        self.assertRaises(KeyError, self.node4.remove_successors, 'identifier 8')

    def test_add_successor(self):
        self.node4.set_successors([1, 2, 3], 'identifier 4')
        self.node4.add_successor('identifier 2', 'tree 2')
        self.node4.add_successor(int, 'tree 3')
        self.node4.add_successor(tuple, 'tree 7')
        self.assertTrue(isinstance(self.node4.successors('identifier 2'), list))
        self.assertTrue(isinstance(self.node4.successors(int), list))
        self.assertTrue(isinstance(self.node4.successors(tuple), list))
        self.assertRaises(TypeError, self.node4.add_successor, [], 'identifier 5')
        self.node1.add_successor('identifier 2', 'tree 1')
        self.assertEqual(self.node1.successors('tree 1'), ['identifier 2'])
        self.assertEqual(self.node1._successors['tree 1'], ['identifier 2'])
        self.node1.set_successors([], 'tree 1')
        self.assertEqual(self.node1._successors['tree 1'], [])

    def test_remove_successor(self):
        self.node4.set_successors([1, 2, 3], 'identifier 4')
        self.node4.add_successor('identifier 2', 'tree 2')
        self.node4.add_successor(int, 'tree 3')
        self.node4.add_successor(tuple, 'tree 7')
        self.node4.remove_successor('identifier 2', 'tree 2')
        self.assertEqual(self.node2.is_leaf('tree 4'), True)
        self.assertRaises(ValueError, self.node4.remove_successor, 'identifier 8', 'tree 8')

    def test_replace_successor(self):
        self.node4.set_successors([1, 2, 3], 'identifier 4')
        self.node4.set_successors([10, 20], 'identifier 6')
        self.node4.replace_successor(1, 'identifier 4', 'identifier 6')
        self.assertTrue(self.node4.successors('identifier 6'))
        self.assertRaises(ValueError, self.node4.replace_successor, 30, 'identifier 4', 'identifier 6')

    def test_unhashables(self):
        self.assertRaises(TypeError, self.node1.add_successor, set(), dict())
        with self.assertRaises(TypeError):
            self.node1.nid = None

    def test_clone_pointers(self):
        self.node1._predecessor[1] = 'test predecessor'
        self.node2._predecessor[2] = 'test predecessor 2'
        self.node1.clone_pointers(1, 2, True)  # TODO Ezt még ki kellene taláni, hogy hogyan érdemes tesztelni

    def test_delete_pointers(self):
        self.node1._predecessor[1] = 'test predecessor'
        self.node2._predecessor[2] = 'test predecessor 2'
        self.node1.delete_pointers(1)
        self.node2.delete_pointers(2)
        self.assertEqual(self.node1._predecessor, {})
        self.assertEqual(self.node2._predecessor, {})

    def test_set_is_leaf(self):
        self.node1.add_successor('identifier 2', 'tree 2')
        self.node2.set_predecessor('identifier 1', 'tree 1')
        self.node3 = Node('Test Three', 'identifier 3', [3, 5, 10])
        self.assertEqual(self.node1.is_leaf('tree 2'), False)
        self.assertEqual(self.node2.is_leaf('tree 1'), True)
        self.assertEqual(self.node3.is_leaf('tree 1'), True)  # ??

    def test_is_root(self):
        pass

    def test_is_multiple_trees(self):
        self.node1.add_successor('identifier 2', 'tree 2')
        self.node1.set_predecessor('identifier 2454', 'tree 45453')
        # TODO ekkor true
        self.node1.remove_predecessor('tree 45453')
        # TODO ekkor meg false nak kellene lennie
        pass

    def test_tree_wise_is_leaf(self):
        self.node1.add_successor('identifier 2', 'tree 1')
        self.node2.set_predecessor('identifier 1', 'tree 1')
        self.assertEqual(self.node1.is_leaf('tree 1'), False)
        self.assertEqual(self.node2.is_leaf('tree 1'), True)

    def test_data(self):

        class Flower(object):
            def __init__(self, color):
                self.color = color

            def __str__(self):
                return f'{self.color}'

        self.node1.data = Flower('red')
        self.assertEqual(self.node1.data.color, 'red')
