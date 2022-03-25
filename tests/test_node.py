import unittest

from collections import defaultdict
from treelib import Node


class NodeCase(unittest.TestCase):
    def setUp(self):
        self.node1 = Node("Test One", "identifier 1")
        self.node2 = Node("Test Two", "identifier 2")

    def test_initialization(self):
        self.assertEqual(self.node1.tag, "Test One")
        self.assertEqual(self.node1.identifier, "identifier 1")

        self.assertEqual(self.node1._predecessor, {})
        self.assertEqual(self.node1._successors, defaultdict(list))
        self.assertEqual(self.node1.data, None)

    def test_set_tag(self):
        self.node1.tag = "Test 1"
        self.assertEqual(self.node1.tag, "Test 1")
        self.node1.tag = "Test One"

    def test_object_as_node_tag(self):
        node = Node(tag=(0, 1))
        self.assertEqual(node.tag, (0, 1))
        self.assertTrue(node.__repr__().startswith('Node'))

    def test_set_identifier(self):
        self.node1.identifier = "ID1"
        self.assertEqual(self.node1.identifier, "ID1")
        self.node1.identifier = "identifier 1"

    def test_update_successors(self):
        self.node1.update_successors("identifier 2", tree_id="tree 1")
        self.assertEqual(self.node1.successors("tree 1"), ['identifier 2'])
        self.assertEqual(self.node1._successors["tree 1"], ['identifier 2'])
        self.node1.set_successors([], tree_id="tree 1")
        self.assertEqual(self.node1._successors["tree 1"], [])
        self.assertRaises(NotImplementedError, self.node1.set_successors, Exception, tree_id="tree 1")
        self.assertIsNone(self.node1.update_successors(None, tree_id="tree 1"))
        self.assertRaises(NotImplementedError, self.node1.update_successors, "identifier 2", 1000)

    def test_set_predecessor(self):
        self.node2.set_predecessor("identifier 1", "tree 1")
        self.assertEqual(self.node2.predecessor("tree 1"), 'identifier 1')
        self.assertEqual(self.node2._predecessor["tree 1"], 'identifier 1')
        self.node2.set_predecessor(None, "tree 1")
        self.assertEqual(self.node2.predecessor("tree 1"), None)

    def test_set_is_leaf(self):
        self.node1.update_successors("identifier 2")
        self.node2.set_predecessor("identifier 1", "tree 1")
        self.assertEqual(self.node1.is_leaf(), False)
        self.assertEqual(self.node2.is_leaf(), True)

    def test_tree_wise_is_leaf(self):
        self.node1.update_successors("identifier 2", tree_id="tree 1")
        self.node2.set_predecessor("identifier 1", "tree 1")
        self.assertEqual(self.node1.is_leaf("tree 1"), False)
        self.assertEqual(self.node2.is_leaf("tree 1"), True)

    def test_data(self):

        class Flower(object):
            def __init__(self, color):
                self.color = color

            def __str__(self):
                return "%s" % self.color

        self.node1.data = Flower("red")
        self.assertEqual(self.node1.data.color, "red")
