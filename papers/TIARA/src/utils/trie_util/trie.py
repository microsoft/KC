# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy


class TrieNode:
    """A node in the trie structure"""

    def __init__(self, char):
        # the character stored in this node
        self.char = char

        # whether this can be the end of a word
        self.is_end = False

        # a counter indicating how many times a word is inserted
        # (if this node's is_end is True)
        self.counter = 0

        # a dictionary of child nodes
        # keys are characters, values are nodes
        self.children = {}


class Trie(object):
    """The trie object"""

    def __init__(self):
        """
        The trie has at least the root node.
        The root node does not store any character
        """
        self.root = TrieNode('')

    def insert(self, word):
        """Insert a word into the trie"""
        node = self.root

        # Loop through each character in the word
        # Check if there is no child containing the character, create a new child for the current node
        for char in word:
            if char in node.children:
                node = node.children[char]
            else:
                # If a character is not found,
                # create a new node in the trie
                new_node = TrieNode(char)
                node.children[char] = new_node
                node = new_node

        # Mark the end of a word
        node.is_end = True

        # Increment the counter to indicate that we see this word once more
        node.counter += 1

    def dfs(self, node, prefix):
        """Depth-first traversal of the trie

        Args:
            - node: the node to start with
            - prefix: the current prefix, for tracing a word while traversing the trie
        """
        if node.is_end:
            self.output.append((concate(prefix, node.char), node.counter))

        for child in node.children.values():
            self.dfs(child, concate(prefix, node.char))

    def query(self, x):
        """Given an input (a prefix), retrieve all words stored in
        the trie with that prefix, sort the words by the number of
        times they have been inserted
        """
        # Use a variable within the class to keep all possible outputs
        # As there can be more than one word with such prefix
        self.output = []
        node = self.root

        # Check if the prefix is in the trie
        for char in x:
            if char in node.children:
                node = node.children[char]
            else:
                # cannot found the prefix, return empty list
                return []

        # Traverse the trie to get all candidates
        self.dfs(node, x[:-1])

        # Sort the results in reverse order and return
        return sorted(self.output, key=lambda x: x[1], reverse=True)

    def query_child(self, x, get_count=False):
        node = self.root
        for char in x:  # for each token in x
            if char in node.children:  # if this token is in the children
                node = node.children[char]
            else:  # token not in the children, empty result
                if get_count:
                    return [], 0
                return []
        if get_count:
            return list(node.children.keys()), node.counter
        return list(node.children.keys())


def concate(prefix, char):
    if type(prefix) == str:
        return prefix + char
    elif type(prefix) == list:
        res = copy.deepcopy(prefix)
        res.append(char)
        return res


if __name__ == '__main__':
    t = Trie()
    t.insert(['_w', '_a', '_s'])
    t.insert(['_w', '_o', '_r', '_d'])
    t.insert(['_w', '_a', '_r'])
    t.insert(['_w', '_h', '_a', '_t'])
    t.insert(['_w', '_h', '_e', '_r', '_e'])
    print(t.query(['_w', '_h']))
    print(t.query_child(['_w', '_h']))
