"""
A pedagogical implementation of a priority queue
"""

from numbers import Number


class PriorityQueue:
    """ Implements a priority queue """

    def __init__(self):
        """
        Initializes the internal attribute  queue to be an empty list.
        """
        self.queue_list = []

    def check(self):
        """
        Check that the internal representation is a list of (key,value) pairs,
        where value is numerical
        """
        is_valid = True
        for pair in self.queue_list:
            if len(pair) != 2:
                is_valid = False
                break
            if not isinstance(pair[1], Number):
                is_valid = False
                break
        return is_valid

    def insert(self, key, cost):
        """
        Add an element to the queue.
        """
        self.queue_list.append((key, cost))

    def min_extract(self):
        """
        Extract the element with minimum cost from the queue.
        """
        lowest = (None, None)
        # key,cost
        for i in self.queue_list:
            if lowest == (None, None):
                lowest = i
            if i[1] < lowest[1]:
                lowest = i
        if lowest[1] is not None:
            self.queue_list.remove(lowest)
        # return key, cost
        return lowest

    def is_member(self, key):
        """
        Check whether an element with a given key is in the queue or not.
        """
        for i in self.queue_list:
            if i[0] == key:
                return True
        return False

    def display_contents(self):
        """ Displays the contents of Priority queue to the command line"""
        for i in self.queue_list:
            print(i)
        return False
