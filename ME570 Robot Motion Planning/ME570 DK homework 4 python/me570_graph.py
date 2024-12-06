"""
Classes and utility functions for working with graphs (plotting, search, initialization, etc.)
"""
import math
import pickle
from math import pi

import numpy as np
from matplotlib import pyplot as plt

import me570_geometry
import me570_potential
import me570_queue


def plot_arrows_from_list(arrow_list, scale=1.0, color=(0., 0., 0.)):
    """
    Plot arrows from a list of pairs of base points and displacements
    """
    x_edges, v_edges = [np.hstack(x) for x in zip(*arrow_list)]
    plt.quiver(x_edges[0, :],
               x_edges[1, :],
               v_edges[0, :],
               v_edges[1, :],
               angles='xy',
               scale_units='xy',
               scale=scale,
               color=color)


def plot_text(coord, str_label, color=(1., 1., 1.)):
    """
    Wrap plt.text to get a consistent look
    """
    plt.text(coord[0].item(),
             coord[1].item(),
             str_label,
             ha="center",
             va="center",
             fontsize='xx-small',
             bbox={"boxstyle": 'round', "fc": color, "ec": None})
    # bbox=dict(boxstyle="round", fc=color, ec=None))


class Graph:
    """
    A class collecting a graph_vector data structure and all the functions that operate on a graph.
    """

    def __init__(self, graph_vector):
        """
        Stores the arguments as internal attributes.
        """
        self.graph_vector = graph_vector

    def _apply_neighbor_function(self, func):
        """
        Apply a function on each node and chain the result
        """
        list_of_lists = [func(n) for n in self.graph_vector]
        return [e for l in list_of_lists for e in l]

    def _neighbor_weights_with_positions(self, n_current):
        """
        Get all weights and where to display them
        """
        x_current = n_current['x']
        return [
            (weight_neighbor,
             self.graph_vector[idx_neighbor]['x'] * 0.25 + x_current * 0.75)
            for (weight_neighbor, idx_neighbor
                 ) in zip(n_current['neighbors_cost'], n_current['neighbors'])
        ]

    def _neighbor_displacements(self, n_current):
        """
        Get all displacements with respect to the neighbors for a given node
        """
        x_current = n_current['x']
        return [(x_current, self.graph_vector[idx_neighbor]['x'] - x_current)
                for idx_neighbor in n_current['neighbors']]

    def _neighbor_backpointers(self, n_current):
        """
        Get coordinates for backpointer arrows
        """
        x_current = n_current['x']
        idx_backpointer = n_current.get('backpointer', None)
        if idx_backpointer is not None:
            arrow = [
                (x_current,
                 0.5 * (self.graph_vector[idx_backpointer]['x'] - x_current))
            ]
        else:
            arrow = []
        return arrow

    def _neighbor_backpointers_cost(self, n_current):
        """
        Get value and coordinates for backpointer costs
        """
        x_current = n_current['x']
        idx_backpointer = n_current.get('backpointer', None)
        if idx_backpointer is not None:
            arrow = [(n_current['g'],
                      self.graph_vector[idx_backpointer]['x'] * 0.25 +
                      x_current * 0.75)]
        else:
            arrow = []
        return arrow

    def has_backpointers(self):
        """
        Return True if self.graph_vector has a "backpointer" field
        """
        is_empty = len(self.graph_vector) == 0
        is_all_backpointers_none = all(
            [x.get('backpointer', None) is None for x in self.graph_vector])
        return self.graph_vector is not None and not is_empty and not is_all_backpointers_none

    def plot(self,
             flag_edges=True,
             flag_labels=False,
             flag_edge_weights=False,
             flag_backpointers=True,
             flag_backpointers_cost=True,
             flag_heuristic=False,
             node_lists=None,
             idx_closed=None,
             idx_goal=None,
             idx_best=None):
        """
        The function plots the contents of the graph described by the  graph_vector structure,
        alongside other related, optional data.
        """

        if flag_edges:
            displacement_list = self._apply_neighbor_function(
                self._neighbor_displacements)
            plot_arrows_from_list(displacement_list, scale=1.05)

        if flag_labels:
            for idx, n_current in enumerate(self.graph_vector):
                x_current = n_current['x']
                plot_text(x_current, str(idx))

        if idx_closed is not None:
            for idx in idx_closed:
                x_current = self.graph_vector[idx]['x']
                plt.scatter(x_current[0],
                            x_current[1],
                            marker='s',
                            color=(0., 0., 1.))

        if idx_goal is not None:
            x_goal = self.graph_vector[idx_goal]['x']
            plt.plot(x_goal[0, :],
                     x_goal[1, :],
                     marker='d',
                     markersize=10,
                     color=(.8, .1, .1))

        if idx_best is not None:
            x_best = self.graph_vector[idx_best]['x']
            plt.plot(x_best[0, :],
                     x_best[1, :],
                     marker='d',
                     markersize=10,
                     color=(0., 1., 0.))

        if flag_heuristic and idx_goal is not None:
            for idx, n_current in enumerate(self.graph_vector):
                x_current = n_current['x']
                h_current = self.heuristic(idx, idx_goal)
                plot_text(x_current, f'h={h_current:.2f}', color=(.8, 1., .8))
                if flag_heuristic and idx_goal is not None:
                    idx_backpointer = n_current.get('backpointer', None)
                    if idx_backpointer is not None:
                        cost = n_current['g'] + h_current
                        offset = np.array([[0], [.15]])
                        plot_text(x_current + offset,
                                  f'f={cost:.2f}',
                                  color=(.8, 1., .8))

        if flag_edge_weights:
            weight_list = self._apply_neighbor_function(
                self._neighbor_weights_with_positions)
            for (weight, coord) in weight_list:
                plot_text(coord, str(weight), color=(.8, .8, 1.))

        if flag_backpointers and self.has_backpointers():
            backpointer_arrow_list = self._apply_neighbor_function(
                self._neighbor_backpointers)
            plot_arrows_from_list(backpointer_arrow_list,
                                  scale=1.05,
                                  color=(0.1, .8, 0.1))

        if flag_backpointers_cost and self.has_backpointers:
            backpointer_cost_list = self._apply_neighbor_function(
                self._neighbor_backpointers_cost)
            offset = np.array([[0], [-.15]])
            for (cost, coord) in backpointer_cost_list:
                plot_text(coord + offset, f'g={cost:.2f}', color=(.8, 1., .8))

        if node_lists is not None:
            if not isinstance(node_lists[0], list):
                node_lists = [node_lists]
            markers = ['d', 'o', 's', '*', 'h', '^', '8']
            for i, lst in enumerate(node_lists):
                x_list = [self.graph_vector[e]['x'] for e in lst]
                coords = np.hstack(x_list)
                plt.plot(
                    coords[0, :],
                    coords[1, :],
                    markers[i % len(markers)],
                    markersize=10,
                )

    def nearest_neighbors(self, x_query, k_nearest):
        """
        Returns the k nearest neighbors in the graph for a given point.
        """
        x_query = np.reshape(x_query,(2,1))
        x_graph = np.hstack([n['x'] for n in self.graph_vector])
        distances_squared = np.sum((x_graph - x_query) ** 2, 0)
        idx = np.argpartition(distances_squared, k_nearest)
        return idx[:k_nearest]

    def heuristic(self, idx_x, idx_goal):
        """
        Computes the heuristic  h given by the Euclidean distance between the nodes with indexes
        idx_x and  idx_goal.
        """
        idx_x_x = self.graph_vector[idx_x]['x'][0]
        idx_x_y = self.graph_vector[idx_x]['x'][1]
        idx_goal_x = self.graph_vector[idx_goal]['x'][0]
        idx_goal_y = self.graph_vector[idx_goal]['x'][1]
        if not isinstance(idx_x_x, int):
            if not isinstance(idx_x_x, float):
                idx_x_x = idx_x_x[0]
        if not isinstance(idx_x_y, int):
            if not isinstance(idx_x_y, float):
                idx_x_y = idx_x_y[0]
        if not isinstance(idx_goal_x, int):
            if not isinstance(idx_goal_x, float):
                idx_goal_x = idx_goal_x[0]
        if not isinstance(idx_goal_y, int):
            if not isinstance(idx_goal_y, float):
                idx_goal_y = idx_goal_y[0]

        h_val = math.sqrt((idx_goal_x - idx_x_x) ** 2 + (idx_goal_y - idx_x_y) ** 2)
        # print("idx_x,", self.graph_vector[idx_x]['x'])
        # print("idx_goal,", self.graph_vector[idx_goal]['x'])
        # print("heura1,", self.graph_vector[idx_goal]['x'][0])
        # print("heura2,", self.graph_vector[idx_x]['x'][0])
        # print("heurb1,", self.graph_vector[idx_goal]['x'][1])
        # print("heurb2,", self.graph_vector[idx_x]['x'][1])
        # print("heura1 post,", idx_x_x)
        # print("heura2 post,", idx_x_y)
        # print("heurb1 post,", idx_goal_x)
        # print("heurb2 post,", idx_goal_y)

        return h_val

    def get_expand_list(self, idx_n_best, idx_closed):
        """
        Finds the neighbors of element  idx_n_best that are not in  idx_closed (line 12
           in Algorithm~).
        """
        idx_expand = list(self.graph_vector[idx_n_best]['neighbors'])
        idx_neighbors = np.array(self.graph_vector[idx_n_best]['neighbors'])
        # print(self.graph_vector[idx_n_best])
        # print(self.graph_vector[idx_n_best]['neighbors'])
        # print(idx_expand)
        # print(idx_neighbors)
        for i in range(0, idx_neighbors.size):
            if isinstance(idx_closed, int):
                if isinstance(idx_closed, float):
                    if idx_neighbors[i] == idx_closed:
                        idx_expand.remove(idx_neighbors[i])
            elif idx_neighbors[i] in idx_closed:
                idx_expand.remove(idx_neighbors[i])
        # print(idx_expand)
        return idx_expand

    def expand_element(self, idx_n_best, idx_x, idx_goal, pq_open):
        """
        This function expands the vertex with index  idx_x (which is a neighbor of the one with
        index  idx_n_best) and returns the updated versions of  graph_vector and  pq_open.
        """
        """
        i_neighbors_idx = np.array(self.graph_vector[idx_x]['neighbors'])
        i_neighbors_cost = np.array(self.graph_vector[idx_x]['neighbors_cost'])
        for i in range(0,i_neighbors_idx[0].size):
            if i_neighbors_idx[i] != idx_n_best: #remove ind_n_best from consideration
                # can use remove from OG larger array for optimization
                if pq_open.is_member(i_neighbors_idx[i]) is False:
                    pq_open.insert(i_neighbors_idx[i], self.graph_vector[idx_x]['cost'] +
                    i_neighbors_cost[i] + self.graph_vector.heuristic(i_neighbors_idx[i],idx_goal))
                    """

        # possibly put into another section for complete A*????
        """if not pq_open.is_member(idx_x):
            print("idx_n_best,", idx_n_best)
            print("idx_x, ", idx_x)
            self.graph_vector[idx_x]['backpointer'] = idx_n_best
            best_neighbors = np.array(self.graph_vector[idx_n_best]['neighbors'])
            best_neighbors_cost = np.array(self.graph_vector[idx_n_best]['neighbors_cost'])
            costa = best_neighbors_cost[np.where(best_neighbors == idx_x)[0]]
            costb = self.heuristic(idx_x, idx_goal)
            print("costa solo, ", costa)
            print("costb solo, ", costb)
            if not isinstance(costa, int):
                if not isinstance(costa, float):
                    costa = costa[0]
            if not isinstance(costb, int):
                if not isinstance(costb, float):
                    costb = costb[0]
            print("costa solo post, ", costa)
            print("costb solo post, ", costb)
            pq_open.insert(idx_x, costa + costb)
            self.graph_vector[idx_x]['g'] = 0
            if self.graph_vector[idx_n_best]['backpointer'] is None:
                self.graph_vector[idx_x]['g'] = 0
            else:
                self.graph_vector[idx_x]['g'] += self.graph_vector[idx_n_best]['g']
            self.graph_vector[idx_x]['g'] += best_neighbors_cost[np.where(best_neighbors == idx_x)[0]]
            print(pq_open.queue_list)
        else:
            #print("self.graph_vector[idx_x]['backpointer'],", self.graph_vector[idx_x]['backpointer'])
            print("idx_n_best,", idx_n_best)
            print("idx_x, ", idx_x)
            print("idx_goal, ", idx_goal)
            print(pq_open.queue_list)
            i_neighbors_idx = np.array(self.graph_vector[idx_x]['neighbors'])
            i_neighbors_cost = np.array(self.graph_vector[idx_x]['neighbors_cost'])
            nearest_neighbors_idx = np.array(self.get_expand_list(idx_x, idx_n_best))
            print("nearest_neighbors_idx ", nearest_neighbors_idx )
            print("i_neighbors_idx, ", i_neighbors_idx)
            print("i_neighbors_cost, ", i_neighbors_cost)
            #np.delete(i_neighbors_idx,idx_x)
            for i in range(0,nearest_neighbors_idx.size):
                #if not self.pq_closed.is_member(idx_x):
                #HOW CAN YOU NOT BE USING MY GRAPH STRUCT
                if not pq_open.is_member(nearest_neighbors_idx[i]):
                    #will need to update to include if member for updating G for V2
                    if self.graph_vector[nearest_neighbors_idx[i]]['backpointer'] is None:
                        #pq_open.insert(i_neighbors_idx[i], self.graph_vector[idx_x]['cost'] +
                        #               self.graph_vector[idx_x]['neighbors_cost'] +
                        #               self.graph_vector.heuristic(i_neighbors_idx[i], idx_goal))
                        #cost = i_neighbors_cost[i] + self.heuristic(nearest_neighbors_idx[i],idx_goal)
                        #print(cost)
                        print("nearest_neighbors_idx[i], ", nearest_neighbors_idx[i])
                        costa = i_neighbors_cost[np.where(i_neighbors_idx == nearest_neighbors_idx[i])[0]]
                        print("costa, ", costa)
                        costb = self.heuristic(nearest_neighbors_idx[i],idx_goal)
                        print("costb, ", costb)
                        if not isinstance(costa, int):
                            if not isinstance(costa, float):
                                costa = costa[0]
                        if not isinstance(costb, int):
                            if not isinstance(costb, float):
                                costb = costb[0]
                        self.graph_vector[nearest_neighbors_idx[i]]['backpointer'] = idx_x
                        self.graph_vector[nearest_neighbors_idx[i]]['g'] += costa + self.graph_vector[idx_x]['g']
                        print("costa post, ", costa)
                        print("costb post, ", costb)
                        cost = (self.graph_vector[nearest_neighbors_idx[i]]['g'] + costb)
                        print("cost, ", cost)
                        pq_open.insert(nearest_neighbors_idx[i], cost + 1)
                        print(pq_open.queue_list)
                    else:
                        costa = i_neighbors_cost[np.where(i_neighbors_idx == nearest_neighbors_idx[i])[0]]
                        if not isinstance(costa, int):
                            if not isinstance(costa, float):
                                costa = costa[0]
                        if (self.graph_vector[idx_x]['g'] + costa) < self.graph_vector[nearest_neighbors_idx[i]]['g']:
                            self.graph_vector[nearest_neighbors_idx[i]]['g'] = self.graph_vector[idx_x]['g'] + costa
                            #update for V2
                            #pq_open.remove(nearest_neighbors_idx[i])
                            #pq_open.insert(nearest_neighbors_idx[i], costa + )
                            self.graph_vector[nearest_neighbors_idx[i]]['backpointer'] = idx_x

            #update graph_vector nodes
            best_neighbors = np.array(self.graph_vector[idx_n_best]['neighbors'])
            best_neighbors_cost = np.array(self.graph_vector[idx_n_best]['neighbors_cost'])
            self.graph_vector[idx_x]['g'] = best_neighbors_cost[np.where(best_neighbors == idx_x)[0]] + self.graph_vector[idx_n_best]['g']

            print(pq_open.queue_list)"""
        # if not self.pq_closed.is_member(idx_x):
        if not pq_open.is_member(idx_x):
            # print("idx_n_best,", idx_n_best)
            # print("idx_x, ", idx_x)
            self.graph_vector[idx_x]['backpointer'] = idx_n_best
            best_neighbors = np.array(self.graph_vector[idx_n_best]['neighbors'])
            best_neighbors_cost = np.array(self.graph_vector[idx_n_best]['neighbors_cost'])
            costa = best_neighbors_cost[np.where(best_neighbors == idx_x)[0]]
            costb = self.heuristic(idx_x, idx_goal)
            # print("costa solo, ", costa)
            # print("costb solo, ", costb)
            if not isinstance(costa, int):
                if not isinstance(costa, float):
                    costa = costa[0]
            if not isinstance(costb, int):
                if not isinstance(costb, float):
                    costb = costb[0]
            # print("costa solo post, ", costa)
            # print("costb solo post, ", costb)
            self.graph_vector[idx_x]['g'] = 0
            if self.graph_vector[idx_n_best]['backpointer'] is None:
                self.graph_vector[idx_x]['g'] = 0
            else:
                self.graph_vector[idx_x]['g'] += self.graph_vector[idx_n_best]['g']
            self.graph_vector[idx_x]['g'] += best_neighbors_cost[
                np.where(best_neighbors == idx_x)[0]]
            pq_open.insert(idx_x, costa + costb + self.graph_vector[idx_n_best]['g'])
            # print(pq_open.queue_list)
        return pq_open

    def path(self, idx_start, idx_goal):
        """
        This function follows the backpointers from the node with index  idx_goal in  graph_vector
        to the one with index  idx_start node, and returns the  coordinates (not indexes) of the
        sequence of traversed elements.
        """
        print("idx_start, ", idx_start)
        print("idx_goal, ", idx_goal)
        idx_backpoint = self.graph_vector[idx_goal]['backpointer']
        x_path = np.array(self.graph_vector[idx_goal]['x'])
        idx_last = None
        if idx_start == idx_goal:
            return x_path
        # print("idx_backpoint, ", idx_backpoint)
        # print("x_path, ", x_path)
        while idx_backpoint is not None:
            # print("self.graph_vector[idx_backpoint]['x'][0], ", self.graph_vector[idx_backpoint]['x'][0])
            # print("x_path[0], ", x_path[0])
            # print(np.concatenate((self.graph_vector[idx_backpoint]['x'][0], x_path[0])))
            newx = np.concatenate((self.graph_vector[idx_backpoint]['x'][0], x_path[0]))
            newy = np.concatenate((self.graph_vector[idx_backpoint]['x'][1], x_path[1]))
            x_path = np.array([newx, newy])
            # print("x_path[0] post, ", x_path[0])
            idx_last = idx_backpoint
            idx_backpoint = self.graph_vector[idx_backpoint]['backpointer']
            print("idx_backpoint_l, ", idx_backpoint)
            print("x_path_l, ", x_path)
        # code 1.1 add check to see if path from goal to start exists
        if idx_start == idx_last:
            return x_path
        else:
            return np.array([[], []])

    def search(self, idx_start, idx_goal, method='astar', plot=False):
        """
        Implements the  A^* algorithm, as described by the pseudo-code in Algorithm~ .
        """
        self.graph_vector[idx_start]['backpointer'] = None
        self.graph_vector[idx_start]['g'] = 0
        idx_closed = [-1]
        pq_open = me570_queue.PriorityQueue()
        pq_open.insert(idx_start, 0)
        while len(pq_open.queue_list) != 0:
            pq_pop = pq_open.min_extract()
            idx_best = pq_pop[0]
            idx_closed.append(idx_best)
            if idx_best == idx_goal:
                break
            print("self.graph_vector[idx_best]['neighbors'], ",
                  self.graph_vector[idx_best]['neighbors'])
            # print("np.array(self.graph_vector[idx_best]['neighbors']), ",
            #       np.array(self.graph_vector[idx_best]['neighbors']))
            neighbors_idx = np.array(self.graph_vector[idx_best]['neighbors'])
            neighbors_cost = np.array(self.graph_vector[idx_best]['neighbors_cost'])
            for i_neighbors_idx in range(0, neighbors_idx.size):
                idx_x = neighbors_idx[i_neighbors_idx]
                if idx_x not in idx_closed:
                    if not pq_open.is_member(idx_x):
                        self.graph_vector[idx_x]['g'] = (self.graph_vector[idx_best]['g'] +
                                                         neighbors_cost[i_neighbors_idx])
                        self.graph_vector[idx_x]['backpointer'] = idx_best
                        costh = self.heuristic(idx_x, idx_goal)
                        if method.lower() == 'bfs'.lower():
                            pq_open.insert(idx_x, self.graph_vector[idx_x]['g'])
                        elif method.lower() == 'greedy'.lower():
                            pq_open.insert(idx_x, costh)
                        elif method.lower() == 'astar'.lower():
                            pq_open.insert(idx_x, self.graph_vector[idx_x]['g'] + costh)
                    else:
                        costa = neighbors_cost[i_neighbors_idx]
                        print("costa, ", costa)
                        # if not isinstance(costa, int):
                        #    if not isinstance(costa, float):
                        #        costa = costa[0]
                        if (self.graph_vector[idx_best]['g'] + costa) < self.graph_vector[idx_x][
                            'g']:
                            self.graph_vector[idx_x]['g'] = (self.graph_vector[idx_best]['g']
                                                             + costa)
                            # update for V2
                            costh = self.heuristic(idx_x, idx_goal)
                            print(idx_x)
                            #print(pq_open.queue_list)
                            pq_open.remove(idx_x)
                            if method.lower() == 'bfs'.lower():
                                pq_open.insert(idx_x, self.graph_vector[idx_x]['g'])
                            elif method.lower() == 'greedy'.lower():
                                pq_open.insert(idx_x, costh)
                            elif method.lower() == 'astar'.lower():
                                pq_open.insert(idx_x, self.graph_vector[idx_x]['g'] + costh)
                            self.graph_vector[idx_x]['backpointer'] = idx_best
                    if plot:
                        fig = plt.figure()
                        self.plot(idx_goal=idx_goal)
                        fig.show()
            if plot:
                fig = plt.figure()
                self.plot(idx_goal=idx_goal)
                fig.show()
        x_path = self.path(idx_start, idx_goal)
        print("x_path_search, ", x_path)
        return x_path

    def search_start_goal(self, x_start, x_goal):
        """
        This function performs the following operations:
         - Identifies the two indexes  idx_start,  idx_goal in  graph.graph_vector that are closest
        to  x_start and  x_goal (using Graph.nearestNeighbors twice, see Question~ -nearest).
         - Calls Graph.search to find a feasible sequence of points  x_path from  idx_start to
        idx_goal.
         - Appends  x_start and  x_goal, respectively, to the beginning and the end of the array
        x_path.
        """
        print("x_start, x_goal, ", x_start, " ", x_goal)
        start_neighbor = self.nearest_neighbors(x_start, 1)
        goal_neighbor = self.nearest_neighbors(x_goal, 1)
        start_neighbor = start_neighbor[0]
        goal_neighbor = goal_neighbor[0]
        print("start_neighbor, goal_neighbor, ", start_neighbor, " ", goal_neighbor)
        x_path = self.search(start_neighbor, goal_neighbor)

        if x_start[0] != x_path[0][0] and x_start[1] != x_path[1][0]:
            print("x_path, ", x_path)
            newx = np.concatenate((np.reshape(x_start, (2, 1))[0], x_path[0]))
            newy = np.concatenate((np.reshape(x_start, (2, 1))[1], x_path[1]))
            x_path = np.array([newx, newy])
        if x_goal[0] != x_path[0][-1] and x_goal[1] != x_path[1][-1]:
            print("x_path, ", x_path)
            newx = np.concatenate((x_path[0], np.reshape(x_goal, (2, 1))[0]))
            newy = np.concatenate((x_path[1], np.reshape(x_goal, (2, 1))[1]))
            x_path = np.array([newx, newy])



        """if not isinstance(x_start, int):
            print("x_path, ", x_path)
            newx = np.concatenate(x_start[0], x_path[0])
            newy = np.concatenate(x_start[1], x_path[1])
            x_path = np.array([newx, newy])
        if not isinstance(x_goal, int):
            print("x_path, ", x_path)
            newx = np.concatenate(x_path[0], x_goal[0])
            newy = np.concatenate(x_path[1], x_goal[1])
            x_path = np.array([newx, newy])"""

        # print("x_path, ", x_path)
        # newx = np.concatenate((self.graph_vector[start_neighbor]['x'][0],
        # x_path[0], self.graph_vector[goal_neighbor]['x'][0]))
        # newy = np.concatenate((self.graph_vector[start_neighbor]['x'][1],
        # x_path[1], self.graph_vector[goal_neighbor]['x'][1]))
        # x_path = np.array([newx, newy])
        # print("new x_path, ", x_path)
        return x_path


class SphereWorldGraph:
    """
    A discretized version of the  SphereWorld from Homework 3 with the addition of a search
function.
    """

    def __init__(self, nb_cells):
        """
        The function performs the following steps:
         - Instantiate an object of the class  SphereWorld from Homework 3 to load the contents of
        the file sphereworld.mat. Store the object as the internal attribute  sphereworld.d
         - Initializes an object  grid from the class  Grid initialized with arrays  xx_grid and
        yy_grid, each one containing  nb_cells values linearly spaced values from  -10 to  10.
         - Use the method grid.eval to obtain a matrix in the format expected by grid2graph in
        Question~ , i.e., with a  true if the space is free, and a  false if the space is occupied
        by a sphere at the corresponding coordinates. The quickest way to achieve this is to
        manipulate the output of Total.eval (for checking collisions with the spheres) while using
        it in conjunction with grid.eval (to evaluate the collisions along all the points on the
        grid); note that the choice of the attractive potential here does not matter.
         - Call grid2graph.
         - Store the resulting  graph object as an internal attribute.
        """
        self.sphereworld = me570_potential.SphereWorld()
        #plt.figure()
        #self.sphereworld.plot()
        # plt.show()

        xx_grid = np.linspace(-10, 10, nb_cells)
        yy_grid = np.linspace(-10, 10, nb_cells)
        self.potentialQ = {
            'shape': 'quadratic',
            'repulsive_weight': .001,
            'x_goal': np.array([[self.sphereworld.x_goal[0, 0]], [self.sphereworld.x_goal[1, 0]]])
        }
        self.totalQ = me570_potential.Total(self.sphereworld, self.potentialQ)

        # testobj = lambda x: (self.totalQ.eval > 0)
        self.grid = me570_geometry.Grid(xx_grid, yy_grid)
        self.grid.eval(self.testobj)
        self.graph = grid2graph(self.grid)
        #self.graph.plot()

    def testobj(self, x_eval):
        """
        Compute the function U=U_attr+a*iU_rep,i, where a is given by the variable
        potential.repulsiveWeight and return True if occupied and False if not
        """
        if self.totalQ.eval(x_eval) > 0:
            return True
        else:
            return False

    def plot(self):
        """
        Plots the graph attribute
        """
        self.graph.plot()

    def run_plot(self):
        """
        - Load the variables  x_start,  x_goal from the internal attribute  sphereworld.
        homework4_sphereworldPlot
        """
        x_start = self.sphereworld.x_start
        x_goal = self.sphereworld.x_goal
        #first goal displays without G, second with G
        for i_goal in range(1, x_goal[0].size):
            plt.figure()
            self.sphereworld.plot()
            self.graph.plot()
            for i_start in range(0, x_start[0].size):
                temp_start = np.array([x_start[0][i_start], x_start[1][i_start]])
                temp_goal = np.array([x_goal[0][i_goal], x_goal[1][i_goal]])
                x_path = self.graph.search_start_goal(temp_start, temp_goal)
                plt.plot(x_path[0],x_path[1])
            plt.show()


def graph_test_data_load(variable_name):
    """
    Loads data from the file graph_test_data.pkl.
    """
    with open('graph_test_data.pkl', 'rb') as fid:
        saved_data = pickle.load(fid)
    return saved_data[variable_name]


def graph_test_data_plot():
    """
    Plot two solved graphs
    """

    graph = Graph(graph_test_data_load('graphVector_solved'))
    plt.figure()
    graph.plot(flag_heuristic=True, idx_goal=1)

    graph = Graph(graph_test_data_load('graphVectorMedium_solved'))
    plt.figure()
    graph.plot(flag_heuristic=True, idx_goal=14)


def grid2graph(grid):
    """
    The function returns a  Graph object described by the inputs. See Figure~  for an example of the
    expected inputs and outputs.
    """

    # Make sure values in F are logicals
    fun_evalued = np.vectorize(bool)(grid.fun_evalued)

    # Get number of columns, rows, and nodes
    nb_xx, nb_yy = fun_evalued.shape
    nb_nodes = np.sum(fun_evalued)

    # Get indeces of non-zero entries, and assign a progressive number to each
    idx_xx, idx_yy = np.where(fun_evalued)
    idx_assignment = range(0, nb_nodes)

    # Lookup table from xx,yy element to assigned index (-1 means not assigned)
    idx_lookup = -1 * np.ones(fun_evalued.shape, 'int')
    for i_xx, i_yy, i_assigned in zip(idx_xx, idx_yy, idx_assignment):
        idx_lookup[i_xx, i_yy] = i_assigned

    def grid2graph_neighbors(idx_xx, idx_yy):
        """
        Finds the neighbors of a given element
        """

        displacements = [(idx_xx + dx, idx_yy + dy) for dx in [-1, 0, 1]
                         for dy in [-1, 0, 1] if not (dx == 0 and dy == 0)]
        neighbors = []
        for i_xx, i_yy in displacements:
            if 0 <= i_xx < nb_xx and 0 <= i_yy < nb_yy and idx_lookup[
                i_xx, i_yy] >= 0:
                neighbors.append(idx_lookup[i_xx, i_yy].item())

        return neighbors

    # Create graph_vector data structure and populate 'x' and 'neighbors' fields
    graph_vector = [None] * nb_nodes
    for i_xx, i_yy, i_assigned in zip(idx_xx, idx_yy, idx_assignment):
        x_current = np.array([[grid.xx_grid[i_xx]], [grid.yy_grid[i_yy]]])
        neighbors = grid2graph_neighbors(i_xx, i_yy)
        graph_vector[i_assigned] = {'x': x_current, 'neighbors': neighbors}

    # Populate the 'neighbors_cost' field
    # Cannot be done in the loop above because not all 'x' fields would be filled
    for idx, n_current in enumerate(graph_vector):
        x_current = n_current['x']

        if len(n_current['neighbors']) > 0:
            x_neighbors = np.hstack(
                [graph_vector[idx]['x'] for idx in n_current['neighbors']])
            neighbors_cost_np = np.sum((x_neighbors - x_current) ** 2, 0)
            graph_vector[idx]['neighbors_cost'] = list(neighbors_cost_np)
        else:
            graph_vector[idx]['neighbors_cost'] = []

    return Graph(graph_vector)


def test_nearest_neighbors():
    """
    Tests Graph.nearest_neighbors by picking a random point and finding the 3 nearest neighbors
    in graphVectorMedium
    """
    graph = Graph(graph_test_data_load('graphVectorMedium'))
    x_query = np.array([[5], [4]]) * np.random.rand(2, 1)
    idx_neighbors = graph.nearest_neighbors(x_query, 3)
    graph.plot(node_lists=idx_neighbors)
    plt.scatter(x_query[[0]], x_query[[1]])


def test_grid2graph():
    """
    Tests grid2graph() by creating an arbitrary function returning bools
    """
    xx_grid = np.linspace(0, 2 * pi, 40)
    yy_grid = np.linspace(0, pi, 20)
    func = lambda x: (x[[1]] > pi / 2 or np.linalg.norm(x - np.ones(
        (2, 1))) < 0.75) and np.linalg.norm(x - np.array([[4], [2.5]])
                                            ) >= 0.5
    grid = me570_geometry.Grid(xx_grid, yy_grid)
    grid.eval(func)
    graph = grid2graph(grid)
    graph.plot()


if __name__ == "__main__":
    # graph_test_data_plot()
    # plt.show()
    # graph.search_start_goal()

    # report 2.1
    # nb cells too low
    # swg = SphereWorldGraph(6)
    # plt.show()
    # nb cells just right
    # swg = SphereWorldGraph(11)
    # plt.show()
    # nb cells too high
    # swg = SphereWorldGraph(100)
    # plt.show()

    # report 2.2
    # nb cells too low
    swg = SphereWorldGraph(6)
    swg.run_plot()
    # nb cells just right
    swg = SphereWorldGraph(11)
    swg.run_plot()
    # nb cells too high
    swg = SphereWorldGraph(40)
    swg.run_plot()
