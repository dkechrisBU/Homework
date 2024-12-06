"""
Demetrios Kechris
ME570
HW4
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import io as scio

import me570_robot
import me570_graph

plt.ion()


def graph_search_test():
    """
    Call graph_search to find a path between the bottom left node and the
    top right node of the  graphVectorMedium graph from the
    graph_test_data_load function (see Question~ q:graph test data). Then
    use Graph.plot() to visualize the result.
    """
    graph = me570_graph.Graph(me570_graph.graph_test_data_load('graphVectorMedium'))
    plt.figure()
    graph.search(0, 14,'astar','True')
    graph.plot(flag_heuristic=True, idx_goal=14)


def twolink_search_plot_solution(theta_path):
    '''
    Plot a two-link path both on the graph and in the workspace
    '''
    twolink_graph = me570_robot.TwoLinkGraph()
    plt.figure(1)
    twolink_graph.plot()
    plt.plot(theta_path[0, :], theta_path[1, :], 'r')

    twolink = me570_robot.TwoLink()
    obstacle_points = scio.loadmat('twolink_testData.mat')['obstaclePoints']
    plt.figure(2)
    plt.scatter(obstacle_points[0, :], obstacle_points[1, :], marker='*')
    twolink.animate(theta_path)

    plt.show()


def twolink_test_path():
    '''
    Visualize, both in the graph, and in the workspace, a sample path where the second link rotates
    and then the first link rotates (both with constant speeds).
    '''
    theta_m = 3 / 4 * np.pi
    theta_path = np.vstack((np.zeros((1, 75)), np.linspace(0, theta_m, 75)))
    theta_path = np.hstack(
        (theta_path,
         np.vstack((np.linspace(0, theta_m, 75), theta_m * np.ones((1, 75))))))
    twolink_search_plot_solution(theta_path)


if __name__ == "__main__":
    #graph_search_test()
    #plt.show()
    twolink_test_path()
    plt.figure()
    plt.show()