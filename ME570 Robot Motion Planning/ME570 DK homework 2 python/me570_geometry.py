"""
 Please merge the functions and classes from this file with the same file from the previous
 homework assignment
"""

import math
import numbers

import numpy as np
from matplotlib import pyplot as plt


# import scipy as scipy

# import me570_robot

def gca_3d():
    """
    Get current Matplotlib axes, and if they do not support 3-D plotting,
    add new axes that support it
    """
    fig = plt.gcf()
    if len(fig.axes) == 0 or not hasattr(plt.gca(), 'plot3D'):
        axis = fig.add_subplot(111, projection='3d')
    else:
        axis = plt.gca()
    return axis


def numel(var):
    """
    Counts the number of entries in a numpy array, or returns 1 for fundamental numerical
    types
    """
    if isinstance(var, numbers.Number):
        size = int(1)
    elif isinstance(var, np.ndarray):
        size = var.size
    else:
        raise NotImplementedError(f'number of elements for type {type(var)}')
    return size


def rot2d(theta):
    """
    Create a 2-D rotation matrix from the angle theta according to (1).
    """
    # print("theta, ", theta)
    if isinstance(theta, np.ndarray):
        theta = theta[0]
    rot_theta = np.array([[math.cos(theta), -math.sin(theta)],
                          [math.sin(theta), math.cos(theta)]])
    # print("rot_theta, ", rot_theta)

    # plot to show that R(theta) is a rotation for all theta
    """
    r = np.arange(0, 2, 0.01)
    theta = 2 * np.pi * r
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(theta, r)
    ax.set_rmax(2)
    ax.set_rticks([0.5, 1, 1.5, 2])  # Less radial ticks
    ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    ax.grid(True)
    
    ax.set_title("A line plot on a polar axis", va='bottom')
    plt.show()
    """
    return rot_theta


def rot2d_test(rotm):
    """
    a function to check if a 2d matrix with a theta value passed in is a rotation matrix
    could not use/plot generic theta as originally planned
    true if RT - R^-1 = 0 +/- tol and det R =1
    """
    # tolerance to check for rounding errors
    tol = 2.22e-16
    dif = rotm.transpose() - np.linalg.inv(rotm)
    try:
        detrotm = np.linalg.det(rotm)
    except np.linalg.LinAlgError:
        return False
    if (-tol <= dif[0, 0] <= tol and -tol <= dif[0, 1] <= tol and -tol <= dif[
        1, 0] <= tol and -tol <=
            dif[1, 1] <= tol and 1 - tol <= detrotm <= 1 + tol):
        return True
    # if np.array_equal(rotm.transpose(), np.linalg.inv(rotm)):
    #    return True
    return False

def rot2d_test_circle():
    """
    a function to check if all theta valuse for phicircle(t) =  rot(theta)[1,0]
    plot to a circle S1
    """
    theta = np.arange(0, 2*np.pi, 0.01)
    point = np.zeros((2, theta.size))
    fig, ax = plt.subplots()
    for i in range(0,theta.size):
        calc = np.matmul(rot2d(theta[i]),np.array([[1],[0]]))
        point[:,[i]] = calc
    ax.scatter(point[0], point[1])
    ax.grid(True)

    ax.set_title("phicircle(t) =  rot(theta)[1,0]", va='bottom')
    plt.show()


def from2dto3d(eqnum, theta):
    """
    implements all 5 R1 to R5 equations
    inputs:
        eqnum = equation number to use
        theta = theta to use
    output = output matrix
    """
    a_1 = rot2d(theta)
    match eqnum:
        case 1:
            # rotation of theta about the x axis
            a_1 = rot2d(theta)
            # return scipy.linalg.block_diag(1, a_1)
            return np.array([[1, 0, 0], [0, a_1[0, 0], a_1[0, 1]], [0, a_1[1, 0], a_1[1, 1]]])

        case 2:
            # rotation of theta about the y axis
            a_1 = rot2d(theta)
            return np.array([[a_1[0, 0], 0, a_1[0, 1]], [0, 1, 0], [a_1[1, 0], 0, a_1[1, 1]]])

        case 3:
            # rotation of theta about the z axis
            a_1 = rot2d(theta)
            return np.array([[a_1[0, 0], a_1[0, 1], 0], [a_1[1, 0], a_1[1, 1], 0], [0, 0, 1]])
            # return scipy.linalg.block_diag(a_1, 1)

        case 4:
            # negative rotation of theta about the x axis
            a_1 = rot2d(theta)
            return np.array([[-a_1[0, 0], -a_1[0, 1], 0], [-a_1[1, 0], -a_1[1, 1], 0], [0, 0, 1]])
            # return scipy.linalg.block_diag(-1*a_1, 1)

        case 5:
            # rotation of negative theta about the x axis
            a_1 = rot2d(-1 * theta)
            return np.array([[a_1[0, 0], a_1[0, 1], 0], [a_1[1, 0], a_1[1, 1], 0], [0, 0, 1]])
            # return scipy.linalg.block_diag(a_1, 1)
        case _:
            output = False
            return output
    """
        print("from2dto3d(*args): ", args)
    #case R2
    if len(args) == 1:
        a0 = args[0]
        a_1 = rot2d(a0)
        print(a_1)
        output = np.array([a_1[0, 0], 0, a_1[0, 1], 0, 1, 0, a_1[1, 0], 0, a_1[1, 1]]).reshape(3, 3)
    else:
        #all other Rs
        output = scipy.linalg.block_diag(args[0], args[1])
    #old
    if len(args) == 1:
        a_1 = args[0]
        output = np.array([a_1[0,0], 0, a_1[0,1],0, 1, 0, a_1[1,0], 0, a_1[1,1]]).reshape(3,3)
    else:
        output = scipy.linalg.block_diag(args[0], args[1])
    return output
    """


def line_linspace(a_line, b_line, t_min, t_max, nb_points):
    """
    Generates a discrete number of  nb_points points along the curve
    (t)=( a(1)t + b(1), a(2)t + b(2))  R^2 for t ranging from  tMin to  tMax.
    """
    linepoints = np.linspace(t_min, t_max, nb_points)
    theta_points = np.zeros((2, nb_points))
    for i in range(0, nb_points):
        theta_points[0, i] = a_line.item(0) * linepoints[i] + b_line.item(0)
        theta_points[1, i] = a_line.item(1) * linepoints[i] + b_line.item(1)
    # print("theta_points, ", theta_points)
    return theta_points


class Grid:
    """
    A function to store the coordinates of points on a 2-D grid and evaluate arbitrary
    functions on those points.
    """

    def __init__(self, xx_grid, yy_grid):
        """
        Stores the input arguments in attributes.
        """
        self.xx_grid = xx_grid
        self.yy_grid = yy_grid

    def eval(self, fun):
        """
        This function evaluates the function  fun (which should be a function)
        on each point defined by the grid.
        """
        dim_domain = [numel(self.xx_grid), numel(self.yy_grid)]
        dim_range = [numel(fun(np.array([[0], [0]])))]
        fun_eval = np.nan * np.ones(dim_domain + dim_range)
        for idx_x in range(0, dim_domain[0]):
            for idx_y in range(0, dim_domain[1]):
                x_eval = np.array([[self.xx_grid[idx_x]],
                                   [self.yy_grid[idx_y]]])
                fun_eval[idx_x, idx_y, :] = np.reshape(fun(x_eval),
                                                       [1, 1, dim_range[0]])

        # If the last dimension is a singleton, remove it
        if dim_range == [1]:
            fun_eval = np.reshape(fun_eval, dim_domain)

        return fun_eval

    def mesh(self):
        """
        Shorhand for calling meshgrid on the points of the grid
        """
        return np.meshgrid(self.xx_grid, self.yy_grid)


class Torus:
    """
    A class that holds functions to compute the embedding and display a torus and curves on it.
    """

    def phi(self, theta):
        """
        Implements equation (eq:chartTorus).
        """
        # print(theta)
        radius = 3
        phi_torus = np.zeros((3, theta[0].size))
        # print("phi_torus, ", phi_torus)
        for i in range(0, theta[0].size):
            mat3by2 = np.reshape(np.array([1, 0, 0, 0, 0, 1]), (3, 2))
            r_3 = from2dto3d(3, theta[1, i])
            # print("r_3, ", r_3)
            b_3 = np.array([[1], [0]])
            # print("b_3, ", b_3)
            rot = rot2d(theta[0])
            # print("rot, ", rot)
            phicircle = np.matmul(rot, b_3)
            # print("b2, ", phicircle)
            mat3by2xphicircle = np.matmul(mat3by2, phicircle)
            # print("mat3by2xphicircle, ", mat3by2xphicircle)
            b_lint = mat3by2xphicircle + np.array([[radius], [0], [0]])
            # print("b_lint, ", b_lint)
            phi_torusi = np.matmul(r_3, b_lint)
            # print("phi_torusi, ", phi_torusi)
            # phi_torus[0,i] = phi_torusi[0,0]
            # phi_torus[1, i] = phi_torusi[1, 0]
            # phi_torus[2, i] = phi_torusi[2, 0]
            phi_torus[:, [i]] = phi_torusi[:, [i]]
            # print("phi_torus, ", phi_torus)

        # pass  # Substitute with your code
        return phi_torus

    def plot(self):
        """
        For each one of the chart domains U_i from the previous question:
        - Fill a  grid structure with fields  xx_grid and  yy_grid that define a grid of regular
          point in U_i. Use nb_grid=33.
        - Call the function Grid.eval with argument Torus.phi.
        - Plots the surface described by the previous step using the Matplotlib function
        ax_lint.plot_surface (where  ax_lint represents the axes of the current figure) in a
        separate figure. Plot a final additional figure showing all the charts at the same time.
        To better show the overlap between the charts, you can use different colors each one of
        them, and making them slightly transparent.
        """
        ax_lint = plt.gca()
        nb_grid = 33
        nb_grid_x = np.linspace(0, 2 * np.pi, nb_grid)
        nb_grid_y = np.linspace(0, 2 * np.pi, nb_grid)
        gridxy = Grid(nb_grid_x, nb_grid_y)
        # print("gridxy_x, ", nb_grid_x)
        # print("gridxy_y, ", nb_grid_y)
        func_eval = gridxy.eval(self.phi)
        # print("func_eval, ", func_eval)
        ax_lint = plt.axes(projection='3d')
        """        
        for i in range(0,nb_grid):
            if i == 0:
                ax_lint.plot_surface(func_eval[:,nb_grid-1], func_eval[:,0], func_eval[:,1])
            elif i == 32:
                ax_lint.plot_surface(func_eval[:,nb_grid-2], func_eval[:,nb_grid-1], func_eval[:,0])
            else:
                ax_lint.plot_surface(func_eval[:,i-1], func_eval[:,i], func_eval[:,i+1])
        """
        # print("func_eval[0, :], ", func_eval[:,:,0])

        ax_lint.plot_surface(func_eval[:, :, 0], func_eval[:, :, 1], func_eval[:, :, 2], alpha=.5)

    def phi_push_curve(self, a_line, b_line):
        """
        This function evaluates the curve x(t)= phi_torus ( phi(t) )  R^3 at  nb_points=31 points
        generated along the curve phi(t) using line_linspaceLine.linspace with  tMin=0 and  tMax=1,
        and a, b as given in the input arguments.
        """
        num_points = 31
        line_points = line_linspace(a_line, b_line, 0, 1, num_points)

        # print("line_points, ", line_points)

        x_points = np.zeros((3, num_points))

        for i in range(0, num_points):
            # x_points[i] = self.phi()
            x_points[:, [i]] = self.phi(line_points[:, [i]])

        # print("x_points, ", x_points)
        return x_points

    def plot_curves(self):
        """
        The function should iterate over the following four curves:
        - 3/4*pi0
        - 3/4*pi3/4*pi
        - -3/4*pi3/4*pi
        - 0 -3/4*pi  and  b=np.array([[-1],[-1]]).
        The function should show an overlay containing:
        - The output of Torus.plotCharts;
        - The output of the functions torus_pushCurveTorus.pushCurve for each one of the curves.
        """
        a_lines = [
            np.array([[3 / 4 * math.pi], [0]]),
            np.array([[3 / 4 * math.pi], [3 / 4 * math.pi]]),
            np.array([[-3 / 4 * math.pi], [3 / 4 * math.pi]]),
            np.array([[0], [-3 / 4 * math.pi]])
        ]

        b_line = np.array([[-1], [-1]])

        axis = gca_3d()
        for a_line in a_lines:
            x_points = self.phi_push_curve(a_line, b_line)
            axis.plot(x_points[0, :], x_points[1, :], x_points[2, :], linewidth=3)

    def phi_test(self):
        """
        Uses the function phi to plot two perpendicular rings
        """
        nb_points = 200
        theta_ring = np.linspace(0, 15 / 8 * np.pi, nb_points)
        theta_zeros = np.zeros((1, nb_points))
        data = [
            np.vstack((theta_ring, theta_zeros)),
            np.vstack((theta_zeros, theta_ring))
        ]
        axis = gca_3d()
        for theta in data:
            ring = np.zeros((3, nb_points))
            for idx in range(nb_points):
                ring[:, [idx]] = self.phi(theta[:, [idx]])
            axis.plot(ring[0, :], ring[1, :], ring[2, :])


# HW1 Code
class Polygon:
    """
    Class for plotting, drawing, checking visibility and collision with
    polygons.
    vertices (dim. [2 × nb_vertices], type nparray ): array where each column represents the
    coordinates of a vertex in the polygon.
    """

    def __init__(self, vertices):
        """
        Save the input coordinates to the internal attribute  vertices.
        """
        self.vertices = vertices

    def flip(self):
        """
        Reverse the order of the vertices (i.e., transform the polygon from
        filled in to hollow and viceversa).
        """
        vsize = self.vertices[0].size
        tempx = np.zeros(self.vertices[0].size)
        tempy = np.zeros(self.vertices[0].size)
        for i in range(0, vsize):
            tempx[vsize - i - 1] = self.vertices[0, i]
            tempy[vsize - i - 1] = self.vertices[1, i]
        self.vertices = np.array([tempx, tempy])

    def plot(self, style):
        """
        Plot the polygon using Matplotlib.
        """
        # close the loop
        tempvertices = np.array([np.append(self.vertices[0, :], self.vertices[0, 0]),
                                 np.append(self.vertices[1, :], self.vertices[1, 0])])
        vsize = tempvertices[0].size
        arrowsx = np.zeros(vsize - 1)
        arrowsy = np.zeros(vsize - 1)

        if self.is_filled():
            plt.gca().set_facecolor("white")
            plt.fill(tempvertices[0], tempvertices[1], "grey")
        else:
            plt.gca().set_facecolor("grey")
            plt.fill(tempvertices[0], tempvertices[1], "white")

        for i in range(0, vsize - 1):
            if i != vsize - 1:
                arrowsx[i] = tempvertices[0, i + 1] - tempvertices[0, i]
                arrowsy[i] = tempvertices[1, i + 1] - tempvertices[1, i]
        plt.quiver(tempvertices[0, 0:vsize - 1], tempvertices[1, 0:vsize - 1],
                   arrowsx, arrowsy, angles='xy', scale=1,
                   scale_units='xy', color=style)
        plt.ylabel('Y')
        plt.xlabel('X')
        plt.show(block=False)

    def is_filled(self):
        """
        Checks the ordering of the vertices, and returns whether the polygon is
        filled in or not.
        """
        tol = 2.22e-16
        vsize = self.vertices[0].size
        asum = 0

        fill = True
        for i in range(0, vsize):
            v_cur = np.reshape(self.vertices[:, i], (2, 1))

            if i - 1 < 0:
                v_prev = np.reshape(self.vertices[:, i - 1 + vsize], (2, 1))
            else:
                v_prev = np.reshape(self.vertices[:, i - 1], (2, 1))

            if i + 1 >= vsize:
                v_next = np.reshape(self.vertices[:, i + 1 - vsize], (2, 1))
            else:
                v_next = np.reshape(self.vertices[:, i + 1], (2, 1))

            # print("v_cur ", v_cur)
            # print("v_prev ", v_prev)
            # print("v_next ", v_next)
            atemp = angle(v_cur, v_prev, v_next, 'unsigned')
            asum += atemp

        if asum <= (np.pi + tol) * self.vertices[0].size:
            fill = False
        return fill

    def is_self_occluded(self, idx_vertex, point):
        """
        Given the corner of a polygon, checks whether a given point is self-occluded or not by that
        polygon (i.e., if it is ``inside'' the corner's cone or not). Points on boundary (i.e., on
        one of the sides of the corner) are not considered self-occluded. Note that to check
        self-occlusion, we just need a vertex index  idx_vertex. From this, one can obtain the
        corresponding  vertex, and the  vertex_prev and vertex_next that precede and follow that
        vertex in the polygon. This information is sufficient to determine self-occlusion. To
        convince yourself, try to complete the corners shown in Figure~ fig:self-occlusion with
        clockwise and counterclockwise polygons, and you will see that, for each example, only one
        of these cases can be consistent with the arrow directions.
        """
        vcur = np.reshape(self.vertices[:, idx_vertex], (2, 1))

        # add isfilled check to ensure proper order of vertices
        filled = self.is_filled()

        if filled:
            if idx_vertex + 1 == self.vertices[0].size:
                vnext = np.reshape(self.vertices[:, (idx_vertex + 1 - (self.vertices[0].size))],
                                   (2, 1))
            else:
                vnext = np.reshape(self.vertices[:, idx_vertex + 1], (2, 1))
            if idx_vertex - 1 < 0:
                vprev = np.reshape(self.vertices[:, (idx_vertex - 1 + (self.vertices[0].size))],
                                   (2, 1))
            else:
                vprev = np.reshape(self.vertices[:, idx_vertex - 1], (2, 1))
        else:
            # not filled, previous is + next is -
            if idx_vertex - 1 < 0:
                vnext = np.reshape(self.vertices[:, (idx_vertex - 1 + (self.vertices[0].size))],
                                   (2, 1))
            else:
                vnext = np.reshape(self.vertices[:, idx_vertex - 1], (2, 1))
            if idx_vertex + 1 == self.vertices[0].size:
                vprev = np.reshape(self.vertices[:, (idx_vertex + 1 - (self.vertices[0].size))],
                                   (2, 1))
            else:
                vprev = np.reshape(self.vertices[:, idx_vertex + 1], (2, 1))

        # check if on edge lines prev-cur and cur-next
        a_1 = angle(point, vprev, vcur, 'unsigned')
        a_2 = angle(point, vcur, vnext, 'unsigned')
        if math.isnan(a_1) or math.isnan(a_2) or a_1 == np.pi or a_2 == np.pi:
            return False

        if filled:
            avert = angle(vcur, vprev, vnext, 'unsigned')
            apoint = angle(vcur, vprev, point, 'unsigned')
            if avert < apoint:
                return True
        else:
            avert = angle(vcur, vprev, vnext, 'unsigned')
            apoint = angle(vcur, vprev, point, 'unsigned')
            if avert > apoint:
                return True
        return False

    def is_visible(self, idx_vertex, test_points):
        """
        Checks whether a point p is visible from a vertex v of a polygon. In
        order to be visible, two conditions need to be satisfied:
         - The point p should not be self-occluded with respect to the vertex
        v\\ (see Polygon.is_self_occluded).
         - The segment p--v should not collide with  any of the edges of the
        polygon (see Edge.is_collision).
        """
        i_v = idx_vertex - 1
        flag_points = np.ones(test_points[0].size, dtype=bool)
        for i in range(0, test_points[0].size):
            if np.array_equal(test_points[:, i], self.vertices[:, i_v]):
                pass
            else:
                if self.is_self_occluded(i_v, np.reshape(test_points[:, i], (2, 1))):
                    flag_points[i] = False
                testedge = Edge(np.array([np.append(self.vertices[0, i_v], test_points[0, i]),
                                          np.append(self.vertices[1, i_v], test_points[1, i])]))

                # check edge collisions
                for j in range(0, self.vertices[0].size):
                    if j + 1 == self.vertices[0].size:
                        poledge = Edge(np.array([np.append(self.vertices[0, j],
                                                           self.vertices[
                                                               0, j + 1 - self.vertices[
                                                                   0].size]),
                                                 np.append(self.vertices[1, j],
                                                           self.vertices[1, j + 1 -
                                                                            self.vertices[
                                                                                0].size])]))
                    else:
                        poledge = Edge(np.array([np.append(self.vertices[0, j],
                                                           self.vertices[0, j + 1]),
                                                 np.append(self.vertices[1, j],
                                                           self.vertices[1, j + 1])]))
                    if testedge.is_collision(poledge):
                        flag_points[i] = False
        # print(flag_points)

        return flag_points

    def is_collision(self, test_points):
        """
        Checks whether the a point is in collsion with a polygon (that is,
        inside for a filled in polygon, and outside for a hollow polygon). In
        the context of this homework, this function is best implemented using
        Polygon.is_visible.
        """
        # if not visible and not self occluded
        is_visiblefrom_v = np.zeros(test_points[0].size, dtype=bool)

        # for each test point, iterate around polygon
        # if any true, is not in collision, otherwie is
        # check visibility for all vertices, if one is visible, is not in collision
        for i in range(1, self.vertices[0].size + 1):
            tempis_visiblefrom_v = self.is_visible(i, test_points)
            is_visiblefrom_v = np.logical_or(is_visiblefrom_v, tempis_visiblefrom_v)
        flag_points = np.invert(is_visiblefrom_v)

        return flag_points

    def kinematic_map(self, theta, w_t_b):
        """
        The function returns the coordinate of the vertices of the link, all
        transformed according to  theta_1.
        theta and w_t_b are both arrays corresponding to the degrees and distance to the current
        link where the lowest corresponds to the first link and the highest corresponds to the
        current

        wRb2 = wRb1 * b1Rb2
        w_t_b2 = wRb1 * b1Tb2 + w_t_b1
        w_x = wRb2 * b2x + w_t_b2
        b2x = [5 0]
        w_t_b1 = 0
        b1Tb2 = [5 0]
        """
        i_r_j = np.zeros((2, 2))
        i_t_j = np.zeros((2, w_t_b[0].size))
        # print(i_t_j)
        for i in range(0, theta.size):
            if i == 0:
                i_r_j = np.array([np.array(rot2d(theta[0]))])
                i_t_j[:, i] = w_t_b[:, 0]
            else:
                i_r_j = np.append(i_r_j, np.matmul(rot2d(theta[i]), rot2d(theta[i - 1])))
                i_t_j[:, i] = np.matmul(i_r_j[i - 1], i_t_j[:, i - 1]) + w_t_b[:, i - 1]

        new_xv = np.zeros(self.vertices[0].size)
        new_yv = np.zeros(self.vertices[1].size)
        for i in range(0, self.vertices[0].size):
            rot = i_r_j[theta.size - 1]
            x_lint = self.vertices[:, i]
            w_x = np.matmul(rot, x_lint) + i_t_j[:, i_t_j[0].size - 1]
            new_xv[i] = w_x[0]
            new_yv[i] = w_x[1]

        new_pol = Polygon(np.array([new_xv, new_yv]))
        return new_pol

        """
        theta1 = theta[0]
        theta2 = theta[1]
        wRb1 = rot2d(theta1)
        b1Rb2 = rot2d(theta2)
        b1Tb2 = np.array([[5], [0]])
        w_t_b1 = np.array([[0], [0]])
        polygon1_transf_vert = np.zeros((2,self.vertices[0].size))
        for v in range(0, self.vertices[0].size):
            b2x = np.array([[self.vertices[0, v]], [self.vertices[1, v]]])
            print("b2x, ", b2x)
            # wRb2 = np.matmul(wRb1, b1Rb2)
            # w_t_b2 = np.add(np.matmul(wRb1, b1Tb2), w_t_b1)
            # wXw = np.matmul(wRb2, b2x)
            # wX2 = np.add(np.matmul(wRb2, b2x), w_t_b2)
            w_x = np.matmul(np.matmul(wRb1, b1Rb2), b2x) + np.matmul(wRb1, b1Tb2) + w_t_b1
            polygon1_transf_vert[0, v] = w_x[0, 0]
            polygon1_transf_vert[1, v] = w_x[1, 0]
            print("polygon2_transf_vert[:,v], ", polygon1_transf_vert[:, v])
        newPol1 = Polygon(polygon1_transf_vert)
        return newPol1"""


class Edge:
    """
    Class for storing edges and checking collisions among them.
    """

    def __init__(self, vertices):
        """
        Save the input coordinates to the internal attribute  vertices.
        """
        self.vertices = vertices

    def is_collision(self, edge):
        """
         Returns  True if the two edges intersect.  Note: if the two edges
        overlap but are colinear, or they overlap only at a single endpoint,
        they are not considered as intersecting (i.e., in these cases the
        function returns  False). If one of the two edges has zero length, the
        function should always return the result that edges are
        non-intersecting.
        """
        # here
        # take out vertices and just pull the values into the matrices for calculation
        xa1 = np.reshape(self.vertices[:, 0], (2, 1))
        xa2 = np.reshape(self.vertices[:, 1], (2, 1))
        xb1 = np.reshape(edge.vertices[:, 0], (2, 1))
        xb2 = np.reshape(edge.vertices[:, 1], (2, 1))
        xa1calc = self.vertices[:, 0]
        xa2calc = self.vertices[:, 1]
        xb1calc = edge.vertices[:, 0]
        xb2calc = edge.vertices[:, 1]
        a_lint = np.array([xa2calc - xa1calc, xb2calc - xb1calc])
        b_lint = xb2calc - xa1calc
        a_lint = np.transpose(a_lint)
        try:
            t_lint = np.linalg.solve(a_lint, b_lint)
        except np.linalg.LinAlgError:
            return False
        int_point = np.reshape(t_lint, (2, 1))
        # ta and tb are the percentages "down the line from a1 to a2 and b1 to b2
        # if 0<t<1 then on the line

        # checking vertices equal
        if (np.array_equal(xa1, xb1) or np.array_equal(xa1, xb2) or
                np.array_equal(xa2, xb2)):
            return False

        # checking int_point = vertices
        if (np.array_equal(int_point, xa1) or np.array_equal(int_point, xa2) or
                np.array_equal(int_point, xb1) or np.array_equal(int_point, xb2)):
            return False

        if 0 < int_point[0, 0] < 1 and 0 < int_point[1, 0] < 1:
            return True

        return False

    def plot(self, *args, **kwargs):
        """ Plot the edge """
        plt.plot(self.vertices[0, :], self.vertices[1, :], *args, **kwargs)

    def cprint(self):
        """ Prints the vertices of an edge"""
        # print("ev1: ", self.vertices[0, :])
        # print("ev2: ", self.vertices[1, :])
        # pass


def angle(vertex0, vertex1, vertex2, angle_type='unsigned'):
    # vCur   vPrev   vNext
    """
    Compute the angle between two edges  vertex0-- vertex1 and  vertex0--
    vertex2 having an endpoint in common. The angle is computed by starting
    from the edge  vertex0-- vertex1, and then ``walking'' in a
    counterclockwise manner until the edge  vertex0-- vertex2 is found.
    """
    # tolerance to check for coincident points
    tol = 2.22e-16

    # compute vectors corresponding to the two edges, and normalize
    vec1 = vertex1 - vertex0
    vec2 = vertex2 - vertex0

    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 < tol or norm_vec2 < tol:
        # vertex1 or vertex2 coincides with vertex0, abort
        edge_angle = math.nan
        return edge_angle

    vec1 = vec1 / norm_vec1
    vec2 = vec2 / norm_vec2

    # Transform vec1 and vec2 into flat 3-D vectors,
    # so that they can be used with np.inner and np.cross
    vec1flat = np.vstack([vec1, 0]).flatten()
    vec2flat = np.vstack([vec2, 0]).flatten()

    # c_angle is the value of cos(theta)
    # s_angle is the value of sin(theta)
    c_angle = np.inner(vec1flat, vec2flat)
    s_angle = np.inner(np.array([0, 0, 1]), np.cross(vec1flat, vec2flat))

    # taking the arctangent of the sine and cosine values gives us the
    # angle between the two vectors in radians
    edge_angle = math.atan2(s_angle, c_angle)

    angle_type = angle_type.lower()
    if angle_type == 'signed':
        # nothing to do
        pass
    elif angle_type == 'unsigned':
        edge_angle = (edge_angle + 2 * math.pi) % (2 * math.pi)
    else:
        raise ValueError('Invalid argument angle_type')

    return edge_angle


plt.figure()

# r = np.arange(0, 2, 0.01)
"""r = []
for i in range(0,200,1):
    r.append(rot2d(i/200))

print(r)

theta = []
for i in r:
    theta[i] = 2.0 * np.pi * r[i]

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(theta, r)
ax.set_rmax(2)
ax.set_rticks([0.5, 1, 1.5, 2])  # Less radial ticks
ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
ax.grid(True)

ax.set_title("A line plot on a polar axis", va='bottom')
plt.show()"""

# Rot matrix test
# RT = R−1 and det R = 1.
#rot_theta_test = np.array([[math.cos(1), -math.sin(1)], [math.sin(1), math.cos(1)]])
# print(rot2d_test(rot_theta_test))
#print(rot2d_test_circle())

# print(scipy.linalg.block_diag(1,rot2d(1)))
# print(from2dto3d(1,1))
# print(from2dto3d(1))

"""#th = 45 * np.pi /180
th = 45
a2 = rot2d(45)
print(me570_robot.polygons)
r2 = me570_robot.polygons(1) * a2
r2.plot('STYLE')

# plt.show()
torusT = Torus()
#torusT.phi_test()
torusT.phi_push_curve(np.array([[1],[2]]),np.array([[0],[0]]))
torusT.plot()

plt.show()
plt.figure()

plt.show()
plt.figure()

torusT.plot_curves()

"""
"""
vertices1 = np.array([[0, 5, 5, 0], [-1.11, -0.511, 0.511, 1.11]])
pol1 = Polygon(vertices1)
pol1.plot("STYLE")

plt.show()

plt.figure()
th = 45 * np.pi / 180
pol2 = pol1.kinematic_map(np.array([th]), np.array([[0],[0]]))
pol2.plot("STYLE")"""

plt.show()
