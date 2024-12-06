"""
Classes and functions for Polygons and Edges
#main sources:
python/np/matlibplot basics
https://np.org/doc/stable/reference/
https://matplotlib.org/
stackoverflow
"""

import math

import numpy as np
from matplotlib import pyplot as plt


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
        for x in range(0, vsize):
            tempx[vsize - x - 1] = self.vertices[0, x]
            tempy[vsize - x - 1] = self.vertices[1, x]
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

        for x in range(0, vsize - 1):
            if x != vsize - 1:
                arrowsx[x] = tempvertices[0, x + 1] - tempvertices[0, x]
                arrowsy[x] = tempvertices[1, x + 1] - tempvertices[1, x]
        plt.quiver(tempvertices[0, 0:vsize - 1], tempvertices[1, 0:vsize - 1],
                   arrowsx, arrowsy, angles='xy', scale=1,
                   scale_units='xy')
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
        for x in range(0, vsize):
            vCur = np.reshape(self.vertices[:, x], (2, 1))

            if x - 1 < 0:
                vPrev = np.reshape(self.vertices[:, x - 1 + vsize], (2, 1))
            else:
                vPrev = np.reshape(self.vertices[:, x - 1], (2, 1))

            if x + 1 >= vsize:
                vNext = np.reshape(self.vertices[:, x + 1 - vsize], (2, 1))
            else:
                vNext = np.reshape(self.vertices[:, x + 1], (2, 1))

            print("vCur ", vCur)
            print("vPrev ", vPrev)
            print("vNext ", vNext)
            atemp = angle(vCur, vPrev, vNext, 'unsigned')
            asum += atemp

        if asum <= (np.pi + tol) * self.vertices[0].size:
            fill = False
        return fill

    def is_self_occluded(self, idx_vertex, point):
        """
        Given the corner of a polygon, checks whether a given point is
        self-occluded or not by that polygon (i.e., if it is ``inside'' the
        corner's cone or not). Points on boundary (i.e., on one of the sides of
        the corner) are not considered self-occluded. Note that to check
        self-occlusion, we just need a vertex index  idx_vertex. From this, one
        can obtain the corresponding  vertex, and the  vertex_prev and
        vertex_next that precede and follow that vertex in the polygon. This
        information is sufficient to determine self-occlusion. To convince
        yourself, try to complete the corners shown in Figure~
        fig:self-occlusion with clockwise and counterclockwise polygons, and
        you will see that, for each example, only one of these cases can be
        consistent with the arrow directions.
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
        a1 = angle(point, vprev, vcur, 'unsigned')
        a2 = angle(point, vcur, vnext, 'unsigned')
        if math.isnan(a1) or math.isnan(a2) or a1 == np.pi or a2 == np.pi:
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
        iv = idx_vertex - 1
        flag_points = np.ones(test_points[0].size, dtype=bool)
        for i in range(0, test_points[0].size):
            if np.array_equal(test_points[:, i], self.vertices[:, iv]):
                pass
            else:
                if self.is_self_occluded(iv, np.reshape(test_points[:, i], (2, 1))):
                    flag_points[i] = False
                testedge = Edge(np.array([np.append(self.vertices[0, iv], test_points[0, i]),
                                          np.append(self.vertices[1, iv], test_points[1, i])]))

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
        print(flag_points)

        return flag_points

    def is_collision(self, test_points):
        """
        Checks whether the a point is in collsion with a polygon (that is,
        inside for a filled in polygon, and outside for a hollow polygon). In
        the context of this homework, this function is best implemented using
        Polygon.is_visible.
        """
        # if not visible and not self occluded
        isVisiblefromV = np.zeros(test_points[0].size, dtype=bool)

        # for each test point, iterate around polygon
        # if any true, is not in collision, otherwie is
        # check visibility for all vertices, if one is visible, is not in collision
        for v in range(1, self.vertices[0].size + 1):
            tempisVisiblefromV = self.is_visible(v, test_points)
            isVisiblefromV = np.logical_or(isVisiblefromV, tempisVisiblefromV)
        flag_points = np.invert(isVisiblefromV)

        return flag_points


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
        a = np.array([xa2calc - xa1calc, xb2calc - xb1calc])
        b = xb2calc - xa1calc
        a = np.transpose(a)
        try:
            t = np.linalg.solve(a, b)
        except Exception as LinAlgError:
            return False
        intPoint = np.reshape(t, (2, 1))
        # ta and tb are the percentages "down the line from a1 to a2 and b1 to b2
        # if 0<t<1 then on the line

        # checking vertices equal
        if (np.array_equal(xa1, xb1) or np.array_equal(xa1, xb2) or
                np.array_equal(xa2, xb2)):
            return False

        # checking intpoint = vertices
        if (np.array_equal(intPoint, xa1) or np.array_equal(intPoint, xa2) or
                np.array_equal(intPoint, xb1) or np.array_equal(intPoint, xb2)):
            return False

        if 0 < intPoint[0, 0] < 1 and 0 < intPoint[1, 0] < 1:
            return True

        return False

    def plot(self, *args, **kwargs):
        """ Plot the edge """
        plt.plot(self.vertices[0, :], self.vertices[1, :], *args, **kwargs)

    def cprint(self):
        """ Prints the vertices of an edge"""
        print("ev1: ", self.vertices[0, :])
        print("ev2: ", self.vertices[1, :])


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

vertices1 = np.array([[0, 5, 5, 0], [-1.11, -0.511, 0.511, 1.11]])
pol = Polygon(vertices1)
pol.plot("STYLE")

plt.show()
#occluded
pol.flip()
pol.plot('STYLE')

plt.show()
