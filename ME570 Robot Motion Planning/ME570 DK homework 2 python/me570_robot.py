"""
 Please merge the functions and classes from this file with the same file from the previous
 homework assignment
"""

import numpy as np
from matplotlib import pyplot as plt

import me570_geometry as geometry



def polygons_add_x_reflection(vertices):
    """
    Given a sequence of vertices, adds other vertices by reflection
    along the x axis
    """
    vertices = np.hstack([vertices, np.fliplr(np.diag([1, -1]).dot(vertices))])
    return vertices


def polygons_generate():
    """
    Generate the polygons to be used for the two-link manipulator
    """
    # modified to close polygon
    vertices1 = np.array([[0, 5], [-1.11, -0.511]])
    vertices1 = polygons_add_x_reflection(vertices1)
    vertices2 = np.array([[0, 3.97, 4.17, 5.38, 5.61, 4.5],
                          [-0.47, -0.5, -0.75, -0.97, -0.5, -0.313]])
    vertices2 = polygons_add_x_reflection(vertices2)
    return (geometry.Polygon(vertices1), geometry.Polygon(vertices2))


polygons = polygons_generate()

class TwoLink:
    """ This class was introduced in a previous homework. """




    def kinematic_map(self, theta):
        """
        The function returns the coordinate of the end effector, plus the vertices of the links, all
        transformed according to  theta_1, theta_2.

        wRb2 = w_r_b1 * b1_r_b2
        wTb2 = w_r_b1 * b1_t_b2 + w_t_b1
        w_x = wRb2 * b2x + wTb2
        b2x = [5 0]
        w_t_b1 = 0
        b1_t_b2 = [5 0]
        """
        b2peff = np.array([[5], [0]])
        w_r_b1 = geometry.rot2d(theta[0])
        b1_r_b2 = geometry.rot2d(theta[1])
        b1_t_b2 = np.array([[5], [0]])
        w_t_b1 = np.array([[0], [0]])
        vertex_effector_transf = (np.matmul(np.matmul(w_r_b1, b1_r_b2), b2peff) +
                                  np.matmul(w_r_b1, b1_t_b2) + w_t_b1)
        polygon1_transf_vert = np.zeros((2,polygons[0].vertices[0].size))
        polygon2_transf_vert = np.zeros((2,polygons[1].vertices[0].size))
        # print("vertex_effector_transf, ", vertex_effector_transf)
        for i in range(0, polygons[0].vertices[0].size):
            b2x = np.array([[polygons[0].vertices[0, i]], [polygons[0].vertices[1, i]]])
            # print("b2x, ", b2x)
            #wRb2 = np.matmul(w_r_b1, b1_r_b2)
            #wTb2 = np.add(np.matmul(w_r_b1, b1_t_b2), w_t_b1)
            #wXw = np.matmul(wRb2, b2x)
            #wX2 = np.add(np.matmul(wRb2, b2x), wTb2)
            w_x = np.matmul(w_r_b1, b2x) + w_t_b1
            polygon1_transf_vert[0, i] = w_x[0, 0]
            polygon1_transf_vert[1, i] = w_x[1, 0]
            # print("polygon1_transf_vert[:,i], ", polygon1_transf_vert[:,i])
        for i in range(0, polygons[1].vertices[0].size):
            b2x = np.array([[polygons[1].vertices[0, i]], [polygons[1].vertices[1, i]]])
            # print("b2x, ", b2x)
            #wRb2 = np.matmul(w_r_b1, b1_r_b2)
            #wTb2 = np.add(np.matmul(w_r_b1, b1_t_b2), w_t_b1)
            #wXw = np.matmul(wRb2, b2x)
            #wX2 = np.add(np.matmul(wRb2, b2x), wTb2)
            w_x = np.matmul(np.matmul(w_r_b1, b1_r_b2), b2x) + np.matmul(w_r_b1, b1_t_b2) + w_t_b1
            polygon2_transf_vert[0, i] = w_x[0, 0]
            polygon2_transf_vert[1, i] = w_x[1, 0]
            # print("polygon2_transf_vert[:,i], ", polygon2_transf_vert[:,i])
        new_pol1 = geometry.Polygon(polygon1_transf_vert)
        new_pol2 = geometry.Polygon(polygon2_transf_vert)
        return vertex_effector_transf, new_pol1, new_pol2

    def plot(self, theta, color):
        """
        This function should use TwoLink.kinematic_map from the previous question together with
        the method Polygon.plot from Homework 1 to plot the manipulator.
        """
        output = self.kinematic_map(theta)
        polygon1_transf = output[1]
        polygon2_transf = output[2]
        polygon1_transf.plot(color)
        polygon2_transf.plot(color)

    def is_collision(self, theta, points):
        """
        For each specified configuration, returns  True if  any of the links of the manipulator
        collides with  any of the points, and  False otherwise. Use the function
        Polygon.is_collision to check if each link of the manipulator is in collision.
        """
        flag_theta = np.zeros((theta[0].size))
        #new_polygons = self.polygons
        #temp_output1 = np.full((theta[0].size), False, dtype=bool)
        #temp_output2 = np.full((theta[0].size), False, dtype=bool)

        for i in range(0,theta[0].size):
            output = self.kinematic_map(theta[:,i])
            polygon1_transf = output[1]
            polygon2_transf = output[2]
            temp_output1 = polygon1_transf.is_collision(points)
            temp_output2 = polygon2_transf.is_collision(points)
            flag_theta[i] = np.any(np.logical_or(temp_output1, temp_output2))

        #pass  # Substitute with your code
        return flag_theta

    def plot_collision(self, theta, points):
        """
        This function should:
     - Use TwoLink.is_collision for determining if each configuration is a collision or not.
     - Use TwoLink.plot to plot the manipulator for all configurations, using a red color when the
    manipulator is in collision, and green otherwise.
     - Plot the points specified by  points as black asterisks.
        """
        flag_theta = self.is_collision(theta, points)
        colors = np.full(theta[0].size, 'g', dtype=str)
        for i in range(0,flag_theta.size):
            if flag_theta[i] == 1:
                colors[i] = 'r'
        for i in range(0,flag_theta.size):
            self.plot(theta[:,i], colors[i])
        plt.scatter(points[0],points[1],c='black', marker="*")

    def jacobian(self, theta, theta_dot):
        """
        Implement the map for the Jacobian of the position of the end effector with respect to the
        joint angles as derived in Question~ q:jacobian-effector.
        """
        pass  # Substitute with your code
        #return vertex_effector_dot




def pol_translate(rot, pol, w_t_b):
    """
    returns indeces for a new polygon, based on 'pol' with rotation matrix 'rot' and offset 'wTb'
    inputs:
    rot = rotation matrix
    pol = polygon to be translated
    wTb = offset from world frame 0,0
    return:
    newvert = new set of vertices which can be used to create a new Polygon
    """
    newxv = np.zeros(np.size(pol[1].vertices[0]))
    newyv = np.zeros(np.size(pol[1].vertices[1]))
    for i in range(0, np.size(pol[1].vertices[0])):
        # print("POL Vertices, ", polygons[1].vertices)
        # print(polygons[1].vertices[0, i])
        # print(polygons[1].vertices[1, i])
        v_lint = np.array([pol[1].vertices[0, i], pol[1].vertices[1, i]])
        # print(v_lint)
        newv = np.matmul(rot, v_lint) + w_t_b
        newxv[i] = newv[0]
        newyv[i] = newv[1]

        # print(newv)

    # print('new vx, ', newxv)
    # print('new vy, ', newyv)

    newvert = np.array([newxv, newyv])
    return newvert


#polygons = TwoLink()


plt.figure()
"""
# convert 45 to radians
th = 45 * np.pi / 180
# create rotation matrix for theta
rot45 = geometry.rot2d(th)
# add offset
wTb = np.array([3, 1])
translation = polTranslate(rot45, polygons[1], wTb)
# create new pol
link2 = geometry.Polygon(translation)

link2.plot('STYLE')
plt.ylim(ymin=0)
plt.xlim(xmin=0)
plt.show()


fig, ax = plt.subplots()
ax.set_aspect('equal')

twolink = TwoLink()
# convert 45 to radians
th = 45 * np.pi / 180
output = twolink.kinematic_map(np.array([[th],[th]]))
twolink.plot(np.array([[th],[th]]),'r')

twolink.plot_collision(np.array([[th, th*2, th*3, th*4, th*5],[th, th, th, th, th]]),
                        np.array([[0.1,1,2,3,4],[0.5,1,2,3,4]]))


#print(output)
outputPolygons = output[1]
pol1 = outputPolygons[0]
pol2 = outputPolygons[1]
print(pol1.vertices, pol2.vertices)
pol1.plot("STYLE")
pol2.plot("STYLE")
"""
plt.show()
