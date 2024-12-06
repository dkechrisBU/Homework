"""
Combine the classes below with the file me570_robot.py from previous assignments
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy import io as scio
import me570_geometry
import me570_graph
import me570_potential


class TwoLinkGraph:
    """
    A class for finding a path for the two-link manipulator among given obstacle points using a grid
discretization and  A^*.
    """
    def __init__(self):
        """
        define Grid and Graph
        """
        self.TwoLink = TwoLink()
        self.grid = load_free_space_grid()
        self.graph = me570_graph.grid2graph(self.grid)

    def load_free_space_graph(self):
        """
        The function performs the following steps
         - Calls the method load_free_space_grid.
         - Calls grid2graph.
         - Stores the resulting  graph object of class  Grid as an internal attribute.
        """
        self.grid = load_free_space_grid()
        self.graph = me570_graph.grid2graph(self.grid)

    def plot(self):
        """
        Use the method Graph.plot to visualize the contents of the attribute  graph.
        """
        plt.figure()
        self.graph.plot()
        plt.show()

    def search_start_goal(self, theta_start, theta_goal):
        """
        Use the method Graph.search to search a path in the graph stored in  graph.
        """
        theta_path = 0
        # convert theta to x
        km_start = self.TwoLink.kinematic_map(theta_start)
        km_goal = self.TwoLink.kinematic_map(theta_goal)
        print(me570_geometry.rot2d(theta_start[0]))
        x_start = km_start[0]
        x_goal = km_goal[0]
        #x_goal = me570_geometry.rot2d(theta_goal[0]) + me570_geometry.rot2d(theta_goal[1])
        x_path = self.graph.search_start_goal(x_start, x_goal)
        theta_path = np.zeros((2, x_path[0].size))
        # convert x_path to theta_path
        #for i_path in range(0,x_path[0].size):
        #    theta_path[0][i] =
        return x_path


def load_free_space_grid():
    """
Loads the contents of the file ! twolink_freeSpace_data.mat
    """
    test_data = scio.loadmat('twolink_freeSpace_data.mat')
    test_data = test_data['grid'][0][0]
    grid = me570_geometry.Grid(test_data[0], test_data[1])
    grid.fun_evalued = test_data[2]
    return grid


# HW3
class TwoLink:
    """ See description from previous homework assignments. """
    def kinematic_map(self, theta):
        """
        The function returns the coordinate of the end effector  plus the
        vertices of the links  all transformed according to  _1  _2
        """

        #  Rotation matrices
        rotation_w_b1 = me570_geometry .rot2d(theta[0, 0])
        rotation_b1_b2 = me570_geometry .rot2d(theta[1, 0])
        rotation_w_b2 = rotation_w_b1 @ rotation_b1_b2

        #  Translation matrix
        translation_b1_b2 = np.vstack((5, 0))
        translation_w_b2 = rotation_w_b1 @ translation_b1_b2

        #  Transform end effector from B₂ to W
        p_eff_b2 = np.vstack((5, 0))
        vertex_effector_transf = rotation_w_b2 @ p_eff_b2 + translation_w_b2

        #  Transform polygon 1 from B₁ to W
        polygon1_vertices_b1 = polygons[0].vertices
        polygon1_transf = me570_geometry .Polygon(rotation_w_b1 @ polygon1_vertices_b1)

        #  Transform polygon 2 from B₂ to W
        polygon2_vertices_b2 = polygons[1].vertices
        polygon2_transf = me570_geometry .Polygon(rotation_w_b2 @ polygon2_vertices_b2 +
                                     translation_w_b2)
        return vertex_effector_transf, polygon1_transf, polygon2_transf


    def jacobian_matrix(self, theta):
        """
        Compute the matrix representation of the Jacobian of the position of the end effector with
        respect to the joint angles as derived in Question~ q:jacobian-matrix.
        """
        # missing thetadot from the input variables)
        # create function to calculate thetadot?

        theta_dot = np.zeros((2, 2))
        theta_dot[0, 0] = np.cos(theta[0])
        theta_dot[1, 0] = np.sin(theta[0])
        theta_dot[0, 1] = -np.sin(theta[1])
        theta_dot[1, 1] = np.cos(theta[1])

        # from homework 2
        link_lengths = [5, 5]
        offset = [
            np.vstack([link_lengths[0], 0]),
            np.vstack([link_lengths[1], 0])
        ]
        vertex_effector_dot = np.zeros(theta.shape)

        for i in range(theta.shape[1]):
            theta_i = [theta[0, i], theta[1, i]]
            hat = [hat2(theta_dot[0, 0]), hat2(theta_dot[1, 0])]
            rot = [me570_geometry.rot2d(theta_i[0]), me570_geometry.rot2d(theta_i[1])]

            vertex_effector_dot[:, [i]] = hat[0] @ rot[0] @ \
                                          (rot[1] @ offset[0] + offset[1]) \
                                          + rot[0] @ hat[1] @ rot[1] @ offset[0]
        """
        #  theta_dot = np.gradient(theta)
        radius = 5
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        vertex_effector_dot = hat2(theta) * rot * radius
        """
        return vertex_effector_dot

    def animate(self, theta):
        """
        Draw the two-link manipulator for each column in theta with a small pause between each
        drawing operation
        """
        theta_steps = theta.shape[1]
        for i_theta in range(0, theta_steps, 15):
            self.plot(theta[:, [i_theta]], 'k')


    def plot(self, theta, color):
        """
        This function should use TwoLink kinematic_map from the previous question together with
        the method Polygon plot from Homework 1 to plot the manipulator
        """
        [_, polygon1_transf, polygon2_transf] = self.kinematic_map(theta)
        polygon1_transf.plot(color)
        polygon2_transf.plot(color)


class TwoLinkPotential:
    """ Combines attractive and repulsive potentials """

    def __init__(self, world, potential):
        """
        Save the arguments to internal attributes
        """
        self.world = world
        self.potential = potential

    def eval(self, theta_eval):
        """
        Compute the potential U pulled back through the kinematic map of the two-link manipulator,
        i.e., U(Wp_eff(theta)), where U is defined as in Question~q:total-potential, and
        Wp_ eff(theta) is the position of the end effector in the world frame as a function of
        the joint angles   = _1\\ _2.
        """
        # U = Uattr + α SUM Urep,i
        alpha = self.potential['repulsive_weight']
        attractive_sphere = me570_potential.Attractive(self.potential)
        u_att = attractive_sphere.eval(TwoLink.jacobian_matrix(theta_eval))
        u_rep = 0.0
        # #print("x_eval, ", x_eval)
        for isphere in self.world.world:
            u_rep = (u_rep + me570_potential.RepulsiveSphere(isphere).
                     eval(TwoLink.jacobian_matrix(theta_eval)))
        u_eval_theta = u_att + alpha * u_rep
        return u_eval_theta

    def grad(self, theta_eval):
        """
        Compute the gradient of the potential U pulled back through the kinematic map of the
        two-link manipulator, i.e., grad U(  Wp_ eff(  )).
        """
        u_rep_g = 0.0
        attractive_sphere = me570_potential.Attractive(self.potential)
        # spheres = np.zeros(self.sphereworld.world[0].size)
        # for isphere in range(0,self.sphereworld.world.size):
        #    spheres[0] = me570_geometry.Sphere(self.sphereworld.world[0,isphere],
        #                self.sphereworld.world[1,isphere],self.sphereworld.world[2,isphere])
        for isphere in self.world.world:
            u_rep_g = (u_rep_g + self.potential['repulsive_weight'] *
                       me570_potential.RepulsiveSphere(isphere).
                       grad(TwoLink.jacobian_matrix(theta_eval)))
        if np.isnan(u_rep_g[0]):
            grad_u_eval_theta = attractive_sphere.grad(TwoLink.jacobian_matrix(theta_eval))
        else:
            grad_u_eval_theta = (attractive_sphere.grad(TwoLink.jacobian_matrix(theta_eval)) +
                                 u_rep_g)
        # print("attractive_sphere.grad(x_eval), ", attractive_sphere.grad(x_eval))
        # print("u_rep_g, ", u_rep_g)
        # print(grad_u_eval)
        # return grad_u_eval_theta

    def run_plot(self, epsilon, nb_steps):
        """
        This function performs the same steps as Planner.run_plot in
        Question~q:potentialPlannerTest, except for the following:
     - In step  it:grad-handle:  planner_parameters['U'] should be set to  @twolink_total, and
        planner_parameters['control'] to the negative of  @twolink_totalGrad.
     - In step  it:grad-handle: Use the contents of the variable  thetaStart instead of  xStart to
        initialize the planner, and use only the second goal  x_goal[:,1].
     - In step  it:plot-plan: Use Twolink.plotAnimate to plot a decimated version of the results of
        the planner. Note that the output  xPath from Potential.planner will really contain a
        sequence of join angles, rather than a sequence of 2-D points. Plot only every 5th or 10th
        column of xPath (e.g., use  xPath(:,1:5:end)). To avoid clutter, plot a different figure
        for each start.
        """
        sphere_world = me570_potential.SphereWorld()

        nb_starts = sphere_world.theta_start.shape[1]

        planner = me570_potential.Planner(function=self.eval,
                                          control=self.grad,
                                          epsilon=epsilon,
                                          nb_steps=nb_steps)

        two_link = TwoLink()

        for start in range(0, nb_starts):
            # Run the planner
            theta_start = sphere_world.theta_start[:, [start]]
            theta_path, u_path = planner.run(theta_start)

            # Plots
            _, axes = plt.subplots(ncols=2)
            axes[0].set_aspect('equal', adjustable='box')
            plt.sca(axes[0])
            sphere_world.plot()
            two_link.animate(theta_path)
            axes[1].plot(u_path.T)


def hat2(theta):
    """
    Given a scalar  return the 2x2 skew symmetric matrix corresponding to the
    hat operator
    """
    return np.array([[0, -theta], [theta, 0]])


def polygons_add_x_reflection(vertices):
    """
    Given a sequence of vertices  adds other vertices by reflection
    along the x_axis
    """
    vertices = np.hstack([vertices, np.fliplr(np.diag([1, -1]).dot(vertices))])
    return vertices


def polygons_generate():
    """
    Generate the polygons to be used for the two link manipulator
    """
    vertices1 = np.array([[0, 5], [-1.11, -0.511]])
    vertices1 = polygons_add_x_reflection(vertices1)
    vertices2 = np.array([[0, 3.97, 4.17, 5.38, 5.61, 4.5],
                          [-0.47, -0.5, -0.75, -0.97, -0.5, -0.313]])
    vertices2 = polygons_add_x_reflection(vertices2)
    return (me570_geometry.Polygon(vertices1), me570_geometry.Polygon(vertices2))


polygons = polygons_generate()

if __name__ == "__main__":
    tlg = TwoLinkGraph()
    tlg.load_free_space_graph()
    #tlg.plot()
    tlg.search_start_goal(np.array([[0], [0]]), np.array([[1], [1]]))
    tlg.plot()

    theta_start_easy = np.array([[0.76],[0.12]])
    theta_goal_easy = np.array([[0.76],[6.00]])
    twolink = TwoLink()
    tlg.search_start_goal(theta_start_easy,theta_goal_easy)
    twolink.plotAnimate()

