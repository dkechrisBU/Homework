"""
 Please merge the functions and classes from this file with the same file from the previous
 homework assignment
"""
import numpy as np
import matplotlib.pyplot as plt

import me570_potential
import me570_geometry

def hat2(theta):
    """
    Given a scalar  return the 2x2 skew symmetric matrix corresponding to the
    hat operator
    """
    return np.array([[0, -theta], [theta, 0]])

class TwoLink:
    """ See description from previous homework assignments. """
    def jacobian_matrix(self, theta):
        """
        Compute the matrix representation of the Jacobian of the position of the end effector with
        respect to the joint angles as derived in Question~ q:jacobian-matrix.
        """
        #missing thetadot from the input variables)
        # create function to calculate thetadot?

        theta_dot = np.zeros((2,2))
        theta_dot[0,0] = np.cos(theta[0])
        theta_dot[1,0] = np.sin(theta[0])
        theta_dot[0,1] = -np.sin(theta[1])
        theta_dot[1,1] = np.cos(theta[1])

        #from homework 2
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
        #U = Uattr + α SUM Urep,i
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
        the planner. Note that the output  xPath from Potential.planner will really contain a sequence
        of join angles, rather than a sequence of 2-D points. Plot only every 5th or 10th column of
        xPath (e.g., use  xPath(:,1:5:end)). To avoid clutter, plot a different figure for each start.
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
