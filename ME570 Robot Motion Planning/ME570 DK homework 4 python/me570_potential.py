"""
Classes to define potential and potential planer for the sphere world
"""
import math

import numpy as np
from matplotlib import pyplot as plt
from scipy import io as scio

import me570_geometry
import me570_qp


class SphereWorld:
    """ Class for loading and plotting a 2-D sphereworld. """
    def __init__(self):
        """
        Load the sphere world from the provided file sphereworld.mat, and sets the
    following attributes:
     -  world: a  nb_spheres list of  Sphere objects defining all the spherical obstacles in the
    sphere world.
     -  x_start, a [2 x nb_start] array of initial starting locations (one for each column).
     -  x_goal, a [2 x nb_goal] vector containing the coordinates of different goal locations (one
    for each column).
        """
        data = scio.loadmat('sphereworld.mat')

        self.world = []
        for sphere_args in np.reshape(data['world'], (-1, )):
            sphere_args[1] = sphere_args[1].item()
            sphere_args[2] = sphere_args[2].item()
            self.world.append(me570_geometry.Sphere(*sphere_args))

        self.x_goal = data['xGoal']
        self.x_start = data['xStart']
        self.theta_start = data['thetaStart']

    def plot(self, axes=None):
        """
        Uses Sphere.plot to draw the spherical obstacles together with a  * marker at the goal
        location.
        """

        if axes is None:
            axes = plt.gca()

        for sphere in self.world:
            sphere.plot('black')

        plt.scatter(self.x_goal[0, :], self.x_goal[1, :], c='g', marker='*')

        plt.xlim([-11, 11])
        plt.ylim([-11, 11])
        plt.axis('equal')


class RepulsiveSphere:
    """ Repulsive potential for a sphere """
    def __init__(self, sphere):
        """
        Save the arguments to internal attributes
        """
        self.sphere = sphere

    def eval(self, x_eval):
        """s
        Evaluate the repulsive potential from  sphere at the location x= x_eval.
        The function returns the repulsive potential as given by      (  eq:repulsive  ).
        """
        distance = self.sphere.distance(x_eval)

        distance_influence = self.sphere.distance_influence
        # #print("distance, ", distance)
        # #print("distance influence, ", distance_influence)
        if distance > distance_influence:
            u_rep = 0.0
        elif distance_influence > distance > 0:
            u_rep = ((distance**-1 - distance_influence**-1)**2) / 2.
            u_rep = u_rep.item()
        else:
            u_rep = math.nan
        return u_rep

    def grad(self, x_eval):
        """
        Compute the gradient of U_ rep for a single sphere, as given by (eq:repulsive-gradient).
        """
        #This function must use the outputs of sphere_distanceSphere.
        #looks like it breaks down into x y componenets
        dist_grad = self.sphere.distance_grad(x_eval)
        dist = self.sphere.distance(x_eval)
        u_rep = self.eval(x_eval)
        if np.isnan(u_rep):
            #grad_u_rep = np.array([[np.nan],[np.nan]])
            grad_u_rep = np.array([[0],[0]])
        elif dist > self.sphere.distance_influence:
            grad_u_rep = np.array([[0.0],[0.0]])
        else:
            grad_u_rep = -1 * (math.sqrt(2*u_rep)) * (1 / pow(dist, 2)) * dist_grad
            #grad_u_rep = -1 * (1/dist - 1/self.sphere.distance_influence) * (1 / pow(dist, 2)) * dist_grad
        return grad_u_rep


class Attractive:
    """ Repulsive potential for a sphere """
    def __init__(self, potential):
        """
        Save the arguments to internal attributes
        """
        self.potential = potential

    def eval(self, x_eval):
        """
        Evaluate the attractive potential  U_ attr at a point  xEval with respect to a goal location
    potential.xGoal given by the formula: If  potential.shape is equal to  'conic', use p=1. If
    potential.shape is equal to  'quadratic', use p=2.
        """
        x_goal = self.potential['x_goal']
        shape = self.potential['shape']
        if shape == 'conic':
            expo = 1
        else:
            expo = 2
        u_attr = np.linalg.norm(x_eval - x_goal)**expo
        return u_attr

    def grad(self, x_eval):
        """
        Evaluate the gradient of the attractive potential  U_ attr at a point  xEval. The gradient
        is given by the formula If  potential['shape'] is equal to 'conic', use p=1; if it is
        equal to 'quadratic', use p=2.
        """
        x_goal = self.potential['x_goal']
        shape = self.potential['shape']
        if shape == 'conic':
            pval=1
        elif shape == 'quadratic':
            pval=2
        else:
            pval=0
        u_inv = np.linalg.norm(x_eval - x_goal) ** (pval - 2)
        grad_u_attr = pval * u_inv * (x_eval - x_goal)
        #grad_u_attr = pval * pow(np.linalg.norm(x_eval - x_goal), (pval - 2)) * (x_eval - x_goal)

        return grad_u_attr


class Total:
    """ Combines attractive and repulsive potentials """
    def __init__(self, world, potential):
        """
        Save the arguments to internal attributes
        """
        self.world = world
        self.potential = potential


    def eval(self, x_eval):
        """
        Compute the function U=U_attr+a*iU_rep,i, where a is given by the variable
    potential.repulsiveWeight
        """
        alpha = self.potential['repulsive_weight']
        attractive_sphere = Attractive(self.potential)
        u_att = attractive_sphere.eval(x_eval)
        u_rep = 0.0
        # #print("x_eval, ", x_eval)
        for isphere in self.world.world:
            u_rep = u_rep + RepulsiveSphere(isphere).eval(x_eval)
        u_eval = u_att + alpha * u_rep
        return u_eval

    def grad(self, x_eval):
        """
        Compute the gradient of the total potential,  U=U_ attr+a*U_rep,i, where a is given by
        the variable  potential.repulsiveWeight
        """
        u_rep_g = 0.0
        attractive_sphere = Attractive(self.potential)
        #spheres = np.zeros(self.sphereworld.world[0].size)
        #for isphere in range(0,self.sphereworld.world.size):
        #    spheres[0] = me570_geometry.Sphere(self.sphereworld.world[0,isphere],
        #                self.sphereworld.world[1,isphere],self.sphereworld.world[2,isphere])
        for isphere in self.world.world:
            u_rep_g = (u_rep_g + self.potential['repulsive_weight'] *
                       RepulsiveSphere(isphere).grad(x_eval))
        if np.isnan(u_rep_g[0]):
            grad_u_eval = attractive_sphere.grad(x_eval)
        else:
            grad_u_eval = attractive_sphere.grad(x_eval) + u_rep_g
         #print("attractive_sphere.grad(x_eval), ", attractive_sphere.grad(x_eval))
         #print("u_rep_g, ", u_rep_g)
         #print(grad_u_eval)
        return grad_u_eval

    def neg_grad(self, x_eval):
        """
        Compute the gradient of the total potential,  U=U_ attr+a*U_rep,i, where a is given by
        the variable  potential.repulsiveWeight
        """
        u_rep_g = 0.0
        attractive_sphere = Attractive(self.potential)
        #spheres = np.zeros(self.sphereworld.world[0].size)
        #for isphere in range(0,self.sphereworld.world.size):
        #    spheres[0] = me570_geometry.Sphere(self.sphereworld.world[0,isphere],
        #                self.sphereworld.world[1,isphere],self.sphereworld.world[2,isphere])
        for isphere in self.world.world:
            u_rep_g = (u_rep_g + self.potential['repulsive_weight'] *
                       RepulsiveSphere(isphere).grad(x_eval))
        u_att = attractive_sphere.grad(x_eval)
        grad_u_eval = u_att + u_rep_g
        grad_u_eval = -1 * grad_u_eval
         #print("attractive_sphere.grad(x_eval), ", attractive_sphere.grad(x_eval))
         #print("u_rep_g, ", u_rep_g)
         #print(grad_u_eval)
        return grad_u_eval


class Planner:
    """
    A class implementing a generic potential planner and plot the results.
    """
    def __init__(self, function, control, epsilon, nb_steps):
        """
        Save the arguments to internal attributes
        """
        self.function = function
        self.control = control
        self.epsilon = epsilon
        self.nb_steps = nb_steps

    def run(self, x_start):
        """
        This function uses a given function (given by  control) to implement a
        generic potential-based planner with step size  epsilon, and evaluates
        the cost along the returned path. The planner must stop when either the
        number of steps given by  nb_stepsis reached, or when the norm of the
        vector given by  control is less than 5 10^-3 (equivalently,  5e-3).
        """
        #may need to change code to add nan
        #x_coord = np.array([x_start[0]])
        #y_coord = np.array([x_start[1]])
        x_path = np.zeros((2,self.nb_steps))
        u_path = np.zeros((1,self.nb_steps))
        x_path[0, 0] = x_start[0]
        x_path[1, 0] = x_start[1]
        x_test = np.array([[x_path[0, 0]], [x_path[1, 0]]])
        u_path[0,0] = self.function(x_test)
        isteps = 1
        control_norm = 10
        while isteps < self.nb_steps:
            if control_norm > .005:
                x_test = np.array([[x_path[0, isteps-1]], [x_path[1, isteps-1]]])
                control_current = self.control(x_test)
                # #print(control_current.flatten())
                control_norm = np.linalg.norm(control_current.flatten())
                #control_norm = math.sqrt(pow(control_current[0],2) + pow(control_current[1],2))
                # #print("control_current, ", control_current)
                # #print("control_norm, ", control_norm)
                next_step_x = x_path[0, isteps-1] + self.epsilon * control_current[0]
                next_step_y = x_path[1, isteps-1] + self.epsilon * control_current[1]
                # #print("next steps: ", next_step_x," ", next_step_y)
                x_coord = np.append(x_path[0], next_step_x)
                #y_coord = np.append(x_path[1], next_step_y)
                #x_path = np.array([np.append(x_path[0], next_step_x), np.append(x_path[1],
                                                                                #next_step_y)])
                x_path[0, isteps] = next_step_x
                x_path[1, isteps] = next_step_y
                #u_path = np.append(u_path,self.function(x_test))
                u_path[0, isteps] = self.function(x_test)
            else:
                x_path[0, isteps] = np.nan
                x_path[1, isteps] = np.nan
                u_path[0, isteps] = np.nan
            isteps = isteps + 1

        #plot outputs
        plt.subplot(121)
        plt.plot(x_path[0], x_path[1])
        plt.subplot(122)
        plt.semilogy(u_path[0])

        #print("x_path, ", x_path)
        #print("u_path, ", u_path)
        return x_path, u_path


class Clfcbf_Control:
    """
    A class implementing a CLF-CBF-based control framework.
    """
    def __init__(self, world, potential):
        """
        Save the arguments to internal attributes, and create an attribute
        attractive with an object of class  Attractive using the argument
        potential.
        """
        self.world = world
        self.potential = potential
        self.attractive = Attractive(potential)

    def function(self, x_eval):
        """
        Evaluate the CLF (i.e.,  self.attractive.eval()) at the given input.
        """
        return self.attractive.eval(x_eval)

    def control(self, x_eval):
        """
        Compute u^* according to      (  eq:clfcbf-qp  ).
        """
        a_barrier = np.zeros((len(self.world.world), 2))
        b_barrier = np.zeros((len(self.world.world), 1))
        for iobstacle in range(0,len(self.world.world)):
            #temp_a0 = self.world.world[iobstacle].center[0] - x_eval[0]
            # temp_a1 = self.world.world[iobstacle].center[1] - x_eval[1]

            temp_a0 = x_eval[0] - self.world.world[iobstacle].center[0]
            temp_a1 = x_eval[1] - self.world.world[iobstacle].center[1]
            temp_a = self.world.world[iobstacle].distance_grad(x_eval)
            #temp_a = np.array([temp_a0,temp_a1])
            #temp_ag = temp_a * self.world.world[iobstacle].distance_grad(x_eval)
            norma = np.linalg.norm(x_eval - self.world.world[iobstacle].center)
            if norma == 0:
                a_barrier[iobstacle, 0] = 0
                a_barrier[iobstacle, 1] = 0
            else:
                if self.world.world[iobstacle].radius < 0:
                    a_barrier[iobstacle, 0] = temp_a[0]
                    a_barrier[iobstacle, 1] = temp_a[1]
                #a_barrier[iobstacle, 0] = temp_ag[0]
                #a_barrier[iobstacle, 1] = temp_ag[1]
                else:
                    a_barrier[iobstacle, 0] = -1*temp_a[0]
                    a_barrier[iobstacle, 1] = -1*temp_a[1]
                
                #if self.world.world[iobstacle].radius < 0:
                #a_barrier[iobstacle, 0] = -1 * a_barrier[iobstacle, 0]
                #a_barrier[iobstacle, 1] = -1 * a_barrier[iobstacle, 1]
    # changing b to -b, and -uref to uref, and uopt to -u opt "works" for 1sphere but not multiple
            temp_b = self.world.world[iobstacle].distance(x_eval)
            b_barrier[iobstacle,0] = -1 * temp_b[0] * self.potential['repulsive_weight']
        #print("a, ", a_barrier)
        #print("b, ", b_barrier)
        u_ref = -1*self.attractive.grad(x_eval)
        #print("u, ", u_ref)
        u_opt = me570_qp.qp_supervisor(a_barrier, b_barrier, u_ref)

        #new test
        """a_barrier = np.zeros((len(self.world.world), 2))
        b_barrier = np.zeros((len(self.world.world), 1))
        total = Total(self.world,self.potential)
        for iobstacle in range(0, len(self.world.world)):
            # temp_a0 = self.world.world[iobstacle].center[0] - x_eval[0]
            # temp_a1 = self.world.world[iobstacle].center[1] - x_eval[1]

            temp_a0 = x_eval[0] - self.world.world[iobstacle].center[0]
            temp_a1 = x_eval[1] - self.world.world[iobstacle].center[1]
            temp_a = total.grad(x_eval)
            # temp_a = np.array([temp_a0,temp_a1])
            temp_ag = temp_a * self.world.world[iobstacle].distance_grad(x_eval)
            norma = np.linalg.norm(x_eval - self.world.world[iobstacle].center)
            if norma == 0:
                a_barrier[iobstacle, 0] = 0
                a_barrier[iobstacle, 1] = 0
            else:
                #a_barrier[iobstacle, 0] = temp_a0[0] / norma
                #a_barrier[iobstacle, 1] = temp_a1[0] / norma
                a_barrier[iobstacle, 0] = temp_ag[0]
                a_barrier[iobstacle, 1] = temp_ag[1]
                #if self.world.world[iobstacle].radius < 0:
                #a_barrier[iobstacle, 0] = -1 * a_barrier[iobstacle, 0]
                #a_barrier[iobstacle, 1] = -1 * a_barrier[iobstacle, 1]
            temp_b = self.world.world[iobstacle].distance(x_eval)
            b_barrier[iobstacle, 0] = temp_b[0] * self.potential['repulsive_weight']
        print("a, ", a_barrier)
        print("b, ", b_barrier)
        u_ref = -1 * self.attractive.grad(x_eval)
        print("u, ", u_ref)
        u_opt = me570_qp.qp_supervisor(a_barrier, b_barrier, u_ref)"""
        return u_opt


if __name__ == "__main__":
    xx_ticks = np.linspace(-11, 11, 51)
    grid = me570_geometry.Grid(xx_ticks, xx_ticks)
    sw = SphereWorld()
    potentialA = {
        'shape': 'conic',
        'repulsive_weight': .05,
        'x_goal': np.array([[sw.x_goal[0, 0]], [sw.x_goal[1, 0]]])
    }

    #2.1 report
    # arrows point away because grad is pointing towards the surface of the obstacle
    # arrows will only appear where d(x) < d(influence) of the obstacle
    """smallworld = SphereWorld()
    smallworld.world = smallworld.world[0:2]
    sphere0 = RepulsiveSphere(smallworld.world[0])
    sphere1 = RepulsiveSphere(smallworld.world[1])
    plt.figure()
    sphere0.sphere.plot("black")
    grid.plot_threshold(sphere0.grad) # give U rep grad
    plt.title('2.1 Repulsive sphere0 grad')
    plt.show()
    plt.figure()
    sphere0.sphere.radius = sphere0.sphere.radius * -1
    sphere0.sphere.plot("black")
    grid.plot_threshold(sphere0.grad) # give U rep grad
    plt.title('2.1 Repulsive sphere0 invert-r grad')
    plt.show()

    plt.figure()
    sphere1.sphere.plot("black")
    grid.plot_threshold(sphere1.grad)
    plt.title('2.1 Repulsive sphere1 grad')
    plt.show()
    plt.figure()
    sphere1.sphere.radius = sphere1.sphere.radius * -1
    sphere1.sphere.plot("black")
    grid.plot_threshold(sphere1.grad)
    plt.title('2.1 Repulsive sphere1 invert-r  grad')
    plt.show()"""



    #2.1 optional
    """#conic
    smallworld = SphereWorld()
    smallworld.world = smallworld.world[0:2]
    obstacle1 = RepulsiveSphere(smallworld.world[1])


    #attractive potential graphs
    plt.figure()
    smallworld.plot()
    attractive_sphere = Attractive(potentialA)
    grid.plot_threshold(attractive_sphere.eval) # give U att
    plt.title('Attractive eval')
    plt.show()
    plt.figure()
    smallworld.plot()
    grid.plot_threshold(attractive_sphere.grad) # give U att grad
    plt.title('Attractive grad')
    plt.show()

    #repulsive potential graphs
    plt.figure()
    smallworld.plot()
    grid.plot_threshold(obstacle.eval)
    plt.title('Repulsive eval')
    plt.show()
    plt.figure()
    smallworld.plot()
    grid.plot_threshold(obstacle.grad)
    plt.title('Repulsive grad')
    plt.show()

    #total potential graphs
    total_smallworld = Total(smallworld,potentialA)
    plt.figure()
    smallworld.plot()
    grid.plot_threshold(total_smallworld.eval)
    plt.title('Total eval')
    plt.show()
    plt.figure()
    smallworld.plot()
    grid.plot_threshold(total_smallworld.grad)
    plt.title('Total grad')
    plt.show()"""

    """
    #testing planner
    plt.figure()
    total = Total(sw,potentialA)
    # #print(total.grad(np.array([[0.0],[0.0]])))
    # #print(total.neg_grad(np.array([[0.0],[0.0]])))
    plan = Planner(total.eval,total.neg_grad,.01,1000)
    xstart = np.array([sw.x_start[0, 0], sw.x_start[1, 0]])
    plt.subplot(121)
    sw.plot()
    epsilon = .01
    # nbsteps = 100
    # nbsteps = 1000
    nbsteps = 2500
    total = Total(sw, potentialA)
    plan = Planner(total.eval, total.neg_grad, epsilon, nbsteps)
    for istart in range(0, sw.x_start[0].size):
        xstart = np.array([sw.x_start[0, istart], sw.x_start[1, istart]])
        plan.run(xstart)
        #plt.show()
    plt.title('Distance from goal ' + str(0 + 1))
    #plan.run(xstart)
    plt.show()
    """

    #report 2.4
    """plt.figure()
    # repulsive weight + shape = conic
    potentialC = {
        'shape': 'conic',
        'repulsive_weight': .05,
        'x_goal': np.array([[sw.x_goal[0, 0]], [sw.x_goal[1, 0]]])
    }
    totalC = Total(sw, potentialC)
    grid.plot_threshold(totalC.eval)
    plt.title('2.4 rep_w .05 conic eval')
    plt.show()

    plt.figure()
    grid.plot_threshold(totalC.grad)
    plt.title('2.4 rep_w .05 conic grad')
    plt.show()

    potentialC = {
        'shape': 'conic',
        'repulsive_weight': .001,
        'x_goal': np.array([[sw.x_goal[0, 0]], [sw.x_goal[1, 0]]])
    }
    plt.figure()
    totalC = Total(sw, potentialC)
    grid.plot_threshold(totalC.eval)
    plt.title('2.4 rep_w .001 conic eval')
    plt.show()

    plt.figure()
    grid.plot_threshold(totalC.grad)
    plt.title('2.4 rep_w .001 conic grad')
    plt.show()

    # repulsive weight + shape = conic
    potentialQ = {
        'shape': 'quadratic',
        'repulsive_weight': .05,
        'x_goal': np.array([[sw.x_goal[0, 0]], [sw.x_goal[1, 0]]])
    }
    plt.figure()
    totalQ = Total(sw, potentialQ)
    grid.plot_threshold(totalQ.eval, 10)
    plt.title('2.4 rep_w .05 quad eval')
    plt.show()

    plt.figure()
    grid.plot_threshold(totalQ.grad, 10)
    plt.title('2.4 rep_w .05 quad grad')
    plt.show()

    # repulsive weight + shape = conic
    potentialQ = {
        'shape': 'quadratic',
        'repulsive_weight': .001,
        'x_goal': np.array([[sw.x_goal[0, 0]], [sw.x_goal[1, 0]]])
    }
    plt.figure()
    totalQ = Total(sw, potentialQ)
    grid.plot_threshold(totalQ.eval, 10)
    plt.title('2.4 rep_w .001 quad eval')
    plt.show()

    plt.figure()
    grid.plot_threshold(totalQ.grad, 10)
    plt.title('2.4 rep_w .001 quad grad')
    plt.show()"""

    #3.1
    clfcbf_test = Clfcbf_Control(sw,potentialA)
    clfcbf_test.control(np.array([[0], [0]]))

