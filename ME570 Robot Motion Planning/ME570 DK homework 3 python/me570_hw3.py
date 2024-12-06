"""
Demetrios Kechris
ME570 HW3
"""
import random

import matplotlib.pyplot as plt
import numpy as np

import me570_geometry
import me570_potential


def sphere_test_collision():
    """
    Generates one figure with a sphere (with arbitrary parameters) and
    nb_points=100 random points that are colored according to the sign of
    their distance from the sphere (red for negative, green for positive).
    Generates a second figure in the same way (and the same set of points)
    but flipping the sign of the radius  r of the sphere. For each sampled
    point, plot also the result of the output  pointsSphere.
    """
    center = np.array([[random.uniform(-5,5)],[random.uniform(-5,5)]])
    radius = random.uniform(0,3)
    influence = random.uniform(3,6)
    s1 = me570_geometry.Sphere(center, radius, influence)
    points = np.zeros((2,100))
    points_sign = np.zeros((100), dtype=bool)
    for irandom in range(0,100):
        points[0,irandom] = random.uniform(-10,10)
        points[1,irandom] = random.uniform(-10,10)
    point_dist = s1.distance(points)
    for ipoints in range(0,points[0].size):
        if point_dist[0, ipoints] > 0:
            points_sign[ipoints] = True
        else:
            points_sign[ipoints] = False
    pointsx = points[0]
    pointsy = points[1]
    points_greenx = pointsx[points_sign]
    points_greeny = pointsy[points_sign]
    points_redx = points[0][np.logical_not(points_sign)]
    points_redy = points[1][np.logical_not(points_sign)]

    s1.plot("black")
    plt.scatter(points_greenx, points_greeny, c="g")
    plt.scatter(points_redx, points_redy, c="r")
    plt.show()

    center = np.array([[random.uniform(-5,5)],[random.uniform(-5,5)]])
    radius = random.uniform(-5, 0)
    influence = random.uniform(0, 3)
    s1 = me570_geometry.Sphere(center, radius, influence)
    points = np.zeros((2, 100))
    points_sign = np.zeros((100), dtype=bool)
    for irandom in range(0, 100):
        points[0, irandom] = random.uniform(-10, 10)
        points[1, irandom] = random.uniform(-10, 10)
    point_dist = s1.distance(points)
    for ipoints in range(0, points[0].size):
        if point_dist[0, ipoints] > 0:
            points_sign[ipoints] = True
        else:
            points_sign[ipoints] = False
    pointsx = points[0]
    pointsy = points[1]
    points_greenx = pointsx[points_sign]
    points_greeny = pointsy[points_sign]
    points_redx = points[0][np.logical_not(points_sign)]
    points_redy = points[1][np.logical_not(points_sign)]

    s1.plot("black")
    plt.scatter(points_greenx, points_greeny, c="g")
    plt.scatter(points_redx, points_redy, c="r")
    plt.show()


def clfcbf_control_test_singlesphere():
    """
    Use the provided function Grid.plot_threshold ( ) to visualize the CLF-CBF control field
    for a single filled-in sphere
    """
    # A single sphere whose edge intersects the origin
    world = me570_potential.SphereWorld()
    world.world = [
        me570_geometry.Sphere(center=np.array([[0], [-2]]),
                              radius=2,
                              distance_influence=1)
    ]
    world.x_goal = np.array([[0], [-6]])
    pars = {
        'repulsive_weight': 2,
        'x_goal': np.array([[0], [-6]]),
        'shape': 'conic'
    }

    xx_ticks = np.linspace(-10, 10, 23)
    grid = me570_geometry.Grid(xx_ticks, xx_ticks)

    clfcbf = me570_potential.Clfcbf_Control(world, pars)
    clfcbf.control(np.array([[2], [-3]]))
    plt.figure()
    world.plot()
    grid.plot_threshold(clfcbf.control, 1)


def planner_run_plot_test():
    """
    Show the results of Planner.run_plot for each goal location in
    world.xGoal, and for different interesting combinations of
    potential['repulsive_weight'],  potential['shape'],  epsilon, and
    nb_steps. In each case, for the object of class  Planner should have the
    attribute  function set to  Total.eval, and the attribute  control set
    to the negative of  Total.grad.
    """
    #2.3 report
    sw = me570_potential.SphereWorld()
    potentialC = {
        'shape': 'conic',
        'repulsive_weight': .1,
        'x_goal': np.array([[sw.x_goal[0, 0]], [sw.x_goal[1, 0]]])
    }
    potentialQ = {
        'shape': 'quadratic',
        'repulsive_weight': .1,
        'x_goal': np.array([[sw.x_goal[0, 0]], [sw.x_goal[1, 0]]])
    }
    total = me570_potential.Total(sw,potentialC)

    for igoal in range(0,sw.x_goal[0].size):
        plt.figure()
        potentialC['x_goal'] = np.array([[sw.x_goal[0, igoal]], [sw.x_goal[1, igoal]]])
        plt.subplot(121)
        plt.title('Goal ' + str(igoal+1) + " map")
        sw.plot()
        #too complex?
        """
        for irep in range(1, 10, 2):
            # bring to range(.01,.1,.01)
            potentialA['repulsive_weight'] = irep/100
            for iepsilon in range(1, 10, 1):
                # bring to range(.001,.01,.001):
                epsilon = iepsilon/1000
                for isteps in range(500,3001,500):
                    nbsteps = isteps
                    total = me570_potential.Total(sw, potentialA)
                    plan = me570_potential.Planner(total.eval, total.neg_grad, epsilon, nbsteps)
                    for istart in range(0,sw.x_start[0].size):
                        xstart = np.array([sw.x_start[0, istart], sw.x_start[1, istart]])
                        plan.run(xstart)
        """
        potentialC['repulsive_weight'] = .05
        epsilon = .005
        #nbsteps = 100
        #nbsteps = 1000
        #nbsteps = 5000
        nbsteps = 6500

        total = me570_potential.Total(sw, potentialC)
        plan = me570_potential.Planner(total.eval, total.neg_grad, epsilon, nbsteps)
        for istart in range(0, sw.x_start[0].size):
            xstart = np.array([sw.x_start[0, istart], sw.x_start[1, istart]])
            #plan.run(xstart)
            #plt.show()
        #plt.title('Distance from goal ' + str(igoal+1))
        #plt.show()

    for igoal in range(0,sw.x_goal[0].size):
        plt.figure()
        sw.plot()
        potentialQ['x_goal'] = np.array([[sw.x_goal[0, igoal]], [sw.x_goal[1, igoal]]])
        plt.subplot(121)
        plt.title('Goal ' + str(igoal+1) + " map")
        #too complex?
        """
        for irep in range(1, 10, 2):
            # bring to range(.01,.1,.01)
            potentialA['repulsive_weight'] = irep/100
            for iepsilon in range(1, 10, 1):
                # bring to range(.001,.01,.001):
                epsilon = iepsilon/1000
                for isteps in range(500,3001,500):
                    nbsteps = isteps
                    total = me570_potential.Total(sw, potentialA)
                    plan = me570_potential.Planner(total.eval, total.neg_grad, epsilon, nbsteps)
                    for istart in range(0,sw.x_start[0].size):
                        xstart = np.array([sw.x_start[0, istart], sw.x_start[1, istart]])
                        plan.run(xstart)
        """
        potentialQ['repulsive_weight'] = .05
        epsilon = .005
        nbsteps = 100
        nbsteps = 2000
        potentialQ['repulsive_weight'] = .01
        potentialQ['repulsive_weight'] = .005
        potentialQ['repulsive_weight'] = .0025
        potentialQ['repulsive_weight'] = .001
        epsilon = .0025
        total = me570_potential.Total(sw, potentialQ)
        plan = me570_potential.Planner(total.eval, total.neg_grad, epsilon, nbsteps)
        for istart in range(0, sw.x_start[0].size):
            xstart = np.array([sw.x_start[0, istart], sw.x_start[1, istart]])
            plan.run(xstart)
            #plt.show()
        plt.title('Distance from goal ' + str(igoal+1))
        plt.show()

def clfcbf_run_plot_test():
    """
    Use the function Planner.run_plot to run the planner based on the
    CLF-CBF framework, and show the results for one combination of
    repulsive_weight and  epsilon that makes the planner work reliably.
    """
    # 3.5 report
    sw = me570_potential.SphereWorld()
    xx_ticks = np.linspace(-11, 11, 51)
    grid = me570_geometry.Grid(xx_ticks, xx_ticks)

    for igoal in range(0, sw.x_goal[0].size):
        world = sw.world
        plt.figure()
        plt.subplot(121)
        sw.plot()
        plt.title('Goal ' + str(igoal+1) + " map")
        potentialC = {
            'shape': 'conic',
            #'shape': 'quadratic',
            'repulsive_weight': .005,
            'x_goal': np.array([[sw.x_goal[0, 0]], [sw.x_goal[1, 0]]])
        }
        epsilon = .05
        nbsteps = 2000
        total = me570_potential.Total(sw, potentialC)
        clfcbf_test = me570_potential.Clfcbf_Control(sw, potentialC)
        plan = me570_potential.Planner(clfcbf_test.function, clfcbf_test.control, epsilon, nbsteps)
        for istart in range(0, sw.x_start[0].size):
            #print("istart, ", istart)
            xstart = np.array([sw.x_start[0, istart], sw.x_start[1, istart]])
            plan.run(xstart)
        plt.title('Distance from goal ' + str(igoal + 1))
        plt.show()


        plt.figure()
        plt.subplot(121)
        sw.plot()
        plan = me570_potential.Planner(total.eval, total.neg_grad, epsilon, nbsteps)
        plt.title('Goal ' + str(igoal+1) + " map")
        #planner plot
        for istart in range(0, sw.x_start[0].size):
            xstart = np.array([sw.x_start[0, istart], sw.x_start[1, istart]])
            plan.run(xstart)
        plt.title('Distance from goal ' + str(igoal+1))
        plt.show()
    pass  # Substitute with your code

if __name__ == "__main__":
    xx_ticks = np.linspace(-11, 11, 51)
    grid = me570_geometry.Grid(xx_ticks, xx_ticks)
    sw = me570_potential.SphereWorld()
    potentialA = {
        'shape': 'conic',
        'repulsive_weight': .05,
        'x_goal': np.array([[sw.x_goal[0, 0]], [sw.x_goal[1, 0]]])
    }
    #sphere_test_collision()
    #planner_run_plot_test()

    # 3.3
    #clfcbf_test = me570_potential.Clfcbf_Control(sw, potentialA)
    #plt.figure()
    #grid.plot_threshold(clfcbf_control_test_singlesphere)
    #plt.title('3.3 CLFCBF test_singlephere')
    #plt.show()

    clfcbf_control_test_singlesphere()
    plt.title('3.3 CLFCBF test_singlephere rep_w 1')
    plt.show()

    #3.5
    clfcbf_run_plot_test()
