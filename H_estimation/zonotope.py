import casadi as ca
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

class zonotope_estimation():
    def __init__(self, Param):
        self.SV_Acc = Param["init_acc"]
        self.N = Param["N"] # Reachability Prediction Length
        self.T = Param["T"] # Sampling Interval

    def ReachableSet(self,SV_Acc_new, SV_Pos_new, SV_Vel_new):
        self.stacked_points = self.Offline(SV_Acc_new)

        # A_SV = np.array([[1, self.T, 0, 0], [0, 1, 0, 0], [0, 0, 1, self.T], [0, 0, 0, 1]])
        # B_SV = np.array([[0.5*self.T**2, 0], [self.T, 0], [0, 0.5*self.T**2], [0, self.T]])
        # BU = B_SV*self.U_Hat_Poly
        # Reachable_Set = list()
        # Occupancy_SV = list()
        # x_t = np.array([SV_Pos_new[0], SV_Vel_new[0], SV_Pos_new[1], SV_Vel_new[1]])
        # Reachable_Set.append(x_t)

        # for t in range(1,self.N+1):
        #     if t == 1:
        #         reachable_set_t = (A_SV@Reachable_Set[t - 1] + BU) 
        #     else:
        #         reachable_set_t = (A_SV*Reachable_Set[t - 1] + BU) 
            
        #     vertex = reachable_set_t.V
        #     vertex_xy = np.delete(vertex, [1, 3], axis = 1)
        #     occupancy_SV_t = Polytope(vertex_xy) 
        #     occupancy_SV_t.minimize_V_rep( )
        #     temp_poly   = occupancy_SV_t

        #     Occupancy_SV.append(temp_poly)
        #     Reachable_Set.append(reachable_set_t)

        # return Occupancy_SV

    def opti_circle(self,P):
        center = ca.SX.sym('center', 2)
        radius = ca.SX.sym('radius') 
        
        constraints = []
        N = P.shape[1]
        for i in range(N):
            p1 = P[:, i]
            p2 = P[:, (i + 1) % N]

            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]]) 
            normal = normal / np.linalg.norm(normal)

            distance = ca.dot(normal, center - p1)

            constraints.append(distance - radius)
        g = ca.vertcat(*constraints)

        objective = -radius 

        nlp = {'x': ca.vertcat(center, radius), 'f': objective, 'g': g}
        solver = ca.nlpsol('solver', 'ipopt', nlp)
        x0 = [np.mean(P[0, :]), np.mean(P[1, :]), 0.1]
        lbx = [-ca.inf, -ca.inf, 0]  # x, y, r >= 0
        ubx = [ca.inf, ca.inf, ca.inf]

        lbg = [0] * N 
        ubg = [ca.inf] * N 

        sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

        optimal_center = np.array(sol['x'][0:2].full()).flatten()
        optimal_radius = float(sol['x'][2])

        return optimal_center,optimal_radius
    

    def Offline(self,SV_Acc_new):
        self.SV_Acc = np.hstack((self.SV_Acc,SV_Acc_new))
        self.N_Sam = self.SV_Acc.shape[1]
        if self.N_Sam > 4:
            points = self.SV_Acc
            points = points.T
            hull = ConvexHull(points)
            convex_points = points[hull.vertices]
            convex_points = convex_points.T 
            #N_Convex = convex_points.shape[1]
            self.center, self.radius = self.opti_circle(convex_points)
            rotated_x = -convex_points[0,:]+2*self.center[0]
            rotated_y = -convex_points[1,:]+2*self.center[1]
            rotated_convex_points = np.array([rotated_x,rotated_y])
            stacked_points = np.hstack((convex_points, rotated_convex_points))

            return stacked_points
    

    def Plot_Zonotope(self):
        fig, ax = plt.subplots()
        #sample points
        #plt.scatter(random_vals[0,:],random_vals[1,:],marker='x',color = 'black')
        #plt.scatter(rotated_convex_points[0,:],rotated_convex_points[1,:],marker='x',color = 'blue')

        circle = plt.Circle(self.center, self.radius, color='blue', fill=False)

        # Add the circle to the plot
        ax.add_artist(circle)

        #convexhull
        points = self.SV_Acc.T
        hull = ConvexHull(points)
        for simplex in hull.simplices:
            plt.plot(points[simplex, 0], points[simplex, 1], 'r-')

        points = self.stacked_points.T
        hull = ConvexHull(points)
        for simplex in hull.simplices:
            plt.plot(points[simplex, 0], points[simplex, 1], 'g-')

        plt.gca().set_aspect('equal', adjustable='box')
        plt.title("Polytope Visualization")
        plt.xlabel("Long Acc")
        plt.ylabel("Lat Acc")
        plt.grid(True)
        plt.show()

    
    def Return(self):
        pass