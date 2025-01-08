import casadi as ca
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pytope import Polytope
from scipy.spatial import ConvexHull

class polytope_estimation_offline():
    def __init__(self, Param):
        self.SV_Acc = Param["init_acc"]
        self.N = Param["N"] # Reachability Prediction Length
        self.T = Param["T"] # Sampling Interval
        self.radius = Param["radius"]
        self.num_vertices = Param["num_vertices"]

        self.U_SV_Poly = self.Generate_Polytope(self.radius,self.num_vertices)
        self.H = self.U_SV_Poly.A 
        self.h = self.U_SV_Poly.b 
        self.nv = np.size(self.H, 0)
        self.nu = np.size(self.H, 1)
        for i in range(self.nv):
            self.H[i,:] /= self.h[i]
        self.h = np.ones_like(self.h)
        self.N_Sam = self.SV_Acc.shape[1]


    def ReachableSet(self,SV_Acc_new, SV_Pos_new, SV_Vel_new):
        self.U_Hat_Poly = self.Offline(SV_Acc_new)
        #self.Plot_Polytope()

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


    def Generate_Polytope(self, radius, num_vertices):
        angles = np.linspace(0, 2 * np.pi, num_vertices, endpoint=False)  # Divide circle into 8 parts
        x_coords = radius * np.cos(angles)
        y_coords = radius * np.sin(angles)

        vertices = np.column_stack((x_coords, y_coords))
        U_SV_Poly = Polytope(vertices)

        return U_SV_Poly
    

    def Offline_LP(self):
        H = self.H
        nv = self.nv
        nu = self.nu
        N_Sam = self.N_Sam

        opti = ca.Opti()

        rho = opti.variable()
        theta = opti.variable(nv,1)
        y = opti.variable(nu,1)

        samples = opti.parameter(nu, N_Sam)

        opti.minimize(rho + np.ones((1, nv))@theta)

        for i in range(N_Sam):
            opti.subject_to(H @ samples[:, i]-H @ y <= theta)

        opti.subject_to(H @ y <= (1 - rho)@np.ones((nv,1)))
        opti.subject_to(opti.bounded(0, theta, 1))
        opti.subject_to(opti.bounded(0, rho, 1))
        opti.subject_to(theta <= rho@np.ones((nv,1)))

        opts = {"ipopt.print_level": 0, "print_time": False}
        opti.solver('ipopt', opts)

        return opti.to_function('f', [samples], [theta, y, rho]) 
    

    def Offline(self,SV_Acc_new):

        self.SV_Acc = np.hstack((self.SV_Acc,SV_Acc_new))
        self.N_Sam = self.SV_Acc.shape[1]
        self.LP = self.Offline_LP()

        theta_hat, y_hat, rho_hat = self.LP(self.SV_Acc)
        U_Hat_Poly = Polytope(self.H, theta_hat)+y_hat

        return U_Hat_Poly
    

    def Plot_Polytope(self):
        plt.figure()
        self.U_SV_Poly.plot(color = 'b', alpha = 0.3)

        self.U_Hat_Poly.plot(color = 'r', alpha = 0.3)

        #sample points
        plt.scatter(self.SV_Acc[0,:],self.SV_Acc[1,:],marker='x',color = 'black')

        # y
        # y_s_coodinate = np.array(y_hat)
        # plt.scatter(y_s_coodinate[0], y_s_coodinate[1], marker='x',color='red')

        #convexhull
        points = self.SV_Acc.T
        hull = ConvexHull(points)
        for simplex in hull.simplices:
            plt.plot(points[simplex, 0], points[simplex, 1], 'r-')

        plt.gca().set_aspect('equal', adjustable='box')
        plt.title("Polytope Visualization")
        plt.xlabel("Long Acc")
        plt.ylabel("Lat Acc")
        plt.grid(True)
        plt.show()

    
    def Return(self):
        pass

class polytope_estimation_MH():
    def __init__(self, Param):
        self.SV_Acc = Param["init_acc"]
        self.N = Param["N"] # Reachability Prediction Length
        self.T = Param["T"] # Sampling Interval
        self.radius = Param["radius"]
        self.num_vertices = Param["num_vertices"]
        self.MH_size = Param["MH_size"]

        self.U_SV_Poly = self.Generate_Polytope(self.radius,self.num_vertices)
        self.U_Hat_Poly = self.U_SV_Poly
        self.H = self.U_SV_Poly.A 
        self.h = self.U_SV_Poly.b 
        self.nv = np.size(self.H, 0)
        self.nu = np.size(self.H, 1)
        for i in range(self.nv):
            self.H[i,:] /= self.h[i]
        self.h = np.ones_like(self.h)
        self.N_Sam = self.SV_Acc.shape[1]

        self.LP = self.MH_LP()
        

    def ReachableSet(self,SV_Acc_new, SV_Pos_new, SV_Vel_new):
        self.U_Hat_Poly = self.MH(SV_Acc_new)
        #self.Plot_Polytope()

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


    def Generate_Polytope(self, radius, num_vertices):
        angles = np.linspace(0, 2 * np.pi, num_vertices, endpoint=False)  # Divide circle into 8 parts
        x_coords = radius * np.cos(angles)
        y_coords = radius * np.sin(angles)

        vertices = np.column_stack((x_coords, y_coords))
        U_SV_Poly = Polytope(vertices)

        return U_SV_Poly
    

    def MH_LP(self):
        H = self.H
        nv = self.nv
        nu = self.nu
        N_Sam = self.MH_size

        opti = ca.Opti()

        rho = opti.variable()
        theta = opti.variable(nv,1)
        y = opti.variable(nu,1)

        samples = opti.parameter(nu, N_Sam)

        opti.minimize(rho + np.ones((1, nv))@theta)

        for i in range(N_Sam):
            opti.subject_to(H @ samples[:, i]-H @ y <= theta)

        opti.subject_to(H @ y <= (1 - rho)@np.ones((nv,1)))
        opti.subject_to(opti.bounded(0, theta, 1))
        opti.subject_to(opti.bounded(0, rho, 1))
        opti.subject_to(theta <= rho@np.ones((nv,1)))

        opts = {"ipopt.print_level": 0, "print_time": False}
        opti.solver('ipopt', opts)

        return opti.to_function('f', [samples], [theta, y, rho]) 
    

    def MH(self,SV_Acc_new):
        self.SV_Acc = np.hstack((self.SV_Acc,SV_Acc_new))
        if self.SV_Acc.shape[1] > self.MH_size:
            self.SV_Acc = self.SV_Acc[:,1:]
            theta_hat, y_hat, rho_hat = self.LP(self.SV_Acc)
            U_Hat_Poly = Polytope(self.H, theta_hat)+y_hat
            return U_Hat_Poly
        return self.U_Hat_Poly
    

    def Plot_Polytope(self):
        plt.figure()
        self.U_SV_Poly.plot(color = 'b', alpha = 0.3)

        self.U_Hat_Poly.plot(color = 'r', alpha = 0.3)

        #sample points
        plt.scatter(self.SV_Acc[0,:],self.SV_Acc[1,:],marker='x',color = 'black')

        # y
        # y_s_coodinate = np.array(y_hat)
        # plt.scatter(y_s_coodinate[0], y_s_coodinate[1], marker='x',color='red')

        #convexhull
        points = self.SV_Acc.T
        hull = ConvexHull(points)
        for simplex in hull.simplices:
            plt.plot(points[simplex, 0], points[simplex, 1], 'r-')

        plt.gca().set_aspect('equal', adjustable='box')
        plt.title("Polytope Visualization")
        plt.xlabel("Long Acc")
        plt.ylabel("Lat Acc")
        plt.grid(True)
        plt.show()

    
    def Return(self):
        pass

class polytope_estimation_OR():
    def __init__(self, Param):
        #self.nu = Param["nu"]
        self.SV_Acc = Param["init_acc"]
        self.N = Param["N"] # Reachability Prediction Length
        self.T = Param["T"] # Sampling Interval
        self.radius = Param["radius"]
        self.num_vertices = Param["num_vertices"]

        self.U_SV_Poly = self.Generate_Polytope(self.radius,self.num_vertices)
        self.H = self.U_SV_Poly.A 
        self.h = self.U_SV_Poly.b 
        self.nv = np.size(self.H, 0)
        self.nu = np.size(self.H, 1)
        for i in range(self.nv):
            self.H[i,:] /= self.h[i]
        self.h = np.ones_like(self.h)
        self.N_Sam = self.SV_Acc.shape[1]

        self.LP = self.OR_LP()
        self.theta_pre,self.y_pre,self.rho_pre = self.LP(np.array([[0],[0]]),np.zeros((self.nu,1)),0.1*np.ones((self.nv,1)))



    def ReachableSet(self,SV_Acc_new, SV_Pos_new, SV_Vel_new):
        self.U_Hat_Poly = self.OR(SV_Acc_new)

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


    def Generate_Polytope(self, radius, num_vertices):
        angles = np.linspace(0, 2 * np.pi, num_vertices, endpoint=False)  # Divide circle into 8 parts
        x_coords = radius * np.cos(angles)
        y_coords = radius * np.sin(angles)

        vertices = np.column_stack((x_coords, y_coords))
        U_SV_Poly = Polytope(vertices)

        return U_SV_Poly
    

    def OR_LP(self):
        H = self.H
        nv = self.nv
        nu = self.nu

        opti = ca.Opti()

        rho = opti.variable()
        theta = opti.variable(nv,1)
        y = opti.variable(nu,1)

        samples = opti.parameter(nu, 1)
        y_pre = opti.parameter(nu,1)
        theta_pre = opti.parameter(nv,1)

        opti.minimize(rho + np.ones((1, nv))@theta)

        opti.subject_to(H @ samples-H @ y <= theta)
        opti.subject_to(H @ y_pre + theta_pre <= H @ y + theta)
        opti.subject_to(H @ y <= (1 - rho)@np.ones((nv,1)))
        opti.subject_to(opti.bounded(0, theta, 1))
        opti.subject_to(opti.bounded(0, rho, 1))
        opti.subject_to(theta <= rho@np.ones((nv,1)))

        opts = {"ipopt.print_level": 0, "print_time": False}
        opti.solver('ipopt', opts)

        return opti.to_function('f', [samples, y_pre, theta_pre], [theta, y, rho]) 
    

    def OR(self,SV_Acc_new):
        self.theta_pre,self.y_pre,self.rho_pre = self.LP(SV_Acc_new,self.y_pre,self.theta_pre)
        U_Hat_Poly = Polytope(self.H, self.theta_pre)+self.y_pre

        return U_Hat_Poly
    

    def Plot_Polytope(self):
        plt.figure()
        self.U_SV_Poly.plot(color = 'b', alpha = 0.3)

        self.U_Hat_Poly.plot(color = 'r', alpha = 0.3)

        #sample points
        plt.scatter(self.SV_Acc[0,:],self.SV_Acc[1,:],marker='x',color = 'black')

        # y
        # y_s_coodinate = np.array(y_hat)
        # plt.scatter(y_s_coodinate[0], y_s_coodinate[1], marker='x',color='red')

        #convexhull
        # points = self.SV_Acc.T
        # hull = ConvexHull(points)
        # for simplex in hull.simplices:
        #     plt.plot(points[simplex, 0], points[simplex, 1], 'r-')

        plt.gca().set_aspect('equal', adjustable='box')
        plt.title("Polytope Visualization")
        plt.xlabel("Long Acc")
        plt.ylabel("Lat Acc")
        plt.grid(True)
        plt.show()

    
    def Return(self):
        pass