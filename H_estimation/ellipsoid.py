import casadi as ca
import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.linalg import sqrtm, inv

class ellipsoid_estimation_offline():
    def __init__(self, Param):
        self.SV_Acc = Param["init_acc"]
        self.N = Param["N"] # Reachability Prediction Length
        self.T = Param["T"] # Sampling Interval
        self.nu = self.SV_Acc.shape[0]
        self.N_Sam = self.SV_Acc.shape[1]

    def ReachableSet(self,SV_Acc_new, SV_Pos_new, SV_Vel_new):
        self.A_hat, self.b_hat = self.Ellipsoid(SV_Acc_new)
        # occupancy projection

    def opti_SDP(self):
        nu = self.nu
        N_Sam = self.N_Sam

        A = cp.Variable((nu, nu), PSD=True)
        b = cp.Variable((nu, 1))
        constraints = [A - 1e-8*np.eye(nu) >> 0]
        for i in range(N_Sam):
            constraints += [cp.norm(A@cp.reshape(self.SV_Acc[:,i], (2, 1),order='F') + b, 2) <= 1]
        objective = cp.Maximize(cp.log_det(A))

        problem = cp.Problem(objective, constraints)
        problem.solve(solver = cp.SCS) # verbose=True

        return A.value, b.value

    def Ellipsoid(self, SV_Acc_new):
        self.SV_Acc = np.hstack((self.SV_Acc,SV_Acc_new))
        self.N_Sam = self.SV_Acc.shape[1]
        A_hat, b_hat = self.opti_SDP()

        return A_hat,b_hat

    def Plot_Ellipsoid(self):
        A = self.A_hat
        b = self.b_hat

        # Create a grid of points (u, v) in the plane
        u_range = np.linspace(-7, 7, 400)
        v_range = np.linspace(-7, 7, 400)
        u, v = np.meshgrid(u_range, v_range)

        # Vectorize u and v to apply the transformation
        uv = np.vstack([u.flatten(), v.flatten()]).T

        # Apply the transformation A*u + b
        transformed_uv = np.dot(uv, A.T) + b.T

        # Compute the quadratic form ||A*u + b||^2
        quad_form = np.sum(transformed_uv**2, axis=1)

        # Reshape to match the grid shape
        quad_form = quad_form.reshape(u.shape)

        points = self.SV_Acc.T
        hull = ConvexHull(points)
        for simplex in hull.simplices:
            plt.plot(points[simplex, 0], points[simplex, 1], 'r-')

        # Plot the 2D ellipsoid using contour plot
        plt.contour(u, v, quad_form, levels=[1], colors='b')  # Level set at 1
        plt.scatter(self.SV_Acc[0,:],self.SV_Acc[1,:],marker='x',color = 'black')
        plt.title("Ellipsoid Visualisation")
        plt.xlabel('Long Acc')
        plt.ylabel('Lat Acc')
        plt.axis('equal')
        plt.grid(True)
        plt.show()

    def Return(self):
        pass

class ellipsoid_estimation_online1():
    def __init__(self, Param):
        self.SV_Acc = Param["init_acc"]
        self.N = Param["N"] # Reachability Prediction Length
        self.T = Param["T"] # Sampling Interval
        
        A_init = np.array([[1, 0], [0, 1]])
        b_init = np.array([[0], [0]]) 
        c_init = b_init.T@b_init-1

        self.nu = self.SV_Acc.shape[0]
        self.A_pre, self.b_pre = self.opti_SDP_lemma(A_init,b_init,self.SV_Acc)

    def ReachableSet(self,SV_Acc_new, SV_Pos_new, SV_Vel_new):
        self.Ellipsoid(SV_Acc_new)
        # occupancy projection

    def opti_SDP_lemma(self, A_pre, b_pre,SV_Acc_new):
        nu = self.nu

        A = cp.Variable((nu, nu), PSD=True)
        b = cp.Variable((nu, 1))
        constraints = [A - 1e-8*np.eye(nu) >> 0]

        constraints += [A.T@A - A_pre.T@A_pre >> 0]
        constraints += [cp.norm(A@cp.reshape(SV_Acc_new, (2, 1),order='F') + b, 2) <= 1]
        constraints += [cp.norm(A @ (-inv(A_pre)@b_pre) + b, 2) <= 1]
        objective = cp.Maximize(cp.log_det(A))

        problem = cp.Problem(objective, constraints)
        problem.solve(solver = cp.MOSEK) # verbose=True

        return A.value, b.value

    def Ellipsoid(self, SV_Acc_new):
        self.A_pre, self.b_pre = self.opti_SDP_lemma(self.A_pre,self.b_pre,SV_Acc_new)

    def Plot_Ellipsoid(self):
        A = self.A_pre
        b = self.b_pre
        u_range = np.linspace(-7, 7, 400)
        v_range = np.linspace(-7, 7, 400)
        u, v = np.meshgrid(u_range, v_range)

        # Vectorize u and v to apply the transformation
        uv = np.vstack([u.flatten(), v.flatten()]).T

        # Apply the transformation A*u + b
        transformed_uv = np.dot(uv, A.T) + b.T

        # Compute the quadratic form ||A*u + b||^2
        quad_form = np.sum(transformed_uv**2, axis=1)

        # Reshape to match the grid shape
        quad_form = quad_form.reshape(u.shape)

        # Plot the 2D ellipsoid using contour plot
        plt.contour(u, v, quad_form, levels=[1], colors='b')  # Level set at 1
        #plt.scatter(random_vals[0,:i],random_vals[1,:i],marker='x',color = 'black')
        plt.title("Ellipsoid Visualisation")
        plt.xlabel('Long Acc')
        plt.ylabel('Lat Acc')
        plt.axis('equal')
        plt.grid(True)
        plt.show()

    def Return(self):
        pass

class ellipsoid_estimation_online():
    def __init__(self, Param):
        self.SV_Acc = Param["init_acc"]
        self.N = Param["N"] # Reachability Prediction Length
        self.T = Param["T"] # Sampling Interval
        
        A_init = np.array([[1, 0], [0, 1]])
        b_init = np.array([[0], [0]]) 
        c_init = b_init.T@b_init-1

        self.nu = self.SV_Acc.shape[0]
        self.N_Sam = self.SV_Acc.shape[1]
        self.B_pre, self.b_tilde_pre = self.opti_SDP_lemma(A_init,b_init,c_init,self.SV_Acc)
        self.A_pre = sqrtm(self.B_pre)
        self.b_pre = np.linalg.inv(self.A_pre)@self.b_tilde_pre

    def ReachableSet(self,SV_Acc_new, SV_Pos_new, SV_Vel_new):
        self.Ellipsoid(SV_Acc_new)
        # occupancy projection

    def opti_SDP_lemma(self, A_last, b_last, c_last, u_last):
        nu = self.nu

        B         = cp.Variable((nu, nu), PSD=True) # B = A**2
        tau       = cp.Variable( )
        tau_u     = cp.Variable( )
        b_tilde   = cp.Variable((nu, 1)) # b_tilde = A@b

        A_u = np.eye(nu)
        b_u = -u_last.reshape(2, 1)
        c_u = u_last.T@u_last - 1e-8

        zero_block = np.zeros((nu, nu))
        
        M  = cp.bmat([[B - tau * A_last, b_tilde - tau * b_last, zero_block],
                        [(b_tilde - tau * b_last).T, cp.reshape(-1 - tau * c_last, (1, 1),order='F'), b_tilde.T],
                        [zero_block, b_tilde, -B]])
        Mu = cp.bmat([[B - tau_u * A_u, b_tilde - tau_u * b_u, zero_block],
                        [(b_tilde - tau_u * b_u).T, cp.reshape(-1 - tau_u * c_u, (1, 1),order='F'), b_tilde.T],
                        [zero_block, b_tilde, -B]])
        
        constraints = [tau >= 0, tau_u >= 0, -M >> 0, -Mu >> 0, B - 1e-8*np.eye(nu) >> 0]  
        objective   = cp.Maximize(cp.log_det(B))
        problem     = cp.Problem(objective, constraints)
        problem.solve(solver = cp.MOSEK)

        return B.value, b_tilde.value

    def Ellipsoid(self, SV_Acc_new):
        A_last = self.A_pre.T@self.A_pre
        b_last = self.A_pre.T@self.b_pre
        c_last = self.b_pre.T@self.b_pre-1
        self.B_pre, self.b_tilde_pre = self.opti_SDP_lemma(A_last,b_last,c_last,SV_Acc_new)
        self.A_pre = sqrtm(self.B_pre)
        self.b_pre = np.linalg.inv(self.A_pre)@self.b_tilde_pre

    def Plot_Ellipsoid(self):
        A = self.A_pre
        b = self.b_pre
        u_range = np.linspace(-7, 7, 400)
        v_range = np.linspace(-7, 7, 400)
        u, v = np.meshgrid(u_range, v_range)

        # Vectorize u and v to apply the transformation
        uv = np.vstack([u.flatten(), v.flatten()]).T

        # Apply the transformation A*u + b
        transformed_uv = np.dot(uv, A.T) + b.T

        # Compute the quadratic form ||A*u + b||^2
        quad_form = np.sum(transformed_uv**2, axis=1)

        # Reshape to match the grid shape
        quad_form = quad_form.reshape(u.shape)

        # Plot the 2D ellipsoid using contour plot
        plt.contour(u, v, quad_form, levels=[1], colors='b')  # Level set at 1
        #plt.scatter(random_vals[0,:i],random_vals[1,:i],marker='x',color = 'black')
        plt.title("Ellipsoid Visualisation")
        plt.xlabel('Long Acc')
        plt.ylabel('Lat Acc')
        plt.axis('equal')
        plt.grid(True)
        plt.show()

    def Return(self):
        pass