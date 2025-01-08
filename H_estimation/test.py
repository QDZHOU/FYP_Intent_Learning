import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

from polytope import polytope_estimation_offline, polytope_estimation_MH, polytope_estimation_OR
from ellipsoid import ellipsoid_estimation_offline, ellipsoid_estimation_online
from zonotope import zonotope_estimation

data_id = "08"
track_id  = 14

background_filepath = f"../data/{data_id}_background.png"
recordingMeta_filepath = f"../data/{data_id}_recordingMeta.csv"
tracks_filepath = f"../data/{data_id}_tracks.csv"
tracksMeta_filepath = f"../data/{data_id}_tracksMeta.csv"
tracks_data = pd.read_csv(tracks_filepath)
filtered_data = tracks_data[tracks_data['trackId'] == track_id]

xAcc = filtered_data['xAcceleration'].to_numpy()
yAcc = filtered_data['yAcceleration'].to_numpy()
acc_vals= np.vstack((xAcc,yAcc))
N_Sam = acc_vals.shape[1]

xPos = filtered_data["xCenter"].to_numpy()
yPos = filtered_data["yCenter"].to_numpy()
position_vals = np.vstack((xPos,yPos))

xVel = filtered_data["xVelocity"].to_numpy()
yVel = filtered_data["yVelocity"].to_numpy()
velocity_vals = np.vstack((xVel,yVel))


# test for offline polytope
def offline_polytope_test():
    lp_time = []
    Init_Param = {"init_acc": acc_vals[:,0].reshape(2,1), "N":5, "T":0.25, "radius": 6, "num_vertices": 8}
    test_polytope = polytope_estimation_offline(Init_Param)

    for i in range(1,N_Sam):
        start_time = time.time() 
        test_polytope.ReachableSet(acc_vals[:,i].reshape(2,1),0,0)
        end_time = time.time()
        lp_time.append(end_time-start_time)
        if i % 50 == 0:
            test_polytope.Plot_Polytope()


    # Assuming lp_time is already populated with times
    plt.plot(range(1, N_Sam), lp_time, marker='o', linestyle='-', color='b')

    # Add labels and title
    plt.xlabel('Iteration')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Execution Time of ReachableSet Across Iterations')

    # Optionally, add grid and improve visual aesthetics
    plt.grid(True)

    # Show the plot
    plt.show()

# test for mh(moving horizon) polytope
def mh_polytope_test():
    lp_time = []
    Init_Param = {"init_acc": acc_vals[:,0].reshape(2,1), "N":5, "T":0.25, "radius": 6, "num_vertices": 8, "MH_size": 50}
    test_polytope = polytope_estimation_MH(Init_Param)

    for i in range(1,N_Sam):
        start_time = time.time() 
        test_polytope.ReachableSet(acc_vals[:,i].reshape(2,1),0,0)
        end_time = time.time()
        lp_time.append(end_time-start_time)
        if i % 50 == 0:
            test_polytope.Plot_Polytope()


    # Assuming lp_time is already populated with times
    plt.plot(range(1, N_Sam), lp_time, marker='o', linestyle='-', color='b')

    # Add labels and title
    plt.xlabel('Iteration')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Execution Time of ReachableSet Across Iterations')

    # Optionally, add grid and improve visual aesthetics
    plt.grid(True)

    # Show the plot
    plt.show()

# test for or(online recursive) polytope
def or_polytope_test():
    lp_time = []
    Init_Param = {"init_acc": acc_vals[:,0].reshape(2,1), "N":5, "T":0.25, "radius": 6, "num_vertices": 8}
    test_polytope = polytope_estimation_OR(Init_Param)

    for i in range(1,N_Sam):
        start_time = time.time() 
        test_polytope.ReachableSet(acc_vals[:,i].reshape(2,1),0,0)
        end_time = time.time()
        lp_time.append(end_time-start_time)
        if i % 50 == 0:
            test_polytope.Plot_Polytope()

    # Assuming lp_time is already populated with times
    plt.plot(range(1, N_Sam), lp_time, marker='o', linestyle='-', color='b')

    # Add labels and title
    plt.xlabel('Iteration')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Execution Time of ReachableSet Across Iterations')

    # Optionally, add grid and improve visual aesthetics
    plt.grid(True)

    # Show the plot
    plt.show()

# test for offline ellipsoid
def offline_ellipsoid_test():
    lp_time = []
    Init_Param = {"init_acc": acc_vals[:,0].reshape(2,1), "N":5, "T":0.25}
    test_ellipsoid = ellipsoid_estimation_offline(Init_Param)

    for i in range(1,N_Sam):
        start_time = time.time() 
        test_ellipsoid.ReachableSet(acc_vals[:,i].reshape(2,1),0,0)
        end_time = time.time()
        lp_time.append(end_time-start_time)
        if i % 50 == 0:
            pass
            #test_ellipsoid.Plot_Ellipsoid()
    # Assuming lp_time is already populated with times
    plt.plot(range(1, N_Sam), lp_time, marker='o', linestyle='-', color='b')

    # Add labels and title
    plt.xlabel('Iteration')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Execution Time of ReachableSet Across Iterations')

    # Optionally, add grid and improve visual aesthetics
    plt.grid(True)

    # Show the plot
    plt.show()

# test for online ellipsoid
def online_ellipsoid_test():
    lp_time = []
    Init_Param = {"init_acc": acc_vals[:,0].reshape(2,1), "N":5, "T":0.25}
    test_ellipsoid = ellipsoid_estimation_online(Init_Param)

    for i in range(1,N_Sam):
            start_time = time.time() 
            test_ellipsoid.ReachableSet(acc_vals[:,i].reshape(2,1),0,0)
            end_time = time.time()
            lp_time.append(end_time-start_time)
            if i % 50 == 0:
                test_ellipsoid.Plot_Ellipsoid()

    # Assuming lp_time is already populated with times
    plt.plot(range(1, N_Sam), lp_time, marker='o', linestyle='-', color='b')

    # Add labels and title
    plt.xlabel('Iteration')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Execution Time of ReachableSet Across Iterations')

    # Optionally, add grid and improve visual aesthetics
    plt.grid(True)

    # Show the plot
    plt.show()

# test for zonotope
def offline_zonotope_test():
    lp_time = []
    Init_Param = {"init_acc": acc_vals[:,0].reshape(2,1), "N":5, "T":0.25}
    test_ellipsoid = zonotope_estimation(Init_Param)

    for i in range(1,N_Sam):
            start_time = time.time() 
            test_ellipsoid.ReachableSet(acc_vals[:,i].reshape(2,1),0,0)
            end_time = time.time()
            lp_time.append(end_time-start_time)
            if i % 50 == 0:
                test_ellipsoid.Plot_Zonotope()

    # Assuming lp_time is already populated with times
    plt.plot(range(1, N_Sam), lp_time, marker='o', linestyle='-', color='b')

    # Add labels and title
    plt.xlabel('Iteration')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Execution Time of ReachableSet Across Iterations')

    # Optionally, add grid and improve visual aesthetics
    plt.grid(True)

    # Show the plot
    plt.show()

offline_zonotope_test()