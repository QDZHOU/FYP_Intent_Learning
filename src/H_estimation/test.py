import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from pytope import Polytope
import matplotlib.image as mpimg

from polytope import polytope_estimation_offline, polytope_estimation_MH, polytope_estimation_OR
from ellipsoid import ellipsoid_estimation_offline, ellipsoid_estimation_online
from zonotope import zonotope_estimation,zonotope_estimation_min_area

data_id = "08"
track_id  = 14

background_filepath = f"../../data/{data_id}_background.png"
recordingMeta_filepath = f"../../data/{data_id}_recordingMeta.csv"
tracks_filepath = f"../../data/{data_id}_tracks.csv"
tracksMeta_filepath = f"../../data/{data_id}_tracksMeta.csv"
tracks_data = pd.read_csv(tracks_filepath)
filtered_data = tracks_data[tracks_data['trackId'] == track_id]

xAcc = filtered_data['xAcceleration'].to_numpy()
yAcc = filtered_data['yAcceleration'].to_numpy()
# xAcc = filtered_data['lonAcceleration'].to_numpy()
# yAcc = filtered_data['latAcceleration'].to_numpy()
acc_vals= np.vstack((xAcc,yAcc))
N_Sam = acc_vals.shape[1]

xPos = filtered_data["xCenter"].to_numpy()
yPos = filtered_data["yCenter"].to_numpy()
position_vals = np.vstack((xPos,yPos))

xVel = filtered_data["xVelocity"].to_numpy()
yVel = filtered_data["yVelocity"].to_numpy()
# xVel = filtered_data["lonVelocity"].to_numpy()
# yVel = filtered_data["latVelocity"].to_numpy()
velocity_vals = np.vstack((xVel,yVel))

yaw = filtered_data["heading"].to_numpy()

road_vertices = np.array([
    [270, -78],
    [338, -15],
    [786, -657],
    [857, -592]
])
Drive_Area = Polytope(road_vertices*(0.00814636091724916*12))
Drive_Area_visual = Polytope(road_vertices)
A_road = Drive_Area.A
b_road = Drive_Area.b

image_param = {
  "datasets": {
    "ind": {
      "scale_down_factor": 12,  
      "relevant_areas": { 
        "1": {
          "x_lim": [2000, 11500],  
          "y_lim": [9450, 0]  
        },
        "2": {
          "x_lim": [0, 12500],
          "y_lim": [7400, 0]
        },
        "3": {
          "x_lim": [0, 11500],
          "y_lim": [9365, 0]
        },
        "4": {
          "x_lim": [2700, 15448],
          "y_lim": [9365, 0]
        }
      }
    }
  }
}

x_lim_0 = 2000/12
x_lim_1 = 11500/12
y_lim_0 = 9450/12
y_lim_1 = 0/12

# # Assuming lp_time is already populated with times
# plt.plot(range(1, N_Sam), lp_time, marker='o', linestyle='-', color='b')

# # Add labels and title
# plt.xlabel('Iteration')
# plt.ylabel('Execution Time (seconds)')
# plt.title('Execution Time of ReachableSet Across Iterations')

# # Optionally, add grid and improve visual aesthetics
# plt.grid(True)

# # Show the plot
# plt.show()


# test for offline polytope
def offline_polytope_test():
    lp_time = []
    Init_Param = {"init_acc": acc_vals[:,0].reshape(2,1), "N":5, "T":0.25, "radius": 10, "num_vertices": 4}
    test_polytope = polytope_estimation_offline(Init_Param)

    for i in range(1,N_Sam):
        start_time = time.time() 
        test_polytope.ReachableSet(acc_vals[:,i].reshape(2,1),0,0)
        end_time = time.time()
        lp_time.append(end_time-start_time)
        if i % 50 == 0:
            pass
            #test_polytope.Plot_Polytope()
    
    return lp_time, test_polytope

# test for mh(moving horizon) polytope
def mh_polytope_test(mh_size):
    lp_time = []
    Init_Param = {"init_acc": acc_vals[:,0].reshape(2,1), "N":5, "T":0.25, "radius": 10, "num_vertices": 8, "MH_size": mh_size}
    test_polytope = polytope_estimation_MH(Init_Param)

    SV_x = filtered_data["xCenter"].to_numpy()
    SV_y = filtered_data["yCenter"].to_numpy()
    fig, ax = plt.subplots()
    img   = mpimg.imread(background_filepath)

    for i in range(1,N_Sam):
        start_time = time.time() 
        tmp = test_polytope.ReachableSet(acc_vals[:,i].reshape(2,1),position_vals[:,i].reshape(2,1),velocity_vals[:,i].reshape(2,1))
        end_time = time.time()
        lp_time.append(end_time-start_time)
        if i % 20 == 0 and i > mh_size:
            print(tmp[2].b)
            for i in range(5):
                polytopetmp = Polytope(tmp[i].A, tmp[i].b/0.00814636091724916/12)
                polytopetmp.plot(color = 'b', alpha = 0.3)

    plt.imshow(img,alpha = 0.6, extent=[0, img.shape[1], -img.shape[0], 0])
    plt.plot(SV_x/0.00814636091724916/12, SV_y/0.00814636091724916/12)
    ax.set_xlim([x_lim_0,x_lim_1])
    ax.set_ylim([-y_lim_0,-y_lim_1])
    plt.show()
    
    return lp_time, test_polytope

Param = {
    "init_acc": acc_vals[:,0].reshape(2,1),
    "N": 5, # Reachability Prediction Length
    "T": 0.25, # Sampling Interval
    "radius": 8,
    "num_vertices": 5,
    "l_f": 1.5, # Distance from CoG to Front Axle
    "l_r": 1.5, # Distance from CoG to Rear Axle
    "l_veh": 4.7, # length of vehicle
    "w_veh": 1.8, # width of vehicle
    "DEV": 5,
    "Q1": 1, # steering wheel angle
    "Q2": 1, # longitudinal jerk
    "Q3": 1, # long. velocity error
    "Q4": 5, # long. pos. error
    "Q5": 5, # lat. pos. error
    "Q6": 2, # heading. error
    "Q7": 300,
    "d_min": 2,
    "A_road": A_road,
    "b_road": b_road,
    "v_low": 0,
    "v_up": 13,
    "acc_low": -5,
    "acc_up": 5,
    "delta_low": -0.38,
    "delta_up": 0.38,
    "RefSpeed": 0,
    "RefPos": [-4930,5640,0] # 824,-597->342,-46
}

# test for or(online recursive) polytope
def or_polytope_test():
    lp_time = []
    test_polytope = polytope_estimation_OR(Param)

    SV_x = filtered_data["xCenter"].to_numpy()
    SV_y = filtered_data["yCenter"].to_numpy()
    fig, ax = plt.subplots()
    img   = mpimg.imread(background_filepath)

    for i in range(1,N_Sam):
        start_time = time.time() 
        G,g,tmp = test_polytope.ReachableSet(acc_vals[:,i].reshape(2,1),position_vals[:,i].reshape(2,1),velocity_vals[:,i].reshape(2,1))
        end_time = time.time()
        lp_time.append(end_time-start_time)

        if i % 50 == 0 or i == N_Sam-1:
            for i in range(5):
                polytopetmp = Polytope(tmp[i].A, tmp[i].b/0.00814636091724916/12)
                polytopetmp.plot(color = 'b', alpha = 0.3)
    
    Drive_Area_visual.plot(color = 'r', alpha = 0.3)

    plt.imshow(img,alpha = 0.6, extent=[0, img.shape[1], -img.shape[0], 0])
    plt.plot(SV_x/0.00814636091724916/12, SV_y/0.00814636091724916/12)
    ax.set_xlim([x_lim_0,x_lim_1])
    ax.set_ylim([-y_lim_0,-y_lim_1])
    # ax.set_xlim([0,1500])
    # ax.set_ylim([-1500,0])
    plt.show()
    return lp_time, test_polytope

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
    return lp_time, test_ellipsoid

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
            pass
            #test_ellipsoid.Plot_Ellipsoid()

    return lp_time, test_ellipsoid

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

    return lp_time,test_ellipsoid

# test for zonotope min area
def offline_zonotope_test_min_area():
    lp_time = []
    Init_Param = {"init_acc": acc_vals[:,0].reshape(2,1), "N":5, "T":0.25}
    test_ellipsoid = zonotope_estimation_min_area(Init_Param)

    for i in range(1,N_Sam):
            start_time = time.time() 
            test_ellipsoid.ReachableSet(acc_vals[:,i].reshape(2,1),0,0)
            end_time = time.time()
            lp_time.append(end_time-start_time)
            if i  == N_Sam -1:
                test_ellipsoid.Plot_Zonotope()

    return lp_time,test_ellipsoid

# time complexity for polytope
def plot_time_complexity_polytope():
    lp_time1,tmp= offline_polytope_test()
    lp_time2,tmp= mh_polytope_test(50)
    lp_time3,tmp= or_polytope_test()
    lp_time4,tmp= mh_polytope_test(100)

    # Assuming lp_time is already populated with times
    plt.plot(range(1, N_Sam), lp_time3, linestyle='-', color='b', label="Online Recursive")
    plt.plot(range(1, N_Sam), lp_time1, linestyle='-', color='r', label="Offline")
    plt.plot(range(1, N_Sam), lp_time2, linestyle='-', color='g', label="Moving Horizon(size = 50)")
    plt.plot(range(1, N_Sam), lp_time4, linestyle='-', color='y', label="Moving Horizon(size = 100)")

    # Add labels and title
    plt.xlabel('Number of Samples')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Execution Time of Polytope Estimation')

    # Optionally, add grid and improve visual aesthetics
    plt.grid(True)
    plt.legend()

    # Show the plot
    plt.show()

def plot_H_estimation_polytope():
    tmp,polytope1 = offline_polytope_test()
    tmp,polytope2 = mh_polytope_test(50)
    tmp,polytope3 = or_polytope_test()
    tmp,polytope4 = mh_polytope_test(100)

    plt.figure()
    plt.scatter(polytope1.SV_Acc[0,:],polytope1.SV_Acc[1,:],marker='x',color = 'black',label = "Samples")
    polytope3.U_Hat_Poly.plot(edgecolor = 'b', alpha = 0, linewidth=2,label = "Online Recursive")
    polytope1.U_Hat_Poly.plot(edgecolor = 'r', alpha = 0, linewidth=2,linestyle='dotted',label = "Offline")
    polytope2.U_Hat_Poly.plot(edgecolor = 'g', alpha = 0, linewidth=2, label = "Moving Horizon(size = 50)")
    polytope4.U_Hat_Poly.plot(edgecolor = 'b', alpha = 0, linewidth=2, linestyle='dotted', label = "Moving Horizon(size = 100)")

    points = polytope1.SV_Acc.T
    hull = ConvexHull(points)
    label_added = False 
    for simplex in hull.simplices:
        if not label_added:
            plt.plot(points[simplex, 0], points[simplex, 1], 'r-', label="Convex Hull")
            label_added = True 
        else:
            plt.plot(points[simplex, 0], points[simplex, 1], 'r-')

    plt.xlabel('Long. Acc.')
    plt.ylabel('Lat. Acc.')
    plt.title('Polytope Estimation')

    plt.grid(True)
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_handles_labels = []
    seen_labels = set()
    for handle, label in zip(handles, labels):
        if label not in seen_labels:
            unique_handles_labels.append((handle, label))
            seen_labels.add(label)

    plt.legend(*zip(*unique_handles_labels), loc='best', fontsize=6, labelspacing=0.7)
    plt.show()

# time complexity for ellpsoid
def plot_time_complexity_ellipsoid():
    lp_time1,tmp= offline_ellipsoid_test()
    lp_time2,tmp= online_ellipsoid_test()

    # Assuming lp_time is already populated with times
    plt.plot(range(1, N_Sam), lp_time1, linestyle='-', color='r', label="Offline")
    plt.plot(range(1, N_Sam), lp_time2, linestyle='-', color='b', label="Online")

    # Add labels and title
    plt.xlabel('Number of Samples')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Execution Time of Ellipsoid Estimation')

    # Optionally, add grid and improve visual aesthetics
    plt.grid(True)
    plt.legend()

    # Show the plot
    plt.show()

def plot_H_estimation_ellipsoid():
    tmp,polytope1 = offline_ellipsoid_test()
    tmp,polytope2 = online_ellipsoid_test()

    A1 = polytope1.A_hat
    b1 = polytope1.b_hat
    A2 = polytope2.A_pre
    b2 = polytope2.b_pre


    # Create a grid of points (u, v) in the plane
    u_range = np.linspace(-7, 7, 400)
    v_range = np.linspace(-7, 7, 400)
    u, v = np.meshgrid(u_range, v_range)

    # Vectorize u and v to apply the transformation
    uv = np.vstack([u.flatten(), v.flatten()]).T

    # Apply the transformation A*u + b
    transformed_uv_1 = np.dot(uv, A1.T) + b1.T
    transformed_uv_2 = np.dot(uv, A2.T) + b2.T

    # Compute the quadratic form ||A*u + b||^2
    quad_form_1 = np.sum(transformed_uv_1**2, axis=1)
    quad_form_2 = np.sum(transformed_uv_2**2, axis=1)

    # Reshape to match the grid shape
    quad_form_1 = quad_form_1.reshape(u.shape)
    quad_form_2 = quad_form_2.reshape(u.shape)

    plt.scatter(polytope1.SV_Acc[0,:],polytope1.SV_Acc[1,:],marker='x',color = 'black', label = "Samples")

    points = polytope1.SV_Acc.T
    hull = ConvexHull(points)
    label_added = False 
    for simplex in hull.simplices:
        if not label_added:
            plt.plot(points[simplex, 0], points[simplex, 1], 'r-', label="Convex Hull")
            label_added = True 
        else:
            plt.plot(points[simplex, 0], points[simplex, 1], 'r-')

    # Plot the 2D ellipsoid using contour plot
    plt.contour(u, v, quad_form_1, levels=[1], colors='b',label="Offline")  # Level set at 1
    plt.contour(u, v, quad_form_2, levels=[1], colors='g',label="Online")  # Level set at 1
    plt.plot([], [], color='b', label='Offline')
    plt.plot([], [], color='g', label='Online')
    plt.title("Ellipsoid Estimation")
    plt.xlabel('Long. Acc.')
    plt.ylabel('Lat. Acc.')
    plt.grid(True)
    plt.legend()
    plt.show()

#time complexity for zonotope
def plot_time_complexity_zonotope():
    lp_time1,tmp= offline_zonotope_test_min_area()

    # Assuming lp_time is already populated with times
    plt.plot(range(1, N_Sam), lp_time1, linestyle='-', color='r', label="Zonotope")

    # Add labels and title
    plt.xlabel('Number of Samples')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Execution Time of Zonotope Estimation')

    # Optionally, add grid and improve visual aesthetics
    plt.grid(True)
    plt.legend()

    # Show the plot
    plt.show()

def plot_img():
    fig, ax = plt.subplots()
    img   = mpimg.imread(background_filepath)
    plt.imshow(img,alpha = 1, extent=[0, img.shape[1], -img.shape[0], 0])
    # ax.set_xlim([x_lim_0,x_lim_1])
    # ax.set_ylim([-y_lim_0,-y_lim_1])
    ax.set_xlim([100,1000])
    ax.set_ylim([-800,100])
    plt.show()

plot_img()