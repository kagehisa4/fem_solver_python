#import libraries

import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

#Define physical constants of the beam Area of cross-section (A), Lengh (L),
# Shear Modulus (G), Polar moment of inertia (J), Density (rho), Joint_stiffness (k_j)
A,L,G,J,rho,k_j = .0025, 1, 81e9, 2* (5/100)**4/12, 7850,0

no_of_elements = input('Enter number of elements: ')

n_elements =  2 * int(no_of_elements)

L_e = L / n_elements

total_dof = n_elements+1
if k_j !=0:
    total_dof +=1


#Mass matrix or hte element
Mass_element = (rho*J*L_e/6) * np.array([[2,1],[1,2]])

#Torsion stiffness matrix for the elements
Stiffness_element = (G*J/L_e) * np.array([[1, -1], [-1, 1]])

#stiffness of joint_ elements
Stiffness_joint = k_j * np.array([[1,-1],[-1,1]])



#Assembly of global  matrices

M = np.zeros((total_dof, total_dof))
K = np.zeros((total_dof, total_dof))


joint_node = int(.5 * n_elements) #this is always at centre.
elements = []

# check for the joint present:
for i in range (n_elements):
    if k_j !=0 and i >= joint_node:#This it to create joint as a separate element.
        node_a = i+1
        node_b = node_a +1
    else:
        node_a = i
        node_b = i+1

    elements.append((node_a, node_b))

# global indices:
#Appending into matix:


if k_j != 0:
    print('Joint present')
    K [joint_node, joint_node] += Stiffness_joint [0,0]
    K [joint_node, joint_node+1] +=  Stiffness_joint [0,1]
    K [joint_node+1, joint_node] += Stiffness_joint [1,0]
    K [joint_node+1,joint_node+1] += Stiffness_joint [1,1]

for (node_a, node_b) in elements:
    M [node_a, node_a] += Mass_element [0,0]
    M [node_a, node_b] += Mass_element [0,1]
    M [node_b, node_a] += Mass_element [1,0]
    M [node_b, node_b] += Mass_element [1,1]

    K [node_a, node_a] += Stiffness_element [0,0]
    K [node_a, node_b] +=  Stiffness_element [0,1]
    K [node_b, node_a] += Stiffness_element [1,0]
    K [node_b, node_b] += Stiffness_element [1,1]



fixed_dof = [0]

free = np.setdiff1d(np.arange(total_dof), fixed_dof)

'''for i in range (len(free)):
    print(free[i])
    print('\n')'''

M = M [np.ix_(free,free)]
K = K [np.ix_(free,free)]

eigen_values, eigen_vectors = eigh(K,M)

#reconstruction  of full modes
#This creates a rectangular matrix. Modes are columns = number of elements
# Rows are for each node including the free node.
full_modes = np.zeros((total_dof, eigen_vectors.shape[1]))
full_modes[free, :] = eigen_vectors

freq_ang = np.sqrt(np.abs(eigen_values))
freq = freq_ang/(np.pi*2)
idx = np.argsort(freq)
freq = freq[idx]
full_modes = full_modes[:, idx]#arrange eigen_vecs columns to match sorted frequency.


'''print(len(freq))
print('number of modes:')'''

#plot the curve

x = np.linspace(0, L, n_elements+1)
if k_j != 0:
    x = np.insert(x, joint_node+1, x[joint_node]) #plot across the length of the beam
plt.figure(figsize=(6,4))
count = 0
modes_needed = int(input('Enter number of modes: '))
for mode in range (len(freq)):
    tx = full_modes[:, mode]
    tx = tx / np.max(np.abs(tx)) #np.abs (maximum absolute value)
    # left beam segment
    if k_j!=0:
            j= joint_node
            plt.plot(x[:j+1], tx[:j+1], label=f"{freq[mode]:.1f} Hz")

    # right beam segment
            plt.plot(x[j+1:], tx[j+1:], color=plt.gca().lines[-1].get_color())

    else:
            plt.plot(x, tx, label=f"{freq[mode]:.1f} Hz")
    count += 1
    if count == modes_needed:
        break

   # plot first 5 torsion modes

plt.xlabel("Beam Length (m)")
plt.ylabel("Twist θ") #italics
plt.title(f"Torsion Mode Shapes (Mesh: {n_elements} elements)")
if k_j != 0:

    x_joint = x[joint_node]

    plt.axvline(x=x_joint, color='black', linestyle='--', linewidth=1)

    plt.text(x_joint, 0, "K", ha='center', va='bottom', fontsize=10)
plt.legend()
plt.grid(True)
plt.show()






#
