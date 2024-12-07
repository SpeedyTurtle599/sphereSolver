import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, gridspec

def read_velocity_field(filename):
    # Read the header to get dimensions and sphere info
    with open(filename, 'r') as f:
        _ = f.readline()  # Skip title
        _ = f.readline()  # Skip timestep
        radius_line = f.readline()
        sphere_radius = float(radius_line.split(':')[1])
        dim_line = f.readline()
        dims = {k: int(v) for k, v in 
                [item.split('=') for item in dim_line.strip('# \n').split()]}
        # Skip the column headers line
        f.readline()
        
        # Read the data
        data = np.loadtxt(f)
    
    # Reshape the data
    x = data[:, 0].reshape(dims['nz'], dims['ny'], dims['nx'])
    y = data[:, 1].reshape(dims['nz'], dims['ny'], dims['nx'])
    z = data[:, 2].reshape(dims['nz'], dims['ny'], dims['nx'])
    u = data[:, 3].reshape(dims['nz'], dims['ny'], dims['nx'])
    v = data[:, 4].reshape(dims['nz'], dims['ny'], dims['nx'])
    w = data[:, 5].reshape(dims['nz'], dims['ny'], dims['nx'])
    vel_mag = data[:, 6].reshape(dims['nz'], dims['ny'], dims['nx'])
    
    return x, y, z, u, v, w, vel_mag, sphere_radius

def plot_flow_field(filename):
    # Update unpacking to include sphere_radius
    x, y, z, u, v, w, vel_mag, sphere_radius = read_velocity_field(filename)
    
    # Setup figure with GridSpec
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(1, 2, width_ratios=[20, 1])
    
    # Create 3D axis
    ax = fig.add_subplot(gs[0], projection='3d')
    
    # Take slices at the middle
    mid_z = x.shape[0] // 2
    
    # Downsample for quiver
    skip = 4
    x_sub = x[::skip, ::skip, ::skip]
    y_sub = y[::skip, ::skip, ::skip]
    z_sub = z[::skip, ::skip, ::skip]
    u_sub = u[::skip, ::skip, ::skip]
    v_sub = v[::skip, ::skip, ::skip]
    w_sub = w[::skip, ::skip, ::skip]
    vel_mag_sub = vel_mag[::skip, ::skip, ::skip]
    
    # Setup colormap
    norm = plt.Normalize(vel_mag_sub.min(), vel_mag_sub.max())
    colors = cm.viridis(norm(vel_mag_sub))
    
    # Plot quiver
    quiv = ax.quiver(x_sub, y_sub, z_sub, 
                    u_sub, v_sub, w_sub,
                    length=0.5,
                    normalize=True,
                    colors=colors.reshape(-1, 4))
    
    # Add slice
    slice_x = ax.contourf(x[mid_z,:,:], y[mid_z,:,:], vel_mag[mid_z,:,:],
                         zdir='z', offset=z[mid_z,0,0],
                         levels=20, cmap='viridis', alpha=0.5)
    
    # Use actual sphere radius from file
    sphere_center = np.mean([x.min(), x.max()])
    
    phi = np.linspace(0, 2 * np.pi, 20)
    theta = np.linspace(0, np.pi, 20)
    phi, theta = np.meshgrid(phi, theta)
    
    sphere_x = sphere_center + sphere_radius * np.cos(phi) * np.sin(theta)
    sphere_y = sphere_center + sphere_radius * np.sin(phi) * np.sin(theta)
    sphere_z = sphere_center + sphere_radius * np.cos(theta)
    ax.plot_surface(sphere_x, sphere_y, sphere_z, color='gray', alpha=0.3)
    
    # Labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Flow Field Visualization')
    
    # Add colorbar with proper axes
    cax = fig.add_subplot(gs[1])
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    plt.colorbar(sm, cax=cax, label='Velocity Magnitude')
    
    # Equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    plt.tight_layout()
    plt.show()

# Run visualization
filename = "velocity_field_000000.dat"
plot_flow_field(filename)