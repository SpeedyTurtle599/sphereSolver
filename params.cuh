// k-epsilon parameters
#define C_mu 0.09f
#define C_1 1.44f
#define C_2 1.92f
#define SIGMA_k 1.0f
#define SIGMA_epsilon 1.3f

// Wall law parameters
#define KAPPA 0.41f       // Von Karmann constant
#define E_CONST 9.793f    // Wall roughness parameter
#define Y_PLUS_MIN 11.63f // Lower bound for log-law region
#define Y_PLUS_MAX 300.0f // Upper bound for log-law region

// Constants for SIMPLEC algorithm
#define P_REF 101325.0f // Reference pressure (Pa)
#define ALPHA_P 0.2f    // Pressure under-relaxation
#define ALPHA_U 0.8f    // Velocity under-relaxation
#define ALPHA_K 0.5f    // Turbulent kinetic energy under-relaxation
#define ALPHA_EPS 0.5f  // Dissipation rate under-relaxation
#define MIN_PRESSURE_ITER 20
#define MAX_PRESSURE_ITER 50
#define PRESSURE_TOLERANCE 1e-6f

// Simulation parameters
#define DT 0.0001f             // Time step
#define MIN_DT 1e-6f           // Minimum time step
#define MAX_DT 0.01f           // Maximum time step
#define DX 0.1f                // Grid spacing
#define NU 1.5e-5f             // Kinematic viscosity
#define RHO 1.0f               // Density
#define RESIDUAL_TOL 1e-8f     // Tolerance for residuals
#define MAX_ITER 1000         // Maximum number of iterations
#define MIN_ITER 10          // Minimum number of iterations
#define MIN_RESIDUAL_CHECK 100 // Check residuals every n iterations

// Grid parameters
#define NX 128       // Grid size in x-direction
#define NY 128       // Grid size in y-direction
#define NZ 128       // Grid size in z-direction
#define BLOCK_SIZE 8 // Block size for CUDA shared memory

// Object parameters
#define SPHERE_RADIUS 2.0f
#define SPHERE_CENTER_X (NX / 2.0f)
#define SPHERE_CENTER_Y (NY / 2.0f)
#define SPHERE_CENTER_Z (NZ / 2.0f)

// Boundary parameters
#define INLET_VELOCITY 1.0f
#define CONVECTIVE_VELOCITY INLET_VELOCITY
#define INLET_TURBULENT_INTENSITY 0.05f
#define OUTLET_PRESSURE P_REF