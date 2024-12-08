#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_atomic_functions.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#include "params.cuh"

#define CUDA_CHECK(call)                                                    \
    {                                                                       \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess)                                             \
        {                                                                   \
            printf("CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err));                                \
            exit(1);                                                        \
        }                                                                   \
    }

// Function prototypes
__device__ bool isInsideSphere(int i, int j, int k);
__device__ bool isNearSphere(int i, int j, int k);
__device__ float getInletVelocityProfile(int j, int k, int ny, int nz);
__device__ void getInletConditions(float &k_val, float &eps_val, float u_inlet);
__device__ float calculateYPlus(float uTau, float yWall, float nu);
__device__ float calculateUTau(float uTangential, float yWall, float nu);
__device__ void applyWallFunctions(float &u_new, float &k_new, float &eps_new, float uTangential, float yWall, float nu);
__device__ float calculateStrainRate(float *u, float *v, float *w, int idx, int nx, int ny, int nz);

// MARK: FloatInt union
union FloatInt
{
    float f;
    unsigned int i;
};

// MARK: atomicMaxFloat
__device__ float atomicMaxFloat(float *address, float val)
{
    float old = *address, assumed;

    do
    {
        assumed = old;
        old = __int_as_float(atomicCAS(
            (int *)address,
            __float_as_int(assumed),
            __float_as_int(fmaxf(assumed, val))));
    } while (assumed < val && old != assumed);

    return old;
}

// MARK: def FlowField
struct FlowField
{
    float *u, *v, *w; // Velocity components
    float *p;         // Pressure
    float *k_field;   // Turbulent kinetic energy
    float *epsilon;   // Dissipation rate
    float *nut;       // Turbulent viscosity
    float *residuals; // Residuals for SIMPLE-C and k-epsilon models
};

// MARK: calculateTurbulentViscosity
__global__ void calculateTurbulentViscosity(float *nut, float *k_field, float *epsilon, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        nut[idx] = C_mu * (k_field[idx] * k_field[idx]) / epsilon[idx]; // eddy viscosity model
    }
}

// MARK: calculateCFL
__device__ float calculateCFL(float u, float v, float w, float dx, float dt)
{
    float vel_mag = sqrtf(u * u + v * v + w * w);
    return vel_mag * dt / dx;
}

// MARK: calculateMaxCFL
__global__ void calculateMaxCFL(float *max_cfl, float *u, float *v, float *w, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        float vel_mag = sqrtf(u[idx] * u[idx] + v[idx] * v[idx] + w[idx] * w[idx]);
        float cfl = vel_mag * DT / DX;

        // Use custom atomicMaxFloat function
        atomicMaxFloat(max_cfl, cfl);
    }
}

// MARK: calculateResiduals
__global__ void calculateResiduals(float *residuals,
                                   float *u, float *u_old,
                                   float *v, float *v_old,
                                   float *w, float *w_old,
                                   float *k_field, float *k_old,
                                   float *eps, float *eps_old,
                                   int nx, int ny, int nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = nx * ny * nz;

    if (idx < size)
    {
        // Skip sphere and boundaries
        int k = idx / (nx * ny);
        int j = (idx - k * nx * ny) / nx;
        int i = idx - k * nx * ny - j * nx;

        if (!isInsideSphere(i, j, k) && i > 0 && i < nx - 1 && j > 0 && j < ny - 1 && k > 0 && k < nz - 1)
        {
            // Normalize by reference values
            float u_ref = fmaxf(fabsf(u_old[idx]), INLET_VELOCITY);
            float v_ref = fmaxf(fabsf(v_old[idx]), INLET_VELOCITY);
            float w_ref = fmaxf(fabsf(w_old[idx]), INLET_VELOCITY);
            float k_ref = fmaxf(fabsf(k_old[idx]), 1e-6f);
            float eps_ref = fmaxf(fabsf(eps_old[idx]), 1e-6f);

            // Calculate normalized residuals
            float du = fabsf(u[idx] - u_old[idx]) / u_ref;
            float dv = fabsf(v[idx] - v_old[idx]) / v_ref;
            float dw = fabsf(w[idx] - w_old[idx]) / w_ref;
            float dk = fabsf(k_field[idx] - k_old[idx]) / k_ref;
            float deps = fabsf(eps[idx] - eps_old[idx]) / eps_ref;

            // Update maximum residuals
            atomicMaxFloat(&residuals[0], du);
            atomicMaxFloat(&residuals[1], dv);
            atomicMaxFloat(&residuals[2], dw);
            atomicMaxFloat(&residuals[3], dk);
            atomicMaxFloat(&residuals[4], deps);
        }
    }
}

// MARK: storeOldValues
__global__ void storeOldValues(float *u_old, float *v_old, float *w_old,
                               float *k_old, float *eps_old,
                               float *u, float *v, float *w,
                               float *k_field, float *eps, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        u_old[idx] = u[idx];
        v_old[idx] = v[idx];
        w_old[idx] = w[idx];
        k_old[idx] = k_field[idx];
        eps_old[idx] = eps[idx];
    }
}

// MARK: extractMonitoringValues
__global__ void extractMonitoringValues(float *field, float *monitor_values, int *monitor_indices, int num_points)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_points)
    {
        int field_idx = monitor_indices[idx];
        monitor_values[idx] = field[field_idx];
    }
}

// MARK: computeMonitoringResiduals
__global__ void computeMonitoringResiduals(float *residuals, float *curr_values, float *prev_values, int num_points)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_points)
    {
        residuals[idx] = fabsf(curr_values[idx] - prev_values[idx]);
    }
}

// MARK: initializeFlowField
void initializeFlowField(FlowField *flow, int size)
{
    CUDA_CHECK(cudaMalloc(&flow->u, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&flow->v, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&flow->w, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&flow->p, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&flow->k_field, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&flow->epsilon, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&flow->nut, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&flow->residuals, 5 * sizeof(float))); // u, v, w, k_field, epsilon
}

// MARK: initializePressure
__global__ void initializePressure(float *p, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        p[idx] = P_REF;
    }
}

// MARK: initializeFields
__global__ void initializeFields(float *k_field, float *epsilon, float *u, float *v, float *w, int nx, int ny, int nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = nx * ny * nz;

    if (idx < size)
    {
        int k = idx / (nx * ny);
        int j = (idx - k * nx * ny) / nx;
        int i = idx - k * nx * ny - j * nx;

        // Base turbulence values
        float u_mag = sqrtf(u[idx] * u[idx] + v[idx] * v[idx] + w[idx] * w[idx]);
        float I = 0.05f; // Initial turbulence intensity (5%)

        if (isInsideSphere(i, j, k))
        {
            k_field[idx] = 1e-10f;
            epsilon[idx] = 1e-10f;
        }
        else
        {
            // k_field = 3/2 * (U*I)^2
            k_field[idx] = 1.5f * powf(u_mag * I, 2.0f);

            // epsilon = C_mu^(3/4) * k_field^(3/2) / L
            float L = 0.07f * SPHERE_RADIUS; // Turbulent length scale
            epsilon[idx] = powf(C_mu, 0.75f) * powf(k_field[idx], 1.5f) / L;

            // Ensure minimum values
            k_field[idx] = fmaxf(k_field[idx], 1e-10f);
            epsilon[idx] = fmaxf(epsilon[idx], 1e-10f);
        }
    }
}

// MARK: computeDivergence
__global__ void computeDivergence(float *div, float *u, float *v, float *w, int nx, int ny, int nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = nx * ny * nz;

    if (idx < size)
    {
        float du_dx = (u[idx + 1] - u[idx - 1]) / (2.0f * DX);
        float dv_dy = (v[idx + nx] - v[idx - nx]) / (2.0f * DX);
        float dw_dz = (w[idx + nx * ny] - w[idx - nx * ny]) / (2.0f * DX);

        div[idx] = du_dx + dv_dy + dw_dz;
    }
}

// MARK: calculatePressureCorrection
__global__ void calculatePressureCorrection(float *p_corr, float *div, float *ap, int nx, int ny, int nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = nx * ny * nz;

    if (idx > nx && idx < size - nx)
    {
        if (!isInsideSphere(idx / (nx * ny), (idx / nx) % ny, idx % nx))
        {
            float ae = 1.0f / (DX * DX);
            float aw = ae;
            float an = ae;
            float as = ae;
            float at = ae;
            float ab = ae;

            ap[idx] = -(ae + aw + an + as + at + ab);

            p_corr[idx] = -(div[idx] -
                            ae * p_corr[idx + 1] - aw * p_corr[idx - 1] -
                            an * p_corr[idx + nx] - as * p_corr[idx - nx] -
                            at * p_corr[idx + nx * ny] - ab * p_corr[idx - nx * ny]) /
                          ap[idx];
        }
    }
}

// MARK: correctVelocities
__global__ void correctVelocities(float *u, float *v, float *w, float *p_corr, float *ap, int nx, int ny, int nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = nx * ny * nz;

    if (idx > nx && idx < size - nx && !isInsideSphere(idx / (nx * ny), (idx / nx) % ny, idx % nx))
    {
        // Velocity corrections
        float dp_dx = (p_corr[idx + 1] - p_corr[idx - 1]) / (2.0f * DX);
        float dp_dy = (p_corr[idx + nx] - p_corr[idx - nx]) / (2.0f * DX);
        float dp_dz = (p_corr[idx + nx * ny] - p_corr[idx - nx * ny]) / (2.0f * DX);

        u[idx] -= dp_dx / ap[idx];
        v[idx] -= dp_dy / ap[idx];
        w[idx] -= dp_dz / ap[idx];
    }
}

// MARK: updatePressure
__global__ void updatePressure(float *p, float *p_corr, float alpha, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        p[idx] += alpha * p_corr[idx];
    }
}

// MARK: underRelaxPressureCorrection
__global__ void underRelaxPressureCorrection(float *p_corr, float *p_corr_old, float alpha, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        p_corr[idx] = alpha * p_corr[idx] + (1.0f - alpha) * p_corr_old[idx];
    }
}

// MARK: calculateYPlus
__device__ float calculateYPlus(float uTau, float yWall, float nu)
{
    return uTau * yWall / nu;
}

// MARK: calculateUTau
__device__ float calculateUTau(float uTangential, float yWall, float nu)
{
    // Initial guess for friction velocity
    float uTau = sqrtf(nu * fabsf(uTangential) / yWall);

    // Newton-Raphson iteration to solve wall function
    for (int iter = 0; iter < 10; iter++)
    {
        float yPlus = calculateYPlus(uTau, yWall, nu);
        if (yPlus < Y_PLUS_MIN)
        {
            // Viscous sublayer
            uTau = sqrtf(nu * fabsf(uTangential) / yWall);
            break;
        }

        float u_plus = (1.0f / KAPPA) * logf(E_CONST * yPlus);
        float f = uTau * u_plus - fabsf(uTangential);
        float df = u_plus + uTau / (KAPPA * yPlus);
        uTau -= f / df;

        if (fabsf(f) < 1e-6f)
            break;
    }
    return uTau;
}

// MARK: applyWallFunctions
__device__ void applyWallFunctions(float &u_new, float &k_new, float &eps_new, float uTangential, float yWall, float nu)
{
    float uTau = calculateUTau(uTangential, yWall, nu);
    float yPlus = calculateYPlus(uTau, yWall, nu);

    if (yPlus < Y_PLUS_MIN)
    {
        // Viscous sublayer
        u_new = uTangential * yPlus / Y_PLUS_MIN;
        k_new = 0.0f;
        eps_new = 2.0f * nu * k_new / (yWall * yWall);
    }
    else
    {
        // Log-law region
        u_new = (uTau / KAPPA) * logf(E_CONST * yPlus);
        k_new = uTau * uTau / sqrtf(C_mu);
        eps_new = (uTau * uTau * uTau) / (KAPPA * yWall);
    }
}

// MARK: applyBoundaryConditions
__global__ void applyBoundaryConditions(float *u, float *v, float *w, float *p, float *k_field, float *epsilon, int nx, int ny, int nz)
{
    // Use threadIdx for j and k coordinates
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (j < ny && k < nz)
    {
        // Walls (j = 0 || j = ny-1, k = 0 || k = nz-1)
        if (j == 0 || j == ny - 1)
        {
            // Apply wall or symmetry conditions for y-direction
            int idx = j * nx + k * nx * ny;
            v[idx] = 0.0f;                         // No-slip for walls
            u[idx] = u[idx + (j == 0 ? nx : -nx)]; // Copy tangential velocities
            w[idx] = w[idx + (j == 0 ? nx : -nx)];
        }
        if (k == 0 || k == nz - 1)
        {
            // Apply wall or symmetry conditions for z-direction
            int idx = j * nx + k * nx * ny;
            w[idx] = 0.0f; // No-slip for walls
            u[idx] = u[idx + (k == 0 ? nx * ny : -nx * ny)];
            v[idx] = v[idx + (k == 0 ? nx * ny : -nx * ny)];
        }

        // Inlet boundary (x = 0)
        int inlet_idx = j * nx + k * nx * ny;
        if (!isInsideSphere(0, j, k)) // Check if not inside sphere
        {
            u[inlet_idx] = getInletVelocityProfile(j, k, ny, nz);
            v[inlet_idx] = 0.0f;
            w[inlet_idx] = 0.0f;

            // Set inlet turbulence values
            float k_inlet, eps_inlet;
            getInletConditions(k_inlet, eps_inlet, INLET_VELOCITY);
            k_field[inlet_idx] = k_inlet;
            epsilon[inlet_idx] = eps_inlet;
        }
        else
        {
            // No-slip condition if inside sphere
            u[inlet_idx] = 0.0f;
            v[inlet_idx] = 0.0f;
            w[inlet_idx] = 0.0f;
            k_field[inlet_idx] = 1e-10f;
            epsilon[inlet_idx] = 1e-10f;
        }

        // Outlet boundary (x = nx-1)
        int outlet_idx = (nx - 1) + j * nx + k * nx * ny;
        if (!isInsideSphere(nx - 1, j, k))
        {
            // Convective outlet condition: du/dt + U_conv * du/dx = 0
            float dx_u = (u[outlet_idx] - u[outlet_idx - 1]) / DX;
            float dx_v = (v[outlet_idx] - v[outlet_idx - 1]) / DX;
            float dx_w = (w[outlet_idx] - w[outlet_idx - 1]) / DX;
            float dx_k_field = (k_field[outlet_idx] - k_field[outlet_idx - 1]) / DX;
            float dx_eps = (epsilon[outlet_idx] - epsilon[outlet_idx - 1]) / DX;

            u[outlet_idx] = u[outlet_idx] - DT * CONVECTIVE_VELOCITY * dx_u;
            v[outlet_idx] = v[outlet_idx] - DT * CONVECTIVE_VELOCITY * dx_v;
            w[outlet_idx] = w[outlet_idx] - DT * CONVECTIVE_VELOCITY * dx_w;
            k_field[outlet_idx] = k_field[outlet_idx] - DT * CONVECTIVE_VELOCITY * dx_k_field;
            epsilon[outlet_idx] = epsilon[outlet_idx] - DT * CONVECTIVE_VELOCITY * dx_eps;

            // Fixed pressure at outlet
            p[outlet_idx] = OUTLET_PRESSURE;
        }
        else
        {
            // Keep existing no-slip conditions for sphere intersection
            u[outlet_idx] = 0.0f;
            v[outlet_idx] = 0.0f;
            w[outlet_idx] = 0.0f;
            k_field[outlet_idx] = 1e-10f;
            epsilon[outlet_idx] = 1e-10f;
        }
    }
}

// MARK: getInletConditions
__device__ void getInletConditions(float &k_val, float &eps_val, float u_inlet)
{
    // Use predefined turbulence intensity
    float I = INLET_TURBULENT_INTENSITY; // Already defined as 0.05f
    float L = 0.07f * SPHERE_RADIUS;     // Turbulent length scale

    // Calculate k_field from turbulence intensity
    // k_field = 3/2 * (U*I)^2
    k_val = 1.5f * powf(u_inlet * I, 2.0f);

    // Calculate epsilon from k_field and length scale
    // epsilon = C_mu^(3/4) * k_field^(3/2) / L
    eps_val = powf(C_mu, 0.75f) * powf(k_val, 1.5f) / L;

    // Ensure minimum values
    k_val = fmaxf(k_val, 1e-10f);
    eps_val = fmaxf(eps_val, 1e-10f);
}

// MARK: getInletVelocityProfile
__device__ float getInletVelocityProfile(int j, int k, int ny, int nz)
{
    float y = (j - ny / 2.0f) / (ny / 2.0f);
    float z = (k - nz / 2.0f) / (nz / 2.0f);
    float r = sqrtf(y * y + z * z);

    r = fminf(r, 1.0f); // Cap r at 1.0 to prevent negative values

    return INLET_VELOCITY; // * (1.0f - r * r);
}

// MARK: isInsideSphere
__device__ bool isInsideSphere(int i, int j, int k)
{
    float x = i * DX - SPHERE_CENTER_X * DX;
    float y = j * DX - SPHERE_CENTER_Y * DX;
    float z = k * DX - SPHERE_CENTER_Z * DX;
    float r2 = x * x + y * y + z * z;
    return r2 <= SPHERE_RADIUS * SPHERE_RADIUS;
}

// MARK: isNearSphere
__device__ bool isNearSphere(int i, int j, int k)
{
    float x = i * DX - SPHERE_CENTER_X * DX;
    float y = j * DX - SPHERE_CENTER_Y * DX;
    float z = k * DX - SPHERE_CENTER_Z * DX;
    float r2 = x * x + y * y + z * z;
    float outer2 = (SPHERE_RADIUS + 1.5f * DX) * (SPHERE_RADIUS + 1.5f * DX);
    return r2 <= outer2;
}

// MARK: calculateStrainRate
__device__ float calculateStrainRate(float *u, float *v, float *w, int idx, int nx, int ny, int nz)
{
    // Calculate strain rate tensor magnitude
    float du_dx = (u[idx + 1] - u[idx - 1]) / (2.0f * DX);
    float dv_dy = (v[idx + nx] - v[idx - nx]) / (2.0f * DX);
    float dw_dz = (w[idx + nx * ny] - w[idx - nx * ny]) / (2.0f * DX);

    float du_dy = (u[idx + nx] - u[idx - nx]) / (2.0f * DX);
    float du_dz = (u[idx + nx * ny] - u[idx - nx * ny]) / (2.0f * DX);
    float dv_dx = (v[idx + 1] - v[idx - 1]) / (2.0f * DX);
    float dv_dz = (v[idx + nx * ny] - v[idx - nx * ny]) / (2.0f * DX);
    float dw_dx = (w[idx + 1] - w[idx - 1]) / (2.0f * DX);
    float dw_dy = (w[idx + nx] - w[idx - nx]) / (2.0f * DX);

    return sqrtf(2.0f * (du_dx * du_dx + dv_dy * dv_dy + dw_dz * dw_dz +
                         0.5f * ((du_dy + dv_dx) * (du_dy + dv_dx) +
                                 (du_dz + dw_dx) * (du_dz + dw_dx) +
                                 (dv_dz + dw_dy) * (dv_dz + dw_dy))));
}

// MARK: solveKEquation
__global__ void solveKEquation(float *k_new, float *k_field, float *epsilon, float *u, float *v, float *w, float *nut, int nx, int ny, int nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = nx * ny * nz;

    if (idx >= 0 && idx < size)
    {
        int k = idx / (nx * ny);
        int j = (idx - k * nx * ny) / nx;
        int i = idx - k * nx * ny - j * nx;

        // Initialize with old value
        k_new[idx] = k_field[idx];

        // Skip boundaries and sphere
        if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1 || k == 0 || k == nz - 1)
            return;
        if (isInsideSphere(i, j, k))
        {
            k_new[idx] = 1e-10f;
            return;
        }

        // Production term
        float S = calculateStrainRate(u, v, w, idx, nx, ny, nz);
        float P_k = nut[idx] * S * S;

        // Convection terms
        float u_avg = 0.5f * (u[idx] + u[idx - 1]);
        float v_avg = 0.5f * (v[idx] + v[idx - nx]);
        float w_avg = 0.5f * (w[idx] + w[idx - nx * ny]);

        float k_dx = (k_field[idx + 1] - k_field[idx - 1]) / (2.0f * DX);
        float k_dy = (k_field[idx + nx] - k_field[idx - nx]) / (2.0f * DX);
        float k_dz = (k_field[idx + nx * ny] - k_field[idx - nx * ny]) / (2.0f * DX);

        float convection = -(u_avg * k_dx + v_avg * k_dy + w_avg * k_dz);

        // Diffusion term
        float k_lap = (k_field[idx + 1] + k_field[idx - 1] + k_field[idx + nx] + k_field[idx - nx] +
                       k_field[idx + nx * ny] + k_field[idx - nx * ny] - 6.0f * k_field[idx]) /
                      (DX * DX);
        float diff_k = (NU + nut[idx] / SIGMA_k) * k_lap;

        // Update k_field with all terms
        float rhs = P_k - epsilon[idx] + diff_k + convection;
        k_new[idx] = k_field[idx] + DT * rhs;
        k_new[idx] = fmaxf(k_new[idx], 1e-10f);
    }
}

// MARK: solveEpsilonEquation
__global__ void solveEpsilonEquation(float *eps_new, float *k_field, float *epsilon, float *u, float *v, float *w, float *nut, int nx, int ny, int nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = nx * ny * nz;

    if (idx >= 0 && idx < size)
    {
        int k = idx / (nx * ny);
        int j = (idx - k * nx * ny) / nx;
        int i = idx - k * nx * ny - j * nx;

        // Initialize with old value
        eps_new[idx] = epsilon[idx];

        // Skip boundaries and sphere
        if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1 || k == 0 || k == nz - 1)
            return;
        if (isInsideSphere(i, j, k))
        {
            eps_new[idx] = 1e-10f;
            return;
        }

        // Production term
        float S = calculateStrainRate(u, v, w, idx, nx, ny, nz);
        float P_k = nut[idx] * S * S;

        // Convection terms
        float u_avg = 0.5f * (u[idx] + u[idx - 1]);
        float v_avg = 0.5f * (v[idx] + v[idx - nx]);
        float w_avg = 0.5f * (w[idx] + w[idx - nx * ny]);

        float eps_dx = (epsilon[idx + 1] - epsilon[idx - 1]) / (2.0f * DX);
        float eps_dy = (epsilon[idx + nx] - epsilon[idx - nx]) / (2.0f * DX);
        float eps_dz = (epsilon[idx + nx * ny] - epsilon[idx - nx * ny]) / (2.0f * DX);

        float convection = -(u_avg * eps_dx + v_avg * eps_dy + w_avg * eps_dz);

        // Diffusion term
        float eps_lap = (epsilon[idx + 1] + epsilon[idx - 1] + epsilon[idx + nx] + epsilon[idx - nx] +
                         epsilon[idx + nx * ny] + epsilon[idx - nx * ny] - 6.0f * epsilon[idx]) /
                        (DX * DX);
        float diff_eps = (NU + nut[idx] / SIGMA_epsilon) * eps_lap;

        // Source terms
        float k_val = fmaxf(k_field[idx], 1e-10f); // Prevent division by zero
        float eps_val = fmaxf(epsilon[idx], 1e-10f);

        float source = C_1 * eps_val * P_k / k_val -
                       C_2 * eps_val * eps_val / k_val;

        // Update epsilon with all terms
        float rhs = source + diff_eps + convection;
        eps_new[idx] = epsilon[idx] + DT * rhs;
        eps_new[idx] = fmaxf(eps_new[idx], 1e-10f);
    }
}

// MARK: solveXMomentum
__global__ void solveXMomentum(float *u_new, float *u, float *v, float *w, float *p, float *nut, float *k_field, float *epsilon, int nx, int ny, int nz, float dt)
{
    // Compute 3D thread indices
    int i = blockIdx.x * blockDim.x + threadIdx.x; // x-coordinate
    int j = blockIdx.y * blockDim.y + threadIdx.y; // y-coordinate
    int k = blockIdx.z * blockDim.z + threadIdx.z; // z-coordinate

    // Check bounds
    if (i >= nx || j >= ny || k >= nz)
        return;

    // Compute linear index
    int idx = i + j * nx + k * nx * ny;

    // Initialize with old value
    u_new[idx] = u[idx];

    // Skip inlet/outlet boundaries
    if (i == 0 || i == nx - 1)
        return;

    // Handle sphere regions
    if (isInsideSphere(i, j, k))
    {
        u_new[idx] = 0.0f;
        return;
    }

    // Skip boundaries for finite differences
    if (i == 1 || i == nx - 2 || j == 0 || j == ny - 1 || k == 0 || k == nz - 1)
        return;

    // Compute velocity gradients
    float u_dx = (u[idx + 1] - u[idx - 1]) / (2.0f * DX);
    float u_dy = (u[idx + nx] - u[idx - nx]) / (2.0f * DX);
    float u_dz = (u[idx + nx * ny] - u[idx - nx * ny]) / (2.0f * DX);

    // Convection term
    float convection = -(u[idx] * u_dx + v[idx] * u_dy + w[idx] * u_dz);

    // Pressure gradient
    float pressure_grad = -(p[idx + 1] - p[idx - 1]) / (2.0f * DX * RHO);

    // Laplacian of u (del^2 u)
    float lap_u = (u[idx + 1] + u[idx - 1] + u[idx + nx] + u[idx - nx] + u[idx + nx * ny] + u[idx - nx * ny] - 6.0f * u[idx]) / (DX * DX);

    // Viscous term
    float viscous = (NU + nut[idx]) * lap_u;

    // Update u_new
    u_new[idx] += ALPHA_U * dt * (viscous + convection + pressure_grad);
}

// MARK: solveYMomentum
__global__ void solveYMomentum(float *v_new, float *u, float *v, float *w, float *p, float *nut, float *k_field, float *epsilon, int nx, int ny, int nz, float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz)
        return;

    int idx = i + j * nx + k * nx * ny;

    v_new[idx] = v[idx];

    if (j == 0 || j == ny - 1)
        return;

    if (isInsideSphere(i, j, k))
    {
        v_new[idx] = 0.0f;
        return;
    }

    if (i == 0 || i == nx - 1 || j == 1 || j == ny - 2 || k == 0 || k == nz - 1)
        return;

    float v_dx = (v[idx + 1] - v[idx - 1]) / (2.0f * DX);
    float v_dy = (v[idx + nx] - v[idx - nx]) / (2.0f * DX);
    float v_dz = (v[idx + nx * ny] - v[idx - nx * ny]) / (2.0f * DX);

    float convection = -(u[idx] * v_dx + v[idx] * v_dy + w[idx] * v_dz);

    float pressure_grad = -(p[idx + nx] - p[idx - nx]) / (2.0f * DX * RHO);

    float lap_v = (v[idx + 1] + v[idx - 1] + v[idx + nx] + v[idx - nx] + v[idx + nx * ny] + v[idx - nx * ny] - 6.0f * v[idx]) / (DX * DX);

    float viscous = (NU + nut[idx]) * lap_v;

    v_new[idx] += ALPHA_U * dt * (viscous + convection + pressure_grad);
}

// MARK: solveZMomentum
__global__ void solveZMomentum(float *w_new, float *u, float *v, float *w, float *p, float *nut, float *k_field, float *epsilon, int nx, int ny, int nz, float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= nx || j >= ny || k >= nz)
        return;

    int idx = i + j * nx + k * nx * ny;

    w_new[idx] = w[idx];

    if (k == 0 || k == nz - 1)
        return;

    if (isInsideSphere(i, j, k))
    {
        w_new[idx] = 0.0f;
        return;
    }

    if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1 || k == 1 || k == nz - 2)
        return;

    float w_dx = (w[idx + 1] - w[idx - 1]) / (2.0f * DX);
    float w_dy = (w[idx + nx] - w[idx - nx]) / (2.0f * DX);
    float w_dz = (w[idx + nx * ny] - w[idx - nx * ny]) / (2.0f * DX);

    float convection = -(u[idx] * w_dx + v[idx] * w_dy + w[idx] * w_dz);

    float pressure_grad = -(p[idx + nx * ny] - p[idx - nx * ny]) / (2.0f * DX * RHO);

    float lap_w = (w[idx + 1] + w[idx - 1] + w[idx + nx] + w[idx - nx] + w[idx + nx * ny] + w[idx - nx * ny] - 6.0f * w[idx]) / (DX * DX);

    float viscous = (NU + nut[idx]) * lap_w;

    w_new[idx] += ALPHA_U * dt * (viscous + convection + pressure_grad);
}

// MARK: saveFieldData
void saveFieldData(FlowField *flow, int step, int nx, int ny, int nz)
{
    char filename[256];
    sprintf(filename, "velocity_field_%06d.dat", step);
    FILE *fp = fopen(filename, "w");

    // Allocate host memory for the field data
    int size = nx * ny * nz;
    float *h_u = new float[size];
    float *h_v = new float[size];
    float *h_w = new float[size];

    // Copy data from device to host
    cudaMemcpy(h_u, flow->u, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_v, flow->v, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_w, flow->w, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Write header with dimensions
    fprintf(fp, "# Data generated by CFD solver\n");
    fprintf(fp, "# Time step: %d\n", step);
    fprintf(fp, "# Sphere radius: %f\n", SPHERE_RADIUS);
    fprintf(fp, "# nx=%d ny=%d nz=%d\n", nx, ny, nz);
    fprintf(fp, "# x y z u v w velocity_magnitude\n");

    // Write field data
    for (int idx = 0; idx < size; idx++)
    {
        int i = idx % nx;
        int j = (idx / nx) % ny;
        int k = idx / (nx * ny);

        float x = i * DX;
        float y = j * DX;
        float z = k * DX;
        float u = h_u[idx];
        float v = h_v[idx];
        float w = h_w[idx];
        float vel_mag = sqrtf(u * u + v * v + w * w);

        fprintf(fp, "%f %f %f %f %f %f %f\n", x, y, z, u, v, w, vel_mag);
    }

    fclose(fp);
    delete[] h_u;
    delete[] h_v;
    delete[] h_w;
}