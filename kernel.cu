#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_atomic_functions.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#include "params.cuh"
#include "helpers.cuh"

// MARK: main
int main(int argc, char **argv) // for future CLI arguments
{
    // Print CUDA device info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Using GPU: %s\n", prop.name);

    // Print simulation info
    printf("Starting CFD solver...\n");
    printf("Grid size: %d x %d x %d\n", NX, NY, NZ);
    printf("Block size: %d x %d x %d\n", BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);

    int size = NX * NY * NZ;
    FlowField flow;

    // MARK: monitoring points
    // Define monitoring points (i, j, k)
    const int num_monitor_points = 3; // Number of monitoring points
    int h_monitor_indices[num_monitor_points]; // Array to hold linear indices

    // Monitoring points: 1/4, 1/2, 3/4 along x-axis
    int monitor_coords[num_monitor_points][3] = {
        {NX / 4, NY / 2, NZ / 2},
        {NX / 2, NY / 2, NZ / 2},
        {3 * NX / 4, NY / 2, NZ / 2}
    };

    // Compute linear indices
    for (int n = 0; n < num_monitor_points; n++)
    {
        int i = monitor_coords[n][0];
        int j = monitor_coords[n][1];
        int k = monitor_coords[n][2];
        h_monitor_indices[n] = i + j * NX + k * NX * NY;
    }

    // Device arrays for monitoring indices and field values
    int *d_monitor_indices;
    CUDA_CHECK(cudaMalloc(&d_monitor_indices, num_monitor_points * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_monitor_indices, h_monitor_indices, num_monitor_points * sizeof(int), cudaMemcpyHostToDevice));

    // Device arrays for previous and current values
    float *d_u_monitor_prev, *d_u_monitor_curr;
    CUDA_CHECK(cudaMalloc(&d_u_monitor_prev, num_monitor_points * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_u_monitor_curr, num_monitor_points * sizeof(float)));

    // Device array for residuals at monitoring points
    float *d_monitor_residuals;
    CUDA_CHECK(cudaMalloc(&d_monitor_residuals, num_monitor_points * sizeof(float)));

    // MARK: cfl
    float *d_max_cfl;
    CUDA_CHECK(cudaMalloc(&d_max_cfl, sizeof(float)));
    float dt = DT; // initialize time step as DT param val

    // Set up grid and blocks for CUDA
    dim3 block(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((NX + block.x - 1) / block.x,
              (NY + block.y - 1) / block.y,
              (NZ + block.z - 1) / block.z);

    // Initialize memory
    initializeFlowField(&flow, size);
    initializePressure<<<grid, block>>>(flow.p, size);

    float *u_old, *v_old, *w_old, *k_old, *eps_old;
    CUDA_CHECK(cudaMalloc(&u_old, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&v_old, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&w_old, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&k_old, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&eps_old, size * sizeof(float)));

    float *u_new, *v_new, *w_new;
    CUDA_CHECK(cudaMalloc(&u_new, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&v_new, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&w_new, size * sizeof(float)));

    float *k_new, *eps_new;
    CUDA_CHECK(cudaMalloc(&k_new, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&eps_new, size * sizeof(float)));

    float *p_corr, *div, *ap;
    CUDA_CHECK(cudaMalloc(&p_corr, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&div, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ap, size * sizeof(float)));

    // MARK: init velocity field
    // Initialize all velocities with perturbations
    float *h_u = new float[size];
    float *h_v = new float[size];
    float *h_w = new float[size];

    srand(static_cast<unsigned int>(time(NULL))); // Add randomization

    for (int k = 0; k < NZ; k++)
    {
        for (int j = 0; j < NY; j++)
        {
            for (int i = 0; i < NX; i++)
            {
                int idx = i + j * NX + k * NX * NY;
                
                // Inlet
                if (i == 0)
                {
                    h_u[idx] = INLET_VELOCITY;
                    h_v[idx] = 0.0f;
                    h_w[idx] = 0.0f;
                }
                // Outlet
                else if (i == NX-1)
                {
                    h_u[idx] = h_u[idx-1];
                    h_v[idx] = h_v[idx-1];
                    h_w[idx] = h_w[idx-1];
                }
                // Interior with perturbations
                else
                {
                    float perturbation = 0.1f * (2.0f * static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 1.0f);
                    h_u[idx] = INLET_VELOCITY * (1.0f + perturbation);
                    h_v[idx] = 0.05f * INLET_VELOCITY * (2.0f * rand() / (float)RAND_MAX - 1.0f);
                    h_w[idx] = 0.05f * INLET_VELOCITY * (2.0f * rand() / (float)RAND_MAX - 1.0f);
                }
            }
        }
    }

    CUDA_CHECK(cudaMemcpy(flow.u, h_u, size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(flow.v, h_v, size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(flow.w, h_w, size * sizeof(float), cudaMemcpyHostToDevice));

    // Delete unneeded temp arrays
    delete[] h_u;
    delete[] h_v;
    delete[] h_w;

    // Initialize k and epsilon
    initializeFields<<<grid, block>>>(flow.k, flow.epsilon, flow.u, flow.v, flow.w,
                                      NX, NY, NZ);

    // Initialize turbulent viscosity
    calculateTurbulentViscosity<<<grid, block>>>(flow.nut, flow.k, flow.epsilon, size);

    printf("Initialization complete. Starting time stepping...\n");

    // Synchronize all CUDA threads to ensure initialization is complete
    cudaDeviceSynchronize();

    // MARK: timestepping
    for (int step = 0; step < (MAX_ITER+1); step++)
    {
        // Copy old values
        storeOldValues<<<grid, block>>>(u_old, v_old, w_old, k_old, eps_old, flow.u, flow.v, flow.w, flow.k, flow.epsilon, size);

        cudaDeviceSynchronize();
        CUDA_CHECK(cudaGetLastError());

        // Copy monitoring points
        CUDA_CHECK(cudaMemcpy(d_u_monitor_prev, d_u_monitor_curr, num_monitor_points * sizeof(float), cudaMemcpyDeviceToDevice));

        // Zero out residuals for this step
        cudaMemset(flow.residuals, 0, 5 * sizeof(float));

        // MARK: cfl step
        // Calculate maximum CFL
        cudaMemset(d_max_cfl, 0, sizeof(float));
        calculateMaxCFL<<<grid, block>>>(d_max_cfl, flow.u, flow.v, flow.w, size);

        float current_cfl;
        cudaMemcpy(&current_cfl, d_max_cfl, sizeof(float), cudaMemcpyDeviceToHost);
        FloatInt converter;
        converter.i = *(unsigned int *)&current_cfl;
        current_cfl = converter.f;

        // Adjust timestep based on CFL
        if (current_cfl > 1e-6f)
        {
            // Target CFL is 1/2 for stability
            // dt = 1/2 * dx / cfl
            dt = DX / (2 * current_cfl);
            // Clamp dt to min and max values
            dt = fmaxf(dt, MIN_DT);
            dt = fminf(dt, MAX_DT);
        }


        // MARK: momentum
        solveXMomentum<<<grid, block>>>(u_new, flow.u, flow.v, flow.w, flow.p, flow.nut, flow.k, flow.epsilon, NX, NY, NZ, dt);

        cudaDeviceSynchronize();
        CUDA_CHECK(cudaGetLastError());

        solveYMomentum<<<grid, block>>>(v_new, flow.u, flow.v, flow.w, flow.p, flow.nut, flow.k, flow.epsilon, NX, NY, NZ, dt);

        cudaDeviceSynchronize();
        CUDA_CHECK(cudaGetLastError());

        solveZMomentum<<<grid, block>>>(w_new, flow.u, flow.v, flow.w, flow.p, flow.nut, flow.k, flow.epsilon, NX, NY, NZ, dt);

        cudaDeviceSynchronize();
        CUDA_CHECK(cudaGetLastError());

        // Apply boundary conditions
        applyBoundaryConditions<<<grid, block>>>(u_new, v_new, w_new, flow.p, flow.k, flow.epsilon, NX, NY, NZ);

        // MARK: SIMPLEC
        if (SIMPLEC_ENABLED == true){
            for (int iter = 0; iter < MAX_PRESSURE_ITER; iter++)
            {
                // Calculate divergence
                computeDivergence<<<grid, block>>>(div, u_new, v_new, w_new, NX, NY, NZ);

                // Previous pressure correction
                float *p_corr_old;
                CUDA_CHECK(cudaMalloc(&p_corr_old, size * sizeof(float)));
                cudaMemcpy(p_corr_old, p_corr, size * sizeof(float), cudaMemcpyDeviceToDevice);

                for (int gs_iter = 0; gs_iter < 5; gs_iter++)
                {
                    calculatePressureCorrection<<<grid, block>>>(p_corr, div, ap, NX, NY, NZ);
                    cudaDeviceSynchronize();
                }

                // Under-relax pressure correction using kernel
                underRelaxPressureCorrection<<<grid, block>>>(p_corr, p_corr_old, ALPHA_P, size);

                correctVelocities<<<grid, block>>>(u_new, v_new, w_new, p_corr, ap, NX, NY, NZ);

                // Update pressure field
                updatePressure<<<grid, block>>>(flow.p, p_corr, ALPHA_P, size);

                cudaFree(p_corr_old);

                if (iter < MIN_PRESSURE_ITER)
                    continue;
            }
        }
        
        // MARK: k-epsilon
        solveKEquation<<<grid, block>>>(k_new, flow.k, flow.epsilon, flow.u,
                                        flow.v, flow.w, flow.nut, NX, NY, NZ);
        solveEpsilonEquation<<<grid, block>>>(eps_new, flow.k, flow.epsilon,
                                              flow.u, flow.v, flow.w, flow.nut, NX, NY, NZ);

        // Update fields
        cudaMemcpy(flow.u, u_new, size * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(flow.v, v_new, size * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(flow.w, w_new, size * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(flow.k, k_new, size * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(flow.epsilon, eps_new, size * sizeof(float), cudaMemcpyDeviceToDevice);

        calculateTurbulentViscosity<<<grid, block>>>(flow.nut, flow.k, flow.epsilon, size);

        // Extract current values at monitoring points
        int threads_per_block = 128;
        int blocks_per_grid = (num_monitor_points + threads_per_block - 1) / threads_per_block;

        extractMonitoringValues<<<blocks_per_grid, threads_per_block>>>(flow.u, d_u_monitor_curr, d_monitor_indices, num_monitor_points);

        // MARK: residuals
        calculateResiduals<<<grid, block>>>(flow.residuals,
                                            flow.u, u_old,
                                            flow.v, v_old,
                                            flow.w, w_old,
                                            flow.k, k_old,
                                            flow.epsilon, eps_old,
                                            size);

        // Calculate monitoring residuals
        computeMonitoringResiduals<<<blocks_per_grid, threads_per_block>>>(d_monitor_residuals, d_u_monitor_curr, d_u_monitor_prev, num_monitor_points);

        float h_monitor_residuals[num_monitor_points];
        CUDA_CHECK(cudaMemcpy(h_monitor_residuals, d_monitor_residuals, num_monitor_points * sizeof(float), cudaMemcpyDeviceToHost));

        // // Output residuals at monitoring points
        // printf("Monitoring point residuals at step %d:\n", step);
        // for (int n = 0; n < num_monitor_points; n++)
        // {
        //     printf("Point (%d, %d, %d): Residual = %.6e\n",
        //         monitor_coords[n][0], monitor_coords[n][1], monitor_coords[n][2],
        //         h_monitor_residuals[n]);
        // }

        // MARK: convergence
        float h_residuals[5];
        cudaMemcpy(h_residuals, flow.residuals, 5 * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < 5; i++)
        {
            FloatInt converter;
            converter.i = *(unsigned int *)&h_residuals[i];
            h_residuals[i] = converter.f;
        }

        // Normalize and check residuals
        bool converged = (step > MIN_ITER) &&
                         (h_residuals[0] < RESIDUAL_TOL) &&
                         (h_residuals[1] < RESIDUAL_TOL) &&
                         (h_residuals[2] < RESIDUAL_TOL) &&
                         (h_residuals[3] < RESIDUAL_TOL) &&
                         (h_residuals[4] < RESIDUAL_TOL);

        if (converged)
        {
            static float prev_residual_sum = FLT_MAX;
            float residual_sum = 0;
            for (int i = 0; i < 5; i++)
            {
                residual_sum += h_residuals[i];
            }

            if (fabsf(residual_sum - prev_residual_sum) / fmaxf(residual_sum, 1e-10f) < 1e-3f)
            {
                printf("Converged at step %d\n", step);
                break;
            }
            prev_residual_sum = residual_sum;
        }

        // MARK: reporting
        if (step % 100 == 0)
        {
            printf("Step #: Residuals = u v w k epsilon\n");
            printf("Step %d: Residuals = %.2e %.2e %.2e %.2e %.2e\n",
               step, h_residuals[0], h_residuals[1], h_residuals[2],
               h_residuals[3], h_residuals[4]);
            printf("Max CFL = %.2f, dt = %.2e\n\n", current_cfl, dt);
            saveFieldData(&flow, step, NX, NY, NZ);
        }
    }
    printf("Simulation complete\n");

    // Cleanup
    cudaFree(u_old);
    cudaFree(v_old);
    cudaFree(w_old);
    cudaFree(k_old);
    cudaFree(eps_old);
    cudaFree(d_max_cfl);

    return 0;
}