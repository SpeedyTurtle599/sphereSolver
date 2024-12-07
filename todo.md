1. Figure out why velocity field does not propagate forward in time
2. Revise SIMPLEC algorithm to show debugging output
3. Check boundary conditions for pressure field
4. Check code for consistency with constant values usage -- ensure no hardcoded values wherever possible

5. Check `momentum` functions for correct indexing
6. Ensure u_new[idx], v_new[idx], w_new[idx] are updated for all idx
7. Confirm that updated fields are correctly copied with `cudaMemcpy`