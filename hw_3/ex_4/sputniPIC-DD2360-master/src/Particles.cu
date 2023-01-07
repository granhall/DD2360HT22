#include "Particles.h"
#include "Alloc.h"
#include <cuda.h>
#include <cuda_runtime.h>
#define TPB 256

/** allocate particle arrays */
void particle_allocate(struct parameters* param, struct particles* part, int is)
{
    
    // set species ID
    part->species_ID = is;
    // number of particles
    part->nop = param->np[is];
    // maximum number of particles
    part->npmax = param->npMax[is];
    
    // choose a different number of mover iterations for ions and electrons
    if (param->qom[is] < 0){  //electrons
        part->NiterMover = param->NiterMover;
        part->n_sub_cycles = param->n_sub_cycles;
    } else {                  // ions: only one iteration
        part->NiterMover = 1;
        part->n_sub_cycles = 1;
    }
    
    // particles per cell
    part->npcelx = param->npcelx[is];
    part->npcely = param->npcely[is];
    part->npcelz = param->npcelz[is];
    part->npcel = part->npcelx*part->npcely*part->npcelz;
    
    // cast it to required precision
    part->qom = (FPpart) param->qom[is];
    
    long npmax = part->npmax;
    
    // initialize drift and thermal velocities
    // drift
    part->u0 = (FPpart) param->u0[is];
    part->v0 = (FPpart) param->v0[is];
    part->w0 = (FPpart) param->w0[is];
    // thermal
    part->uth = (FPpart) param->uth[is];
    part->vth = (FPpart) param->vth[is];
    part->wth = (FPpart) param->wth[is];
    
    
    //////////////////////////////
    /// ALLOCATION PARTICLE ARRAYS
    //////////////////////////////
    part->x = new FPpart[npmax];
    part->y = new FPpart[npmax];
    part->z = new FPpart[npmax];
    // allocate velocity
    part->u = new FPpart[npmax];
    part->v = new FPpart[npmax];
    part->w = new FPpart[npmax];
    // allocate charge = q * statistical weight
    part->q = new FPinterp[npmax];
    
}
/** deallocate */
void particle_deallocate(struct particles* part)
{
    // deallocate particle variables
    delete[] part->x;
    delete[] part->y;
    delete[] part->z;
    delete[] part->u;
    delete[] part->v;
    delete[] part->w;
    delete[] part->q;
}

/** particle mover */
int mover_PC(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param)
{
    // print species and subcycling
    std::cout << "***  MOVER with SUBCYCLYING "<< param->n_sub_cycles << " - species " << part->species_ID << " ***" << std::endl;
 
    // auxiliary variables
    FPpart dt_sub_cycling = (FPpart) param->dt/((double) part->n_sub_cycles);
    FPpart dto2 = .5*dt_sub_cycling, qomdt2 = part->qom*dto2/param->c;
    FPpart omdtsq, denom, ut, vt, wt, udotb;
    
    // local (to the particle) electric and magnetic field
    FPfield Exl=0.0, Eyl=0.0, Ezl=0.0, Bxl=0.0, Byl=0.0, Bzl=0.0;
    
    // interpolation densities
    int ix,iy,iz;
    FPfield weight[2][2][2];
    FPfield xi[2], eta[2], zeta[2];
    
    // intermediate particle position and velocity
    FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;
    
    // start subcycling
    for (int i_sub=0; i_sub <  part->n_sub_cycles; i_sub++){
        // move each particle with new fields
        for (int i=0; i <  part->nop; i++){
            xptilde = part->x[i];
            yptilde = part->y[i];
            zptilde = part->z[i];
            // calculate the average velocity iteratively
            for(int innter=0; innter < part->NiterMover; innter++){
                // interpolation G-->P
                ix = 2 +  int((part->x[i] - grd->xStart)*grd->invdx);
                iy = 2 +  int((part->y[i] - grd->yStart)*grd->invdy);
                iz = 2 +  int((part->z[i] - grd->zStart)*grd->invdz);
                
                // calculate weights
                xi[0]   = part->x[i] - grd->XN[ix - 1][iy][iz];
                eta[0]  = part->y[i] - grd->YN[ix][iy - 1][iz];
                zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
                xi[1]   = grd->XN[ix][iy][iz] - part->x[i];
                eta[1]  = grd->YN[ix][iy][iz] - part->y[i];
                zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];
                for (int ii = 0; ii < 2; ii++)
                    for (int jj = 0; jj < 2; jj++)
                        for (int kk = 0; kk < 2; kk++)
                            weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
                
                // set to zero local electric and magnetic field
                Exl=0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;
                
                for (int ii=0; ii < 2; ii++)
                    for (int jj=0; jj < 2; jj++)
                        for(int kk=0; kk < 2; kk++){
                            Exl += weight[ii][jj][kk]*field->Ex[ix- ii][iy -jj][iz- kk ];
                            Eyl += weight[ii][jj][kk]*field->Ey[ix- ii][iy -jj][iz- kk ];
                            Ezl += weight[ii][jj][kk]*field->Ez[ix- ii][iy -jj][iz -kk ];
                            Bxl += weight[ii][jj][kk]*field->Bxn[ix- ii][iy -jj][iz -kk ];
                            Byl += weight[ii][jj][kk]*field->Byn[ix- ii][iy -jj][iz -kk ];
                            Bzl += weight[ii][jj][kk]*field->Bzn[ix- ii][iy -jj][iz -kk ];
                        }
                
                // end interpolation
                omdtsq = qomdt2*qomdt2*(Bxl*Bxl+Byl*Byl+Bzl*Bzl);
                denom = 1.0/(1.0 + omdtsq);
                // solve the position equation
                ut= part->u[i] + qomdt2*Exl;
                vt= part->v[i] + qomdt2*Eyl;
                wt= part->w[i] + qomdt2*Ezl;
                udotb = ut*Bxl + vt*Byl + wt*Bzl;
                // solve the velocity equation
                uptilde = (ut+qomdt2*(vt*Bzl -wt*Byl + qomdt2*udotb*Bxl))*denom;
                vptilde = (vt+qomdt2*(wt*Bxl -ut*Bzl + qomdt2*udotb*Byl))*denom;
                wptilde = (wt+qomdt2*(ut*Byl -vt*Bxl + qomdt2*udotb*Bzl))*denom;
                // update position
                part->x[i] = xptilde + uptilde*dto2;
                part->y[i] = yptilde + vptilde*dto2;
                part->z[i] = zptilde + wptilde*dto2;
                
                
            } // end of iteration
            // update the final position and velocity
            part->u[i]= 2.0*uptilde - part->u[i];
            part->v[i]= 2.0*vptilde - part->v[i];
            part->w[i]= 2.0*wptilde - part->w[i];
            part->x[i] = xptilde + uptilde*dt_sub_cycling;
            part->y[i] = yptilde + vptilde*dt_sub_cycling;
            part->z[i] = zptilde + wptilde*dt_sub_cycling;
            
            
            //////////
            //////////
            ////////// BC
                                        
            // X-DIRECTION: BC particles
            if (part->x[i] > grd->Lx){
                if (param->PERIODICX==true){ // PERIODIC
                    part->x[i] = part->x[i] - grd->Lx;
                } else { // REFLECTING BC
                    part->u[i] = -part->u[i];
                    part->x[i] = 2*grd->Lx - part->x[i];
                }
            }
                                                                        
            if (part->x[i] < 0){
                if (param->PERIODICX==true){ // PERIODIC
                    part->x[i] = part->x[i] + grd->Lx;
                } else { // REFLECTING BC
                    part->u[i] = -part->u[i];
                    part->x[i] = -part->x[i];
                }
            }
                
            
            // Y-DIRECTION: BC particles
            if (part->y[i] > grd->Ly){
                if (param->PERIODICY==true){ // PERIODIC
                    part->y[i] = part->y[i] - grd->Ly;
                } else { // REFLECTING BC
                    part->v[i] = -part->v[i];
                    part->y[i] = 2*grd->Ly - part->y[i];
                }
            }
                                                                        
            if (part->y[i] < 0){
                if (param->PERIODICY==true){ // PERIODIC
                    part->y[i] = part->y[i] + grd->Ly;
                } else { // REFLECTING BC
                    part->v[i] = -part->v[i];
                    part->y[i] = -part->y[i];
                }
            }
                                                                        
            // Z-DIRECTION: BC particles
            if (part->z[i] > grd->Lz){
                if (param->PERIODICZ==true){ // PERIODIC
                    part->z[i] = part->z[i] - grd->Lz;
                } else { // REFLECTING BC
                    part->w[i] = -part->w[i];
                    part->z[i] = 2*grd->Lz - part->z[i];
                }
            }
                                                                        
            if (part->z[i] < 0){
                if (param->PERIODICZ==true){ // PERIODIC
                    part->z[i] = part->z[i] + grd->Lz;
                } else { // REFLECTING BC
                    part->w[i] = -part->w[i];
                    part->z[i] = -part->z[i];
                }
            }
                                                                        
            
            
        }  // end of subcycling
    } // end of one particle
                                                                        
    return(0); // exit succcesfully
} // end of the mover

/** Particle Mover GPU Variant */

__global__ void particle_mover(particles* gpupart, EMfield* gpufield, grid* gpugrd, parameters* gpuparam){
    
    // index of the particle
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if( i >= gpupart->nop) return;

    //printf("%f %f %f \n",(gpupart->x[0]), (gpupart->y[0]), (gpupart->z[0]));
    
    //auxiliary variables
    FPpart dt_sub_cycling = (FPpart) gpuparam->dt/((double) gpupart->n_sub_cycles);
    FPpart dto2 = .5*dt_sub_cycling, qomdt2 = gpupart->qom*dto2/gpuparam->c;
    FPpart omdtsq, denom, ut, vt, wt, udotb;

    // local (to the particle) electric and magnetic field
    FPfield Exl=0.0, Eyl=0.0, Ezl=0.0, Bxl=0.0, Byl=0.0, Bzl=0.0;

    // interpolation densities
    int ix,iy,iz;
    FPfield weight[2][2][2];
    FPfield xi[2], eta[2], zeta[2];
    
    // intermediate particle position and velocity
    FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;
    
    //printf("%f %f %f \n",(gpupart->x[i]), (gpupart->y[i]), (gpupart->z[i]));

    // loop over all particles
        for(int i_sub=0; i_sub < gpupart->n_sub_cycles; i_sub++){
            xptilde = gpupart->x[i];
            yptilde = gpupart->y[i];
            zptilde = gpupart->z[i];

            // calculate the average velocity iteratively
            for(int innter=0; innter < gpupart->NiterMover; innter++){
                // interpolation G-->P
                ix = 2 +  int((gpupart->x[i] - gpugrd->xStart)*gpugrd->invdx);
                iy = 2 +  int((gpupart->y[i] - gpugrd->yStart)*gpugrd->invdy);
                iz = 2 +  int((gpupart->z[i] - gpugrd->zStart)*gpugrd->invdz);
                
                // calculate weights
                xi[0]   = gpupart->x[i] - gpugrd->XN_flat[get_idx(ix - 1, iy, iz, gpugrd->nyn, gpugrd->nzn)];
                eta[0]  = gpupart->y[i] - gpugrd->YN_flat[get_idx(ix, iy - 1, iz, gpugrd->nyn, gpugrd->nzn)];
                zeta[0] = gpupart->z[i] - gpugrd->ZN_flat[get_idx(ix, iy, iz - 1, gpugrd->nyn, gpugrd->nzn)];
                xi[1]   = gpugrd->XN_flat[get_idx(ix, iy, iz, gpugrd-> nyn, gpugrd-> nzn)] - gpupart->x[i];
                eta[1]  = gpugrd->YN_flat[get_idx(ix, iy, iz, gpugrd-> nyn, gpugrd-> nzn)] - gpupart->y[i];
                zeta[1] = gpugrd->ZN_flat[get_idx(ix, iy, iz, gpugrd-> nyn, gpugrd-> nzn)] - gpupart->z[i];

                for (int ii = 0; ii < 2; ii++)
                    for (int jj = 0; jj < 2; jj++)
                        for (int kk = 0; kk < 2; kk++)
                            weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * gpugrd->invVOL;
                
                // set to zero local electric and magnetic field
                Exl=0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;
                
                for (int ii=0; ii < 2; ii++)
                    for (int jj=0; jj < 2; jj++)
                        for(int kk=0; kk < 2; kk++){
                            Exl += weight[ii][jj][kk]*gpufield->Ex_flat[get_idx(ix- ii, iy-jj, iz-kk, gpugrd->nyn, gpugrd->nzn)];
                            Eyl += weight[ii][jj][kk]*gpufield->Ey_flat[get_idx(ix- ii, iy-jj, iz-kk, gpugrd->nyn, gpugrd->nzn)];
                            Ezl += weight[ii][jj][kk]*gpufield->Ez_flat[get_idx(ix- ii, iy-jj, iz-kk, gpugrd->nyn, gpugrd->nzn)];
                            Bxl += weight[ii][jj][kk]*gpufield->Bxn_flat[get_idx(ix- ii, iy-jj, iz-kk, gpugrd->nyn, gpugrd->nzn)];
                            Byl += weight[ii][jj][kk]*gpufield->Byn_flat[get_idx(ix- ii, iy-jj, iz-kk, gpugrd->nyn, gpugrd->nzn)];
                            Bzl += weight[ii][jj][kk]*gpufield->Bzn_flat[get_idx(ix- ii, iy-jj, iz-kk, gpugrd->nyn, gpugrd->nzn)];

                        }
                
                // end interpolation
                omdtsq = qomdt2*qomdt2*(Bxl*Bxl+Byl*Byl+Bzl*Bzl);
                denom = 1.0/(1.0 + omdtsq);
                // solve the position equation
                ut= gpupart->u[i] + qomdt2*Exl;
                vt= gpupart->v[i] + qomdt2*Eyl;
                wt= gpupart->w[i] + qomdt2*Ezl;
                udotb = ut*Bxl + vt*Byl + wt*Bzl;
                // solve the velocity equation
                uptilde = (ut+qomdt2*(vt*Bzl -wt*Byl + qomdt2*udotb*Bxl))*denom;
                vptilde = (vt+qomdt2*(wt*Bxl -ut*Bzl + qomdt2*udotb*Byl))*denom;
                wptilde = (wt+qomdt2*(ut*Byl -vt*Bxl + qomdt2*udotb*Bzl))*denom;
                // update position
                gpupart->x[i] = xptilde + uptilde*dto2;
                gpupart->y[i] = yptilde + vptilde*dto2;
                gpupart->z[i] = zptilde + wptilde*dto2;
                
                
            } // end of iteration
            // update the final position and velocity
            gpupart->u[i]= 2.0*uptilde - gpupart->u[i];
            gpupart->v[i]= 2.0*vptilde - gpupart->v[i];
            gpupart->w[i]= 2.0*wptilde - gpupart->w[i];
            gpupart->x[i] = xptilde + uptilde*dt_sub_cycling;
            gpupart->y[i] = yptilde + vptilde*dt_sub_cycling;
            gpupart->z[i] = zptilde + wptilde*dt_sub_cycling;
            
            
            //////////
            //////////
            ////////// BC
                                        
            // X-DIRECTION: BC particles
            if (gpupart->x[i] > gpugrd->Lx){
                if (gpuparam->PERIODICX==true){ // PERIODIC
                    gpupart->x[i] = gpupart->x[i] - gpugrd->Lx;
                } else { // REFLECTING BC
                    gpupart->u[i] = -gpupart->u[i];
                    gpupart->x[i] = 2*gpugrd->Lx - gpupart->x[i];
                }
            }
                                                                        
            if (gpupart->x[i] < 0){
                if (gpuparam->PERIODICX==true){ // PERIODIC
                    gpupart->x[i] = gpupart->x[i] + gpugrd->Lx;
                } else { // REFLECTING BC
                    gpupart->u[i] = -gpupart->u[i];
                    gpupart->x[i] = -gpupart->x[i];
                }
            }
                
            
            // Y-DIRECTION: BC particles
            if (gpupart->y[i] > gpugrd->Ly){
                if (gpuparam->PERIODICY==true){ // PERIODIC
                    gpupart->y[i] = gpupart->y[i] - gpugrd->Ly;
                } else { // REFLECTING BC
                    gpupart->v[i] = -gpupart->v[i];
                    gpupart->y[i] = 2*gpugrd->Ly - gpupart->y[i];
                }
            }
                                                                        
            if (gpupart->y[i] < 0){
                if (gpuparam->PERIODICY==true){ // PERIODIC
                    gpupart->y[i] = gpupart->y[i] + gpugrd->Ly;
                } else { // REFLECTING BC
                    gpupart->v[i] = -gpupart->v[i];
                    gpupart->y[i] = -gpupart->y[i];
                }
            }
                                                                        
            // Z-DIRECTION: BC particles
            if (gpupart->z[i] > gpugrd->Lz){
                if (gpuparam->PERIODICZ==true){ // PERIODIC
                    gpupart->z[i] = gpupart->z[i] - gpugrd->Lz;
                } else { // REFLECTING BC
                    gpupart->w[i] = -gpupart->w[i];
                    gpupart->z[i] = 2*gpugrd->Lz - gpupart->z[i];
                }
            }
                                                                        
            if (gpupart->z[i] < 0){
                if (gpuparam->PERIODICZ==true){ // PERIODIC
                    gpupart->z[i] = gpupart->z[i] + gpugrd->Lz;
                } else { // REFLECTING BC
                    gpupart->w[i] = -gpupart->w[i];
                    gpupart->z[i] = -gpupart->z[i];
                }
            }
        }  // end of subcycling
    } // end of one particle

/** Mover_PC_gpu Variant */
int mover_PC_GPU(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param){
    particles* gpupart;
    EMfield* gpufield;
    grid* gpugrd;
    parameters* gpuparam;
    
    std::cout << "***  MOVER with SUBCYCLYING "<< param->n_sub_cycles << " - species " << part->species_ID << " ***" << std::endl;
    
    cudaMalloc((void**)&gpupart, sizeof(particles));
    cudaMalloc((void**)&gpufield, sizeof(EMfield));
    cudaMalloc((void**)&gpugrd, sizeof(grid));
    cudaMalloc((void**)&gpuparam, sizeof(parameters));

    cudaMemcpy(gpupart, part, sizeof(particles), cudaMemcpyHostToDevice);
    cudaMemcpy(gpufield, field, sizeof(EMfield), cudaMemcpyHostToDevice);
    cudaMemcpy(gpugrd, grd, sizeof(grid), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuparam, param, sizeof(parameters), cudaMemcpyHostToDevice);

    FPpart *gpupart_x, *gpupart_y, *gpupart_z, *gpupart_u, *gpupart_v, *gpupart_w;

    cudaMalloc((void**)&gpupart_x, part->npmax*sizeof(FPpart));
    cudaMalloc((void**)&gpupart_y, part->npmax*sizeof(FPpart));
    cudaMalloc((void**)&gpupart_z, part->npmax*sizeof(FPpart));
    cudaMalloc((void**)&gpupart_u, part->npmax*sizeof(FPpart));
    cudaMalloc((void**)&gpupart_v, part->npmax*sizeof(FPpart));
    cudaMalloc((void**)&gpupart_w, part->npmax*sizeof(FPpart));

    //This part took time to figure out, pointers are not copied above so need to set it correctly.
    cudaMemcpy(&(gpupart->x), &gpupart_x, sizeof(FPpart*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(gpupart->y), &gpupart_y, sizeof(FPpart*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(gpupart->z), &gpupart_z, sizeof(FPpart*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(gpupart->u), &gpupart_u, sizeof(FPpart*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(gpupart->v), &gpupart_v, sizeof(FPpart*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(gpupart->w), &gpupart_w, sizeof(FPpart*), cudaMemcpyHostToDevice);

    cudaMemcpy(gpupart_x, part->x, part->npmax*sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(gpupart_y, part->y, part->npmax*sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(gpupart_z, part->z, part->npmax*sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(gpupart_u, part->u, part->npmax*sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(gpupart_v, part->v, part->npmax*sizeof(FPpart), cudaMemcpyHostToDevice);
    cudaMemcpy(gpupart_w, part->w, part->npmax*sizeof(FPpart), cudaMemcpyHostToDevice);

    
    FPfield *gpugrd_XN_flat, *gpugrd_YN_flat, *gpugrd_ZN_flat;
    cudaMalloc((void**)&gpugrd_XN_flat, grd->nxn*grd->nyn*grd->nzn*sizeof(FPfield));
    cudaMalloc((void**)&gpugrd_YN_flat, grd->nxn*grd->nyn*grd->nzn*sizeof(FPfield));
    cudaMalloc((void**)&gpugrd_ZN_flat, grd->nxn*grd->nyn*grd->nzn*sizeof(FPfield));

    //This part took time to figure out, pointers are not copied above so need to set it correctly.
    cudaMemcpy(&(gpugrd->XN_flat), &gpugrd_XN_flat, sizeof(FPfield*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(gpugrd->YN_flat), &gpugrd_YN_flat, sizeof(FPfield*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(gpugrd->ZN_flat), &gpugrd_ZN_flat, sizeof(FPfield*), cudaMemcpyHostToDevice);
    
    cudaMemcpy(gpugrd_XN_flat, grd->XN_flat, grd->nxn*grd->nyn*grd->nzn*sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(gpugrd_YN_flat, grd->YN_flat, grd->nxn*grd->nyn*grd->nzn*sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(gpugrd_ZN_flat, grd->ZN_flat, grd->nxn*grd->nyn*grd->nzn*sizeof(FPfield), cudaMemcpyHostToDevice);
    
    FPfield *gpufield_Ex_flat, *gpufield_Ey_flat, *gpufield_Ez_flat, *gpufield_Bxn_flat, *gpufield_Byn_flat, *gpufield_Bzn_flat;
    cudaMalloc((void**)&gpufield_Ex_flat, grd->nxn*grd->nyn*grd->nzn*sizeof(FPfield));
    cudaMalloc((void**)&gpufield_Ey_flat, grd->nxn*grd->nyn*grd->nzn*sizeof(FPfield));
    cudaMalloc((void**)&gpufield_Ez_flat, grd->nxn*grd->nyn*grd->nzn*sizeof(FPfield));
    cudaMalloc((void**)&gpufield_Bxn_flat, grd->nxn*grd->nyn*grd->nzn*sizeof(FPfield));
    cudaMalloc((void**)&gpufield_Byn_flat, grd->nxn*grd->nyn*grd->nzn*sizeof(FPfield));
    cudaMalloc((void**)&gpufield_Bzn_flat, grd->nxn*grd->nyn*grd->nzn*sizeof(FPfield));

    //This part took time to figure out, pointers are not copied above so need to set it correctly.
    cudaMemcpy(&(gpufield->Ex_flat), &gpufield_Ex_flat, sizeof(FPfield*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(gpufield->Ey_flat), &gpufield_Ey_flat, sizeof(FPfield*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(gpufield->Ez_flat), &gpufield_Ez_flat, sizeof(FPfield*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(gpufield->Bxn_flat), &gpufield_Bxn_flat, sizeof(FPfield*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(gpufield->Byn_flat), &gpufield_Byn_flat, sizeof(FPfield*), cudaMemcpyHostToDevice);
    cudaMemcpy(&(gpufield->Bzn_flat), &gpufield_Bzn_flat, sizeof(FPfield*), cudaMemcpyHostToDevice);

    cudaMemcpy(gpufield_Ex_flat, field->Ex_flat, grd->nxn*grd->nyn*grd->nzn*sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(gpufield_Ey_flat, field->Ey_flat, grd->nxn*grd->nyn*grd->nzn*sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(gpufield_Ez_flat, field->Ez_flat, grd->nxn*grd->nyn*grd->nzn*sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(gpufield_Bxn_flat, field->Bxn_flat, grd->nxn*grd->nyn*grd->nzn*sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(gpufield_Byn_flat, field->Byn_flat, grd->nxn*grd->nyn*grd->nzn*sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(gpufield_Bzn_flat, field->Bzn_flat, grd->nxn*grd->nyn*grd->nzn*sizeof(FPfield), cudaMemcpyHostToDevice);


    int gridDim = (part->nop + TPB - 1) / TPB;

    particle_mover<<<gridDim, TPB>>>(gpupart, gpufield, gpugrd, gpuparam);
    cudaDeviceSynchronize();

    FPpart *pcpart_x, *pcpart_y, *pcpart_z, *pcpart_u, *pcpart_v, *pcpart_w;
    pcpart_x = (FPpart*)malloc(part->npmax*sizeof(FPpart));
    pcpart_y = (FPpart*)malloc(part->npmax*sizeof(FPpart));
    pcpart_z = (FPpart*)malloc(part->npmax*sizeof(FPpart));
    pcpart_u = (FPpart*)malloc(part->npmax*sizeof(FPpart));
    pcpart_v = (FPpart*)malloc(part->npmax*sizeof(FPpart));
    pcpart_w = (FPpart*)malloc(part->npmax*sizeof(FPpart));

    cudaMemcpy(part, gpupart, sizeof(particles), cudaMemcpyDeviceToHost);
    cudaMemcpy(pcpart_x, gpupart_x, part->npmax*sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(pcpart_y, gpupart_y, part->npmax*sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(pcpart_z, gpupart_z, part->npmax*sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(pcpart_u, gpupart_u, part->npmax*sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(pcpart_v, gpupart_v, part->npmax*sizeof(FPpart), cudaMemcpyDeviceToHost);
    cudaMemcpy(pcpart_w, gpupart_w, part->npmax*sizeof(FPpart), cudaMemcpyDeviceToHost);

    
    //cudaMemcpy(part->x, gpupart_x, part->npmax*sizeof(FPpart), cudaMemcpyDeviceToHost);
    //cudaMemcpy(part->y, gpupart_y, part->npmax*sizeof(FPpart), cudaMemcpyDeviceToHost);
    //cudaMemcpy(part->z, gpupart_z, part->npmax*sizeof(FPpart), cudaMemcpyDeviceToHost);
    //cudaMemcpy(part->u, gpupart_u, part->npmax*sizeof(FPpart), cudaMemcpyDeviceToHost);
    //cudaMemcpy(part->v, gpupart_v, part->npmax*sizeof(FPpart), cudaMemcpyDeviceToHost);
    //cudaMemcpy(part->w, gpupart_w, part->npmax*sizeof(FPpart), cudaMemcpyDeviceToHost);

    //cudaMemcpy(part, gpupart, sizeof(struct particles), cudaMemcpyDeviceToHost);
    //cudaMemcpy(field, gpufield, sizeof(struct EMfield), cudaMemcpyDeviceToHost);
    //cudaMemcpy(grd, gpugrd, sizeof(struct grid), cudaMemcpyDeviceToHost);
    //cudaMemcpy(param, gpuparam, sizeof(struct parameters), cudaMemcpyDeviceToHost);

    part->x = pcpart_x;
    part->y = pcpart_y;
    part->z = pcpart_z;
    part->u = pcpart_u;
    part->v = pcpart_v;
    part->w = pcpart_w;

    cudaFree(gpupart);
    cudaFree(gpufield);
    cudaFree(gpugrd);
    cudaFree(gpuparam);
    cudaFree(gpupart_x);
    cudaFree(gpupart_y);
    cudaFree(gpupart_z);
    cudaFree(gpupart_u);
    cudaFree(gpupart_v);
    cudaFree(gpupart_w);
    cudaFree(gpufield_Ex_flat);
    cudaFree(gpufield_Ey_flat);
    cudaFree(gpufield_Ez_flat);
    cudaFree(gpufield_Bxn_flat);
    cudaFree(gpufield_Byn_flat);
    cudaFree(gpufield_Bzn_flat);

    return(0);
}

/** Interpolation Particle --> Grid: This is for species */
void interpP2G(struct particles* part, struct interpDensSpecies* ids, struct grid* grd)
{
    
    // arrays needed for interpolation
    FPpart weight[2][2][2];
    FPpart temp[2][2][2];
    FPpart xi[2], eta[2], zeta[2];
    
    // index of the cell
    int ix, iy, iz;
    
    
    for (register long long i = 0; i < part->nop; i++) {
        
        // determine cell: can we change to int()? is it faster?
        ix = 2 + int (floor((part->x[i] - grd->xStart) * grd->invdx));
        iy = 2 + int (floor((part->y[i] - grd->yStart) * grd->invdy));
        iz = 2 + int (floor((part->z[i] - grd->zStart) * grd->invdz));
        
        // distances from node
        xi[0]   = part->x[i] - grd->XN[ix - 1][iy][iz];
        eta[0]  = part->y[i] - grd->YN[ix][iy - 1][iz];
        zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
        xi[1]   = grd->XN[ix][iy][iz] - part->x[i];
        eta[1]  = grd->YN[ix][iy][iz] - part->y[i];
        zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];
        
        // calculate the weights for different nodes
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    weight[ii][jj][kk] = part->q[i] * xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
        
        //////////////////////////
        // add charge density
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->rhon[ix - ii][iy - jj][iz - kk] += weight[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add current density - Jx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * weight[ii][jj][kk];
        
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add current density - Jy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        
        ////////////////////////////
        // add current density - Jz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add pressure pxx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->u[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add pressure pxy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        
        /////////////////////////////
        // add pressure pxz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pyy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pyy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pyz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pyz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pzz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii=0; ii < 2; ii++)
            for (int jj=0; jj < 2; jj++)
                for(int kk=0; kk < 2; kk++)
                    ids->pzz[ix -ii][iy -jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
    
    }
}
