//Author: S.S. Khrapov
//Parallel Nbody Code OpenMP-CUDA 4GPU

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdlib>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "math_functions.h"

#define PI 3.14159265358979
#define BLOCK_SIZE 512
#define BLOCK_SIZE_b 512


#define real double
#define real2 double2
#define real3 double3
#define real4 double4
#define make_real3 make_double3
#define make_real4 make_double4
#define make_real2 make_double2

real Z_max, E0;
real3 Imp0, L0;

struct DataBlock{
	int		Ns;
	int		NN;
	real	Mh;
	real	Mh_inf;
	real    a;
	real    Rh;
	real    Mb;
	real    b;
	real    Rb;
	real    Rh2;
	real    con;
	real    const1;
	real    c_psi_h;
	real    c_psi_b;
	real    eps2;
};

DataBlock d;
__constant__ DataBlock dd;

//-----device function-----
__device__ real3 dev_fex(real4 p, real t){
	real3 f;
	real rr, rr3, forcehalo, rrbcore1, root1, forceblg1, forcesph, rrcore;

	rr = sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
	if (rr>0.0)
	{
		rr3 = rr*rr*rr;
		rrcore = rr / dd.a;
		rrbcore1 = rr / dd.b;
		root1 = sqrt(1.0 + rrbcore1*rrbcore1);
	
		//--Halo--
		if (rr<dd.Rh2) forcehalo = -dd.con *(rrcore - atan(rrcore)) / rr3;
		else  forcehalo = -dd.Mh_inf / rr3;
		//--Bulge--
		if (rr<dd.Rb) forceblg1 = -dd.const1*(dd.b*log(rrbcore1 + root1) - rr / root1) / rr3;
		else  forceblg1 = -dd.Mb / rr3;
	
		forcesph = forceblg1 + forcehalo;
		f.x = forcesph * p.x;
		f.y = forcesph * p.y;
		f.z = forcesph * p.z;
	}
	else{
		f = make_real3(0.0, 0.0, 0.0);
	}
	return f;
}

//-----Force_Nbody kernel--------
__global__ void ACC_Zero(real3 *ACC)
{
	int ii = threadIdx.x + blockIdx.x * blockDim.x;
	ACC[ii] = make_real3(0.0, 0.0, 0.0);
}
__global__ void PSI_Zero(real *PSI)
{
	int ii = threadIdx.x + blockIdx.x * blockDim.x;
	PSI[ii] = 0.0;
}
__global__ void ACCEL(real3 *ACC, real4 *Pos_i, real4 *Pos_j, real2 *Mhp_j, real *eps2_pj)
{
	__shared__ real4 sp[BLOCK_SIZE];
	__shared__ real eps2[BLOCK_SIZE];

	int ind = blockIdx.x * blockDim.x;
	int ii = threadIdx.x + ind;
	real4 ps = Pos_i[ii];
	real3 f = make_real3(0.0, 0.0, 0.0);
	int i, j, jj;
	real s, eps2_ii = eps2_pj[ii];
	real3 r;

	ind = 0;
	for (i = 0; i < gridDim.x; i++, ind += BLOCK_SIZE) //(0)
	{
		jj = ind + threadIdx.x;
		sp[threadIdx.x] = make_real4(Pos_j[jj].x, Pos_j[jj].y, Pos_j[jj].z, Mhp_j[jj].x);
		eps2[threadIdx.x] = eps2_pj[jj];
		__syncthreads();
		for (j = 0; j < BLOCK_SIZE; j++)
		{
			r.x = sp[j].x - ps.x; 	r.y = sp[j].y - ps.y;	r.z = sp[j].z - ps.z;
			s = 1.0 / sqrt(r.x*r.x + r.y*r.y + r.z*r.z + 0.5*(eps2_ii+eps2[j]));
			s = s*s*s * sp[j].w;
			f.x += r.x*s; 	f.y += r.y*s; 	f.z += r.z*s;
		}
		__syncthreads();
	}
	ACC[ii] = make_real3(ACC[ii].x + f.x, ACC[ii].y + f.y, ACC[ii].z + f.z); 
	//ACC[ii] = f;
}

//-----Psi_Nbody kernel--------
__global__ void PSI_kernel(real *PSI, real4 *Pos_i, real4 *Pos_j, real2 *Mhp_j, real *eps2_pj)
{
	__shared__ real4 sp[BLOCK_SIZE];
	__shared__ real eps2[BLOCK_SIZE];

	int ind = blockIdx.x * blockDim.x;
	int ii = threadIdx.x + ind;
	real4 ps = Pos_i[ii];
	int i, j, jj;
	real s, eps2_ii = eps2_pj[ii];
	real3 r;
  s=0.0;

	ind = 0;
	for (i = 0; i < gridDim.x; i++, ind += BLOCK_SIZE) //(0)
	{
		jj = ind + threadIdx.x;
		sp[threadIdx.x] = make_real4(Pos_j[jj].x, Pos_j[jj].y, Pos_j[jj].z, Mhp_j[jj].x);
		eps2[threadIdx.x] = eps2_pj[jj];
		__syncthreads();
		for (j = 0; j < BLOCK_SIZE; j++)
		{
			r.x = sp[j].x - ps.x; 	r.y = sp[j].y - ps.y;	r.z = sp[j].z - ps.z;
			s += sp[j].w / sqrt(r.x*r.x + r.y*r.y + r.z*r.z + 0.5*(eps2_ii+eps2[j]));
		}
		__syncthreads();
	}
	PSI[ii] = PSI[ii] - s;
}

__global__ void kernelNbody_integTime(real3 *ACC, real4 *Pos_t, real4 *Vel_t, real4 *Pos, real4 *Vel, real dt, int istep, real t, real3 *ACC0)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	real4 v = Vel[i], r = Pos[i], vt;
	real3 f, fex;

	fex = dev_fex(r, t);
	f.x = ACC[i].x + fex.x;
	f.y = ACC[i].y + fex.y;
	f.z = ACC[i].z + fex.z;

	//----predictor-------------------- q = q(t), q_t = q(t+dt) , dt = dt
	if (istep == 0){
		//----Velosity vx, vy, vz----------------------
		vt.x = v.x + dt*f.x;
		vt.y = v.y + dt*f.y;
		vt.z = v.z + dt*f.z;
		//----Position x,y,z----------------------
		Pos_t[i].z = r.z + 0.5*dt*(v.z + vt.z);
		Pos_t[i].x = r.x + 0.5*dt*(v.x + vt.x);
		Pos_t[i].y = r.y + 0.5*dt*(v.y + vt.y);
		
		Vel_t[i] = vt;
	} else{
		//------corrector---------------------- q_t = q(t), q = q(t+dt) - predictor, dt = dt
		real4 vtt;
		vt = Vel_t[i];
		//----Velosity vz----------------------
		vtt.z = 0.5*(vt.z + v.z + dt*f.z);
		vtt.x = 0.5*(vt.x + v.x + dt*f.x);
		vtt.y = 0.5*(vt.y + v.y + dt*f.y);
		
		Pos_t[i] = r;
		Vel_t[i] = vtt;
		ACC0[i] = ACC[i];
	}
}
//---Host Function----
__host__ void  result(real4 *pos, real4 *vel, real2 *mass, int it, real t, real *PSI, int it_all)
{
	FILE *outf;
	char str[24];
	int Ns = d.Ns, NN = d.NN, i;
	real vfi, vr, vx, vy, vz, x, y, z, r, Ek,Ep,E;
	real Vr_max = 0.0, Vfi_max = 0.0, Vz_max = 0.0, R_max = 0.0, Z_max = 0.0;
	real3 Imp = make_real3(0.0, 0.0, 0.0), L;
	int Ndm = NN-Ns;

	//---Star------
	i = sprintf(str, "bin/S_%5d.bin", it);
	outf = fopen(str, "wb");
	fwrite(&Ns, sizeof(int), 1, outf);
	fwrite(&t, sizeof(double), 1, outf);
	for (i = 0; i<Ns; i++){
		fwrite(&pos[i].x, sizeof(double), 1, outf);
		fwrite(&pos[i].y, sizeof(double), 1, outf);
		fwrite(&pos[i].z, sizeof(double), 1, outf);
		fwrite(&vel[i].x, sizeof(double), 1, outf);
		fwrite(&vel[i].y, sizeof(double), 1, outf);
		fwrite(&vel[i].z, sizeof(double), 1, outf);
	}
	fclose(outf);
	 //---DM------
  if(Ndm>0){
		i = sprintf(str, "bin/DM_%5d.bin", it);
  	outf = fopen(str, "wb");
  		fwrite(&Ndm, sizeof(int), 1, outf);
  		fwrite(&t, sizeof(double), 1, outf);
  		for (i = Ns; i<NN; i++){
    		fwrite(&pos[i].x, sizeof(double), 1, outf);
    		fwrite(&pos[i].y, sizeof(double), 1, outf);
    		fwrite(&pos[i].z, sizeof(double), 1, outf);
    		fwrite(&vel[i].x, sizeof(double), 1, outf);
    		fwrite(&vel[i].y, sizeof(double), 1, outf);
    		fwrite(&vel[i].z, sizeof(double), 1, outf);
  		}
  	fclose(outf);
	}
  L.x = 0.0; L.y = 0.0; L.z = 0.0; 
  Ek = 0.0; Ep = 0.0;
	for (i = 0; i < NN; i++)
	{
		x = pos[i].x; y = pos[i].y; z = pos[i].z;
		vx = vel[i].x; vy = vel[i].y; vz = vel[i].z;
		r = sqrt(x*x + y*y);
		if (r > 0.0) { vr = (vx*x + vy*y) / r; vfi = (vy*x - vx*y) / r; }
		else { vr = 0.0; vfi = 0.0; }

		L.x += (vz*y - vy*z)*mass[i].x;
		L.y += (vx*z - vz*x)*mass[i].x;
    L.z += (vy*x - vx*y)*mass[i].x;
    Imp.x += mass[i].x*vx; 
    Imp.y += mass[i].x*vy; 
    Imp.z += mass[i].x*vz;
    Ek += mass[i].x*(vx*vx+vy*vy+vz*vz);
    Ep += mass[i].x*PSI[i];

		Vr_max = (Vr_max < abs(vr)) ? abs(vr) : Vr_max;
		Vfi_max = (Vfi_max < vfi) ? vfi : Vfi_max;
		Vz_max = (Vz_max < abs(vz)) ? abs(vz) : Vz_max;
		R_max = (R_max < r) ? r : R_max;
		Z_max = (Z_max < abs(z)) ? abs(z) : Z_max;
	}
  E = 0.5*(Ek+Ep);

	if (it == 0) {
		L0.x = L.x; L0.y = L.y; L0.z = L.z;
		Imp0.x = Imp.x; Imp0.y = Imp.y; Imp0.z = Imp.z;
    E0=E;
		outf = fopen("LIE_0.bin", "wb");
			fwrite(&L0, sizeof(real3), 1, outf);
			fwrite(&Imp0, sizeof(real3), 1, outf);
      fwrite(&E0, sizeof(real), 1, outf);
		fclose(outf);
	}

  real LL = sqrt(L.x*L.x+L.y*L.y+L.z*L.z);
  real LL0 = sqrt(L0.x*L0.x+L0.y*L0.y+L0.z*L0.z);


	printf("           ***Star***");
	printf("\n R_max = %g  Z_max = %g \n", R_max, Z_max);
	printf("Vr_max = %g  Vfi_max = %g  Vz_max = %g \n", Vr_max, Vfi_max, Vz_max);
	printf("           ***Conservation Laws***\n");
	printf("----Imp0--- = %g; %g; %g  \n", Imp0.x, Imp0.y, Imp0.z);
	printf("----dImp--- = %g; %g; %g  \n", Imp.x - Imp0.x, Imp.y - Imp0.y, Imp.z - Imp0.z);
	printf("Lz = %g  dLz = %g \n", L.z, L.z / L0.z - 1.0);
	printf("LL = %g  dLL = %g \n", LL, LL/LL0-1.0);
  printf("E = %g  dE = %g \n", E, E/E0 - 1.0);

	outf = (it==0) ? fopen("Lz(t).dat", "w") : fopen("Lz(t).dat", "a");
	fprintf(outf, "%d %f %1.15f %g %g\n", it_all, t, L.z, L.z/L0.z-1.0, fabs(L.z/L0.z-1.0) );
	fclose(outf);
	outf = (it==0) ? fopen("LL(t).dat", "w") : fopen("LL(t).dat", "a");
	fprintf(outf, "%d %f %1.15f %g %g\n", it_all, t, LL, LL/LL0-1.0, fabs(LL/LL0-1.0) );
	fclose(outf);
	outf = (it==0) ? fopen("Imp(t).dat", "w") : fopen("Imp(t).dat", "a");
  real Impls = sqrt(Imp.x*Imp.x+Imp.y*Imp.y+Imp.z*Imp.z);
  real Impls0 = sqrt(Imp0.x*Imp0.x+Imp0.y*Imp0.y+Imp0.z*Imp0.z);
	fprintf(outf, "%d %f %1.15f %g %g %g\n", it_all, t, Impls, Impls-Impls0, Impls/Impls0-1.0, fabs(Impls/Impls0-1.0));
	fclose(outf);
  outf = (it==0) ? fopen("E(t).dat", "w") : fopen("E(t).dat", "a");
	fprintf(outf, "%d %f %1.15f %g %g %g %g\n", it_all, t, E, E/E0-1.0, fabs(E/E0-1.0), 0.5*Ek, 0.5*Ep);
	fclose(outf);
}


int main(int argc, char * argv[])
{
	FILE *outf;
	real r, vr, vfi, fi;
  char str[24];

	//----GPU device------------------------------------------------
	int deviceCount, nGPU;
	cudaDeviceProp prop;
	int j, nthr, i;

	#pragma omp parallel
	{nthr = omp_get_num_threads(); }
	printf("Threads_CPU=%d\n", nthr);

	cudaGetDeviceCount(&deviceCount);
	printf("deviceCount = %d\n", deviceCount);
	for (i = 0; i < deviceCount; i++){
		cudaGetDeviceProperties(&prop, i);
		printf("Id_GPU = %d, Name_GPU = %s\n", i, prop.name);
	}

  char name[FILENAME_MAX];
	outf = fopen("__GPUs.ini", "r");
		fscanf(outf, "%d  %[^\n]", &nGPU, name);
    int *deviceId = new int[nGPU];
	  for (i = 0; i < nGPU; i++) {
      fscanf(outf, "%d  %[^\n]", &deviceId[i], name);
		  if (deviceId[i] > deviceCount - 1)
      {
			  printf("\n Net takogo nomera device GPU"); return 0;
		  }
	  }
	fclose(outf);

	for (i = 0; i < nGPU; i++) printf("deviceId[%d] = %d\n", i, deviceId[i]);
	int can_access_peer, itmp;
	for (i = 0; i < nGPU; i++){
		cudaSetDevice(deviceId[i]);
		for (j = 0; j < nGPU; j++){
			if (j != i){
				cudaDeviceCanAccessPeer(&can_access_peer, deviceId[i], deviceId[j]);
				printf("can_access_peer=%d, %d, %d\n", can_access_peer, deviceId[i], deviceId[j]);
				if (can_access_peer == 0) {
					printf("ERROR! -- can_access_peer = 0 for deviceId = %d  and  deviceId = %d\n", deviceId[i], deviceId[j]);
					printf("Press any key + Enter\n"); 
					scanf("%d", &itmp); 
					exit(0);
				}
			}
		}
	}

	for (i = 0; i < nGPU; i++)
  {
		cudaSetDevice(deviceId[i]);
		for (j = 0; j < nGPU; j++)
    {
			if (j != i) cudaDeviceEnablePeerAccess(deviceId[j], 0);
		}
	}

	cudaEvent_t start, stop, start1, stop1;
	float gpuTime = 0.0, gpuTime_GFC = 0.0, gpuTime_US = 0.0, gpuTime1 = 0.0;
	//-----Unitial State---------------------------------------------------------
	real tmax, t = 0.0, tsave, dtsave, dtgrav=0.001,tgrav;
	real Mh, a, Rh, Mb, b, Rb, eps2;
	int Ns, i_cont, NN, Ndm, n0;
	real K_m, K_r;
	int *N_s, *N_dm, M_glx, k, k_glx;
	double *Mass_s, *Mass_dm, *mp_s, *mp_dm, *X_glx, *Y_glx, *Z_glx, *Vx_glx, *Vy_glx, *Vz_glx, *alpha_glx, *eps_s , *eps_dm;

  //----start_galaxies---------------------------------------------------------------------------
	outf = fopen("__start_galaxies.ini", "r");
  	fscanf(outf, "%d %[^\n]", &M_glx, name);
    N_s = new int[M_glx]; N_dm = new int[M_glx];
    Mass_s = new double[M_glx]; Mass_dm = new double[M_glx];
    mp_s = new double[M_glx]; mp_dm = new double[M_glx];
    X_glx = new double[M_glx]; Y_glx = new double[M_glx]; Z_glx = new double[M_glx];
    Vx_glx = new double[M_glx]; Vy_glx = new double[M_glx]; Vz_glx = new double[M_glx];
    alpha_glx = new double[M_glx];
		eps_s = new double[M_glx]; eps_dm = new double[M_glx];
    for(k=0; k<M_glx; k++){
			fscanf(outf, "%d %[^\n]", &k_glx, name);
      fscanf(outf, "%d,%d %[^\n]", &N_s[k],&N_dm[k], name);
      fscanf(outf, "%lf,%lf %[^\n]", &Mass_s[k],&Mass_dm[k], name);
      fscanf(outf, "%lf,%lf %[^\n]", &eps_s[k],&eps_dm[k], name);
      fscanf(outf, "%lf %[^\n]", &alpha_glx[k], name);
      fscanf(outf, "%lf,%lf,%lf %[^\n]", &X_glx[k],&Y_glx[k],&Z_glx[k], name);
      fscanf(outf, "%lf,%lf,%lf %[^\n]", &Vx_glx[k],&Vy_glx[k],&Vz_glx[k], name);
      if(Mass_s[k]==0.0 || N_s[k]==0) {Mass_s[k]=0.0; N_s[k]=0;}
      if(Mass_dm[k]==0.0 || N_dm[k]==0) {Mass_dm[k]=0.0; N_dm[k]=0;}
      mp_s[k] = (N_s[k]>0) ? Mass_s[k]/N_s[k] : 0.0;
      mp_dm[k] = (N_dm[k]>0) ? Mass_dm[k]/N_dm[k] : 0.0;
      Ns += N_s[k];
      Ndm += N_dm[k];
      alpha_glx[k] *= PI / 180.0;
		}
  NN = Ns + Ndm;
	printf("NN = %d, Ns = %d, Ndm = %d\n", NN, Ns, Ndm);
  printf("mp_s[0] = %g, mp_dm[0] = %g \n", mp_s[0], mp_dm[0]);
	outf = fopen("__start_nbody.ini", "r");
		fscanf(outf, "%d  %[^\n]", &i_cont, name);
		fscanf(outf, "%lf  %[^\n]", &tmax, name);
		fscanf(outf, "%lf  %[^\n]", &dtsave, name);
	fclose(outf);
	//tsave = 0.0;
	tsave = dtsave;

	//test
	printf("%d\n", i_cont);
	printf("%f\n", tmax);
	printf("%f\n", dtsave);

	printf("Ns / BLOCK_SIZE_b = %d\n", Ns / BLOCK_SIZE);
	printf("Ndm / BLOCK_SIZE_b = %d\n", Ns / BLOCK_SIZE);
	printf("NN / BLOCK_SIZE_b = %d\n", NN / BLOCK_SIZE);

	outf = fopen("__gr_par.ini", "r");
		fscanf(outf, "%lf  %[^\n]", &Mh, name);
		fscanf(outf, "%lf  %[^\n]", &a, name);
		fscanf(outf, "%lf  %[^\n]", &Rh, name);
		fscanf(outf, "%lf  %[^\n]", &Mb, name);
		fscanf(outf, "%lf  %[^\n]", &b, name);
		fscanf(outf, "%lf  %[^\n]", &Rb, name);
		fscanf(outf, "%lf  %[^\n]", &eps2, name);
		fscanf(outf, "%lf  %[^\n]", &dtgrav, name);
		fscanf(outf, "%lf  %[^\n]", &K_m, name);    // K_m = Md/(10^{10}*Msun)
		fscanf(outf, "%lf  %[^\n]", &K_r, name);    // K_r = L_r / 10 кпк
	fclose(outf);
	printf("%f\n", Mh);
	printf("%f\n", a);
	printf("%f\n", Rh);
	printf("%f\n", Mb);
	printf("%f\n", b);
	printf("%f\n", Rb);
	printf("%f\n", eps2);
	eps2 *= eps2;
	printf("%f\n", eps2);
	printf("%f\n", dtgrav);

	int is_grav = (int)(dtsave / dtgrav + 0.5), it_grav = 0;

	real Rh2 = 3.0*Rh;
	real rcore1 = Rh / a;
	real rbcore1 = 1.0 / b;
	real rbcore2 = rbcore1 * rbcore1;
	real root1 = sqrt(1.0 + (Rb*Rb)*rbcore2);
	real con = Mh / (rcore1 - atan(rcore1));
	real const1 = Mb / (b*log(Rb*rbcore1 + root1) - Rb / root1);
	real c_psi_h = con / a*(0.5*log(Rh2*Rh2 / a / a + 1.0) + atan(Rh2 / a)*a / Rh2) + Mh / Rh2;
	real c_psi_b = Mb / Rb - const1*log(Rb / b + root1) / Rb;
	real Mh_inf = Mh * (Rh2 / a - atan(Rh2 / a)) / (rcore1 - atan(rcore1));

	printf("*****Rh2 = %g \n", Rh2);
	printf("*****con = %g \n", con);
	printf("*****const1 = %g \n", const1);
	printf("*****c_psi_h = %g \n", c_psi_h);
	printf("*****c_psi_b = %g \n", c_psi_b);


	//-----Allocate Massiv Host-----------------------------
	real4 *pos_host = new real4[NN];
	real4 *vel_host = new real4[NN];
	real2 *mass_host = new real2[NN];
	real *eps2_p = new real[NN];
  real *PSI_host = new real[NN];
	//-----------------------------------------------------

	for (i = 0; i < NN; i++)
	{
		pos_host[i].x = 0.0; pos_host[i].y = 0.0; pos_host[i].z = 0.0; pos_host[i].w = 0.0;
		vel_host[i].x = 0.0; vel_host[i].y = 0.0; vel_host[i].z = 0.0; vel_host[i].w = 0.0;
		mass_host[i].x = 0.0; mass_host[i].y = 0.0; PSI_host[i] = 0.0;
	}

	d.Ns = Ns; d.NN = NN; 
	d.Mh = Mh; d.Mh_inf = Mh_inf; d.a = a; d.Rh = Rh; d.Mb = Mb; d.b = b; d.Rb = Rb; d.Rh2 = Rh2; d.con = con;	d.const1 = const1;
	d.c_psi_h = c_psi_h; d.c_psi_b = c_psi_b; d.eps2 = eps2;

	int it = 1, itt = 1, ittg=0, itg=1;

	printf("***Input Data***\n");

  if (i_cont > 0)
	{
  //---Stars---
			i = sprintf(str, "bin/S_%5d.bin", i_cont);
      outf = fopen(str, "rb");
      	fread(&Ns, sizeof(int), 1, outf);
				fread(&t, sizeof(double), 1, outf);
				n0=0;
				for(k=0; k<M_glx; k++){						
        	for (i = n0; i < n0+N_s[k]; i++) {
						fread(&pos_host[i].x, sizeof(double), 1, outf);
										fread(&pos_host[i].y, sizeof(double), 1, outf);
										fread(&pos_host[i].z, sizeof(double), 1, outf);
										fread(&vel_host[i].x, sizeof(double), 1, outf);
										fread(&vel_host[i].y, sizeof(double), 1, outf);
										fread(&vel_host[i].z, sizeof(double), 1, outf);
										mass_host[i].x = mp_s[k];
										eps2_p[i] = eps_s[k]*eps_s[k];
								}
								n0 += N_s[k];
						}
            fclose(outf);
      //---DM---
			if(Ndm>0)
			{
      	i = sprintf(str, "bin/DM_%5d.bin", i_cont);
        outf = fopen(str, "rb");
        	fread(&Ndm, sizeof(int), 1, outf);
          fread(&t, sizeof(double), 1, outf);
					n0=0;
					for(k=0; k<M_glx; k++){													
           		for (i = n0+Ns; i < n0+Ns+N_dm[k]; i++) {
                  fread(&pos_host[i].x, sizeof(double), 1, outf);
                  fread(&pos_host[i].y, sizeof(double), 1, outf);
                  fread(&pos_host[i].z, sizeof(double), 1, outf);
                  fread(&vel_host[i].x, sizeof(double), 1, outf);
                  fread(&vel_host[i].y, sizeof(double), 1, outf);
                  fread(&vel_host[i].z, sizeof(double), 1, outf);
									mass_host[i].x = mp_dm[k];
									eps2_p[i] = eps_dm[k]*eps_dm[k];
            		}
								n0 += N_dm[k];
					}
        fclose(outf);
			}
      it = (int)(t / dtsave);
      printf("Start time = %f  it = %d\n", t, it);
      printf("***Start result t***\n");
			outf = fopen("LIE_0.bin", "rb");
				fread(&L0, sizeof(real3), 1, outf);
				fread(&Imp0, sizeof(real3), 1, outf);
        fread(&E0, sizeof(real), 1, outf);
			fclose(outf);
			//result(pos_host, vel_host, mass_host, it, t);
      it++;
      itg = (int)(t / dtgrav) + 1;
      tsave = t + dtsave;
      tgrav = t + dtgrav;
	} else
	{
    	printf("t = %f\n",t);
    	int itmp;
      real rtmp;
			//----Stars-------------------------------
			n0=0;
      for(k=0; k<M_glx; k++){
      	if(N_s[k]>0){
        	i = sprintf(str, "start_S%1d.txt", k);
          if((outf = fopen(str, "r"))==NULL){printf("Error OF -- %s ",str); return 0;}
          else {
          	outf = fopen(str, "r");
            	fscanf(outf, "%d %lf", &itmp, &rtmp);
              printf("N_s[%d] = %d, t = %f\n", k, itmp, rtmp);
              for(i=n0; i<n0+N_s[k]; i++){
              	fscanf(outf, "%lf %lf %lf %lf %lf %lf", &pos_host[i].x, &pos_host[i].y, &pos_host[i].z, &vel_host[i].x, &vel_host[i].y, &vel_host[i].z);
                mass_host[i].x = mp_s[k];
                pos_host[i].x = X_glx[k] + pos_host[i].x*cos(alpha_glx[k]) + pos_host[i].z*sin(alpha_glx[k]);
                pos_host[i].y += Y_glx[k];
                pos_host[i].z = Z_glx[k] + pos_host[i].z*cos(alpha_glx[k]) - pos_host[i].x*sin(alpha_glx[k]);
                vel_host[i].x = Vx_glx[k] + vel_host[i].x*cos(alpha_glx[k]) + vel_host[i].z*sin(alpha_glx[k]);;
                vel_host[i].y += Vy_glx[k];
                vel_host[i].z = Vz_glx[k] + vel_host[i].z*cos(alpha_glx[k]) - vel_host[i].x*sin(alpha_glx[k]);
								eps2_p[i] = eps_s[k]*eps_s[k];
              }
            fclose(outf);
          }
          n0 += N_s[k];
        }
      }
      //----DM-------------------------------
      for(k=0; k<M_glx; k++){
      	if(N_dm[k]>0){
        	i = sprintf(str, "start_DM%1d.txt", k);
          if((outf = fopen(str, "r"))==NULL){printf("Error OF -- %s ",str); return 0;}
          else {
          	outf = fopen(str, "r");
            	fscanf(outf, "%d %lf", &itmp, &rtmp);
              printf("N_dm[%d] = %d, t = %f\n", k, itmp, rtmp);
              for(i=n0; i<n0+N_dm[k]; i++){
              	fscanf(outf, "%lf %lf %lf %lf %lf %lf", &pos_host[i].x, &pos_host[i].y, &pos_host[i].z, &vel_host[i].x, &vel_host[i].y, &vel_host[i].z);
                mass_host[i].x = mp_dm[k];
                pos_host[i].x = X_glx[k] + pos_host[i].x*cos(alpha_glx[k]) + pos_host[i].z*sin(alpha_glx[k]);
                pos_host[i].y += Y_glx[k];
                pos_host[i].z = Z_glx[k] + pos_host[i].z*cos(alpha_glx[k]) - pos_host[i].x*sin(alpha_glx[k]);
                vel_host[i].x = Vx_glx[k] + vel_host[i].x*cos(alpha_glx[k]) + vel_host[i].z*sin(alpha_glx[k]);;
                vel_host[i].y += Vy_glx[k];
                vel_host[i].z = Vz_glx[k] + vel_host[i].z*cos(alpha_glx[k]) - vel_host[i].x*sin(alpha_glx[k]);
								eps2_p[i] = eps_dm[k]*eps_dm[k];														
              }
            fclose(outf);
          }
          n0 += N_dm[k];
        }
      }
			tsave = dtsave;
			tgrav = dtgrav;
			t = 0.0;
			//printf("***Start result t=0***\n");
			//result(pos_host, vel_host, mass_host, 0, 0.0, PSI_host, 0);
    }


	printf("***Start GPU***\n");

	//-----Allocate Massiv GPU--------------------------------------
	real4 **pos_dev = new real4*[nGPU], **vel_dev = new real4*[nGPU], **post_dev = new real4*[nGPU], **velt_dev = new real4*[nGPU];
	real3 **ACC_dev = new real3*[nGPU], **ACC_devt = new real3*[nGPU];
	real2 **mass_dev = new real2*[nGPU];
	real **eps2_dev = new real*[nGPU];
  real **PSI_dev = new real*[nGPU];
	//real *dt_dev;
	int Nk = NN / nGPU;

#pragma omp parallel num_threads(nGPU) default(shared)
{
  #pragma omp for schedule(static,1) private(i)
	for (i = 0; i < nGPU; i++){
		cudaSetDevice(deviceId[i]);
		cudaMalloc((void**)&pos_dev[i], Nk * sizeof(real4));
		cudaMalloc((void**)&vel_dev[i], Nk * sizeof(real4));
		cudaMalloc((void**)&post_dev[i], Nk * sizeof(real4));
		cudaMalloc((void**)&velt_dev[i], Nk * sizeof(real4));
		cudaMalloc((void**)&mass_dev[i], Nk * sizeof(real2));
		cudaMalloc((void**)&ACC_dev[i], Nk * sizeof(real3));
		cudaMalloc((void**)&ACC_devt[i], Nk * sizeof(real3));
		cudaMalloc((void**)&eps2_dev[i], Nk * sizeof(real));
    cudaMalloc((void**)&PSI_dev[i], Nk * sizeof(real));
		//Copy data CPU to GPU
		cudaMemcpy(pos_dev[i], pos_host + i*Nk, Nk * sizeof(real4), cudaMemcpyHostToDevice);
		cudaMemcpy(vel_dev[i], vel_host + i*Nk, Nk * sizeof(real4), cudaMemcpyHostToDevice);
		cudaMemcpy(mass_dev[i], mass_host + i*Nk, Nk * sizeof(real2), cudaMemcpyHostToDevice);
		cudaMemcpy(eps2_dev[i], eps2_p + i*Nk, Nk * sizeof(real), cudaMemcpyHostToDevice);
    cudaMemcpy(PSI_dev[i], PSI_host + i*Nk, Nk * sizeof(real), cudaMemcpyHostToDevice);
		//DataBlock dd --- GPU
		cudaMemcpyToSymbol(dd, &d, sizeof(DataBlock), 0, cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();

		int threadNum(omp_get_thread_num());
		printf("deviceId=%d,  threadCPU=%d\n", deviceId[i], threadNum);
	}
	#pragma omp barrier
	//------Расчет грав. сил-----------------------------------------------------
  #pragma omp for schedule(static,1) private(i)
	for (i = 0; i < nGPU; i++){
		cudaSetDevice(deviceId[i]);
		ACC_Zero << <Nk / BLOCK_SIZE, BLOCK_SIZE >> >(ACC_dev[i]);
		cudaDeviceSynchronize();
	}
	#pragma omp barrier
  #pragma omp for schedule(static,1) private(i,j)
	for (i = 0; i < nGPU; i++){
		cudaSetDevice(deviceId[i]);
		for (j = 0; j < nGPU; j++){
			if (j != i) 
			{
				ACCEL << <Nk / BLOCK_SIZE, BLOCK_SIZE >> >(ACC_dev[i], pos_dev[i], pos_dev[j], mass_dev[j], eps2_dev[j]);
				cudaDeviceSynchronize();
        PSI_kernel << <Nk / BLOCK_SIZE, BLOCK_SIZE >> >(PSI_dev[i], pos_dev[i], pos_dev[j], mass_dev[j], eps2_dev[j]);
        cudaDeviceSynchronize();
			}
			else 
      {
        ACCEL << <Nk / BLOCK_SIZE, BLOCK_SIZE >> >(ACC_dev[i], pos_dev[i], pos_dev[i], mass_dev[i], eps2_dev[i]);
				cudaDeviceSynchronize();
        PSI_kernel << <Nk / BLOCK_SIZE, BLOCK_SIZE >> >(PSI_dev[i], pos_dev[i], pos_dev[i], mass_dev[i], eps2_dev[i]);
        cudaDeviceSynchronize();
      }
		}
	}
  #pragma omp barrier
  #pragma omp for schedule(static,1) private(i)
	for (i = 0; i < nGPU; i++){
		cudaSetDevice(deviceId[i]);
    cudaMemcpy(PSI_host + i*Nk, PSI_dev[i], Nk * sizeof(real), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
	}
	#pragma omp barrier
}
if(it==1){
	printf("***Start result t=0***\n");
	result(pos_host, vel_host, mass_host, 0, 0.0, PSI_host, 0);
 }
  
  cudaSetDevice(deviceId[0]);
  cudaEventCreate(&start1);
	cudaEventCreate(&stop1);

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	printf("is_grav = %d  it = %d  itg = %d\n", is_grav, it, itg);
	printf("t = %g  tgrav = %g  tsave = %g\n", t, tgrav, tsave);

	do
	{
	// --- Nbody и самогравитация
	cudaEventRecord(start1, 0);
  #pragma omp parallel num_threads(nGPU) default(shared)
  {
		//------Nbody predictor (tn+dtgrav)----------------------------------------------------------------------------
    #pragma omp for schedule(static,1) private(i)
		for (i = 0; i < nGPU; i++){
			cudaSetDevice(deviceId[i]);
			kernelNbody_integTime << <Nk / BLOCK_SIZE, BLOCK_SIZE >> >(ACC_dev[i], post_dev[i], velt_dev[i], pos_dev[i], vel_dev[i], dtgrav, 0, tgrav - dtgrav, ACC_dev[i]);
      cudaDeviceSynchronize();
		}
		 #pragma omp barrier
		//------Расчет самогравитации Nbody частиц-----------------------------------------------------
    #pragma omp for schedule(static,1) private(i)
		for (i = 0; i < nGPU; i++){
			cudaSetDevice(deviceId[i]);
			ACC_Zero << <Nk / BLOCK_SIZE, BLOCK_SIZE >> >(ACC_devt[i]);
      cudaDeviceSynchronize();
		}
    #pragma omp barrier
    #pragma omp for schedule(static,1) private(i,j)
		for (i = 0; i < nGPU; i++){
			cudaSetDevice(deviceId[i]);
			for (j = 0; j < nGPU; j++){
				if (j != i)
				{
					ACCEL << <Nk / BLOCK_SIZE, BLOCK_SIZE >> >(ACC_devt[i], post_dev[i], post_dev[j], mass_dev[j], eps2_dev[j]);
          cudaDeviceSynchronize();
				}
				else {
					ACCEL << <Nk / BLOCK_SIZE, BLOCK_SIZE >> >(ACC_devt[i], post_dev[i], post_dev[i], mass_dev[i], eps2_dev[i]);
        	cudaDeviceSynchronize();
				}
			}
		}
    #pragma omp barrier
		//------Nbody corrector (tn+dtgrav)----------------------------------------------------------------------------
    #pragma omp for schedule(static,1) private(i)
		for (i = 0; i < nGPU; i++){
			cudaSetDevice(deviceId[i]);
			kernelNbody_integTime << <Nk / BLOCK_SIZE, BLOCK_SIZE >> >(ACC_devt[i], pos_dev[i], vel_dev[i], post_dev[i], velt_dev[i], dtgrav, 1, tgrav, ACC_dev[i]);
      cudaDeviceSynchronize();
		}
    #pragma omp barrier
		//----------------------------------------------------------------------------------------------------------------------
  }
    cudaSetDevice(deviceId[0]);
		cudaEventRecord(stop1, 0);
		cudaEventSynchronize(stop1);
		cudaEventElapsedTime(&gpuTime1, start1, stop1);
		gpuTime_GFC += gpuTime1;
		ittg++;
		itg++;
		t = tgrav;
		tgrav = itg*dtgrav;
		//tgrav += dtgrav;
		it_grav++;

		if (it_grav >= is_grav){
			tsave = tgrav;
			//Copy data GPU to CPU
    #pragma omp parallel num_threads(nGPU) default(shared)
    {
      for (i = 0; i < nGPU; i++){
		    cudaSetDevice(deviceId[i]);
		    PSI_Zero << <Nk / BLOCK_SIZE, BLOCK_SIZE >> >(PSI_dev[i]);
				cudaDeviceSynchronize();
	    }
      #pragma omp barrier
      #pragma omp for schedule(static,1) private(i,j)
	    for (i = 0; i < nGPU; i++){
		    cudaSetDevice(deviceId[i]);
		    for (j = 0; j < nGPU; j++){
			    if (j != i) 
			    {
            PSI_kernel << <Nk / BLOCK_SIZE, BLOCK_SIZE >> >(PSI_dev[i], pos_dev[i], pos_dev[j], mass_dev[j], eps2_dev[j]);
            cudaDeviceSynchronize();
			    }
			    else 
          {
            PSI_kernel << <Nk / BLOCK_SIZE, BLOCK_SIZE >> >(PSI_dev[i], pos_dev[i], pos_dev[i], mass_dev[i], eps2_dev[i]);
            cudaDeviceSynchronize();
          }
		    }
	    }
      #pragma omp barrier
      #pragma omp for schedule(static,1) private(i)
			for (i = 0; i < nGPU; i++){
				cudaSetDevice(deviceId[i]);
				cudaMemcpy(pos_host + i*Nk, pos_dev[i], Nk * sizeof(real4), cudaMemcpyDeviceToHost);
				cudaMemcpy(vel_host + i*Nk, vel_dev[i], Nk * sizeof(real4), cudaMemcpyDeviceToHost);
				cudaMemcpy(mass_host + i*Nk, mass_dev[i], Nk * sizeof(real2), cudaMemcpyDeviceToHost);
        cudaMemcpy(PSI_host + i*Nk, PSI_dev[i], Nk * sizeof(real), cudaMemcpyDeviceToHost);
				cudaDeviceSynchronize();
			}
			#pragma omp barrier
    }
			cudaSetDevice(deviceId[0]);
      cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&gpuTime, start, stop);

			//-------------------------------------------------------------------------------------
			gpuTime = (gpuTime_GFC) / itt;
			printf("--------------------------------------------------------------------------------");
			outf = (it==1) ? fopen("time_frame.dat", "w") : fopen("time_frame.dat", "a");
			fprintf(outf, "%d %g %g %g %g\n", it*itt, t, 0.001f*gpuTime_GFC, 0.001f*gpuTime_US, 0.001f*(gpuTime_US+gpuTime_GFC));
			fclose(outf);

			printf("<Time_frame> = %.3f s, frame = %d, iter = %d, iter_g = %d\n", 0.001f*gpuTime, it, itt, ittg);
			printf("Time = (%g,  %g) ---  dt = (%g,  %g)\n", t, tgrav-dtgrav, dtsave/itt, dtgrav);
			printf("t_GFC = %g s, t_US = %g\n", 0.001f*gpuTime_GFC / ittg, 0.001f*gpuTime_US / itt);


			i = 1;
			printf("i=%d :: rho[i] = %g  e[i] = %g  h[i] = %g\n", i, pos_host[i].w, vel_host[i].w, mass_host[i].y);

			fi = atan2(pos_host[i].y, pos_host[i].x);
			r = sqrt(pos_host[i].x*pos_host[i].x + pos_host[i].y*pos_host[i].y);
			vr = (vel_host[i].x*pos_host[i].x + vel_host[i].y*pos_host[i].y) / r;
			vfi = (vel_host[i].y*pos_host[i].x - vel_host[i].x*pos_host[i].y) / r;
			printf("Vr[i] = %g  Vfi[i] = %g  Vz[i] = %g\n", vr, vfi, vel_host[i].z);
			printf("r[i] = %g  fi[i] = %g  z[i] = %g\n", r, fi, pos_host[i].z);

			result(pos_host, vel_host, mass_host, it, t, PSI_host, it*itt);

			tsave += dtsave; it++;
			itt = 0; ittg = 0; 
			it_grav = 0;
			cudaEventRecord(start, 0);
			gpuTime_GFC = 0.0; gpuTime_US = 0.0;
		}
		itt++;
	} while (t<tmax);

	delete pos_host, vel_host, mass_host;
	for (i = 0; i < nGPU; i++){
		cudaSetDevice(deviceId[i]);
		for (j = 0; j < nGPU; j++){
			if (j != i) cudaDeviceDisablePeerAccess(deviceId[j]);
		}
	}
	for (i = 0; i < nGPU; i++){
		cudaSetDevice(deviceId[i]);
		cudaFree(pos_dev[i]);
		cudaFree(post_dev[i]);
		cudaFree(vel_dev[i]);
		cudaFree(velt_dev[i]);
		cudaFree(mass_dev[i]);
		cudaFree(ACC_dev[i]);
		cudaFree(ACC_devt[i]);
	}

	return 0;
}