#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <stddef.h>
#include "mmio.h"

int dim=10;
char fileName[] = "matrix.txt";

typedef struct data_s {
	double value;
    int row;
    int column;
} data;

long random_at_most(long max) {
	unsigned long
		num_bins = (unsigned long) max + 1,
		num_rand = (unsigned long) RAND_MAX + 1,
		bin_size = num_rand / num_bins,
		defect   = num_rand % num_bins;

	long x;
	do {
		x = random();
	} while (num_rand - defect <= (unsigned long)x);

	return x/bin_size;
}

int main(int argc,char* argv[]) {
	MM_typecode matcode;
	FILE *f;
	int *I, *J, nz=0;
	double *val;

	if (argc>1) {
		double matrix[dim][dim];
		
		if (strcmp(argv[1],"gm")==0) {
			for(int i=0; i<dim; i++) {
				for(int j=i; j<dim; j++) {
					if (i==j) {
						matrix[i][i] = 1;
					} else {
						if (random_at_most(100)<20) {
							long r = random_at_most(100);
							matrix[i][j] = r;
							matrix[j][i] = r;
						} else {
							matrix[i][j] = 0;
							matrix[j][i] = 0;
						}
					}
					
					if (matrix[i][j]!=0) {
						nz++;
					}
				}
			}
			
			nz = ((nz-dim)*2+dim);

			for(int i=0; i<dim; i++) {
				for(int j=0; j<dim; j++) {
				    if(j!=i) {
				    	matrix[i][i] += matrix[i][j];
				    }
				}
			}
			
			I = (int *) malloc(nz * sizeof(int));
			J = (int *) malloc(nz * sizeof(int));
			val = (double *) malloc(nz * sizeof(double));
			
			int k = 0;
			
			for(int i=0; i<dim; i++) {
				for(int j=0; j<dim; j++) {
					if (matrix[i][j]!=0) {
						I[k] = j;
						J[k] = i;
						val[k] = matrix[i][j];
						k++;
					}
				}
			}
		}

		mm_initialize_typecode(&matcode);
		mm_set_matrix(&matcode);
		mm_set_sparse(&matcode);
		mm_set_real(&matcode);
		mm_set_symmetric(&matcode);

		f = fopen(fileName,"w");

		mm_write_banner(f, matcode); 
		mm_write_mtx_crd_size(f, dim, dim, nz);

		for (int i=0; i<nz; i++) fprintf(f, "%d %d %10.3g\n", I[i]+1, J[i]+1, val[i]);
		
		printf("A matrix of dimension %i has been generated in file %s\n",dim,fileName);
	} else {
		int rank; /*RANK OF THE CURRENT PROCESS*/
		int m; /*NUMBER OF PROCESSES*/
		double A[dim][dim]; /*THE MATRIX A*/
		double R[dim][dim]; /*THE ZERO MATRIX*/

		MPI_Init(&argc,&argv); /*START MPI */
		MPI_Comm_rank(MPI_COMM_WORLD,&rank); /*DETERMINE RANK OF THIS PROCESSOR*/
		MPI_Comm_size(MPI_COMM_WORLD,&m); /*DETERMINE TOTAL NUMBER OF PROCESSORS*/
		
		for (int i=0; i<dim; i++) {
			for (int j=0; j<dim; j++) {
				R[i][j]=0;
				A[i][j]=0;
			}
		}
		
		int ret_code;
		f = fopen(fileName,"r");
		if ((ret_code = mm_read_mtx_crd_size(f, &dim, &dim, &nz)) !=0)
        	exit(1);
		
		I = (int *) malloc(nz * sizeof(int));
		J = (int *) malloc(nz * sizeof(int));
		val = (double *) malloc(nz * sizeof(double));
		
		for (int i=0; i<nz; i++) {
			fscanf(f, "%d %d %lg\n", &I[i], &J[i], &val[i]);
			I[i]--;
			J[i]--;
			A[J[i]][I[i]] = val[i];
		}
		
		int binary_matrix[dim][dim];
		for (int i = 0; i<dim; i++)
			for (int j = 0; j<dim; j++)
				binary_matrix[i][j] = 1;
		for (int i = 0; i<nz; i++)
			binary_matrix[J[i]][I[i]] = 0;
		
		/*COMMUNICATION DATA*/
		const int nitems=3;
		int blocklengths[3] = {1,1,1};
		MPI_Datatype types[3] = {MPI_DOUBLE, MPI_INT, MPI_INT};
		MPI_Datatype mpi_data_type;
		MPI_Aint offsets[3];

		offsets[0] = offsetof(data, value);
		offsets[1] = offsetof(data, row);
		offsets[2] = offsetof(data, column);

		MPI_Type_create_struct(nitems, blocklengths, offsets, types, &mpi_data_type);
		MPI_Type_commit(&mpi_data_type);
		/*##################*/
		
		data comm_data;
		int first_index, last_index;
		
		/*DATA DiVISION*/
		if (rank != m-1) {
			first_index = ((rank+1-1)*(floor(dim/m)))+1;
			last_index = (rank+1)*(floor(dim/m));
		} else {
			first_index = ((rank+1-1)*(floor(dim/m)))+1;
			last_index = dim;
		}
		first_index--;
		last_index--;
		/*############*/ 
		
		/*INCOMPLETE CHOLESKY FACTORIZATION*/
		int i=0,j=0;
		for (j=first_index; j<=last_index; j++) {
			for (i=j; i<dim; i++) {
				/*RECIEVE DATA*/
				if(rank != 0) {
					int all_data_revieved = 1;
					for (int k=0; k<j; k++)
						for (int l=0; l<=k; l++)
							if (binary_matrix[k][l]==0) 
								all_data_revieved = 0;
					
					while (all_data_revieved == 0) {
						MPI_Recv(&comm_data,1,mpi_data_type,MPI_ANY_SOURCE,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
						int k = comm_data.row, l = comm_data.column;
						R[k][l] = comm_data.value;
						binary_matrix[k][l] = 1;
						all_data_revieved = 1;
						for (k=0; k<j; k++)
							for (l=0; l<=k; l++)
								if (binary_matrix[k][l]==0) 
									all_data_revieved = 0;
					}
				}
				/*############*/
				
				if (i==j) {
					double sum = 0;
					for (int k=0; k<j; k++)
						sum += pow(R[i][k],2);
					R[i][j]= pow(A[i][j]-sum, 0.5);
				} else {
					if (A[i][j]!=0) {
						double sum = 0;
						for (int k=0; k<j; k++)
							sum += R[i][k]*R[j][k];
						R[i][j]= (A[i][j]-sum)/R[j][j];
					}
				}
				
				binary_matrix[i][j] = 1;
				
				/*SEND THE ELEMENT TO EVERY OTHER PROCESS AFTER THIS ONE*/
				comm_data.value = R[i][j]; /*STORE THE VALUE*/
				comm_data.row = i; /*STORE THE ROW*/
				comm_data.column = j; /*STORE THE COLUMN*/

				if(rank != m-1) {
					for(int k=rank+1; k<m; k++) {
						MPI_Send(&comm_data,1,mpi_data_type,k,0,MPI_COMM_WORLD);
					}
				}
				/*######################################################*/
			}
		}
		/*#################################*/
		
		/*PRINT R AND R TRANSPOSE*/
		if (rank == m-1) {
			double RT[dim][dim];
			
			for (i=0; i<dim; i++) {
				for (j=0; j<dim; j++)
					RT[i][j] = 0;
			}
			
			printf("The Matrix R:\n");
			for (i=0; i<dim; i++) {
				for (j=0; j<dim; j++) {
					printf("%lg\t",R[i][j]);
					RT[j][i] = R[i][j];
				}
				printf("\n");
			}
			
			printf("The Matrix R transpose:\n");
			for (i=0; i<dim; i++) {
				for (j=0; j<dim; j++)
					printf("%lg\t",RT[i][j]);
				printf("\n");
			}
			
		}
		/*#######################*/
		
		MPI_Finalize();
	}
	
	return 0;
}
