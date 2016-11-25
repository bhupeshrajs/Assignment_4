#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

extern void writePPMImage(
    int* data,
    int width, int height,
    const char *filename,
    int maxIterations);


/*
 * Used to measure computation times.
 */
long usecs (void)
{
    struct timeval t;

    gettimeofday(&t,NULL);
    return t.tv_sec*1000000+t.tv_usec;
}

/* 
 *  Used in Serial Implementation of the mandelbrot 
 */
int mandel(double c_re, double c_im, int count)
{
    double z_re = c_re, z_im = c_im;
    int i;
    for (i = 0; i < count; ++i) {

        if (z_re * z_re + z_im * z_im > 4.0)
            break;

        double new_re = z_re*z_re - z_im*z_im;
        double new_im = 2.0 * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }

    return i;
}


/* 
 *  Mandelbrot Serial Function 
 */
void mandelbrotSerial(
    double x0, double y0, double x1, double y1,
    int width, int height,
    int startRow, int totalRows,
    int maxIterations,
    int output[])
{
    double dx = (x1 - x0) / width;
    double dy = (y1 - y0) / height;

    int endRow = startRow + totalRows;

    for (int j = startRow; j < endRow; j++) {
        for (int i = 0; i < width; ++i) {
            double x = x0 + i * dx;
            double y = y0 + j * dy;

            int index = (j * width + i);
            output[index] = mandel(x, y, maxIterations);
        }
    }
}

/*
 *  Used to check the output of serial and CUDA Executions.   
 */
bool verifyResult (int *gold, int *result, int width, int height) {

    int i, j;
    int mismatch_count = 0;
    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            if (gold[i * width + j] != result[i * width + j]) {
                mismatch_count++;
            }
        }
    }

    if( mismatch_count == 0 ) {
        return 1;
    }
    printf("\n The number of mismatches are : %d",mismatch_count);
    return 0;
}

__global__ void mandelbrotCUDA(
                    double *d_x0, double *d_y0, double *d_x1, double *d_y1,
                    int *d_width, int *d_height,
                    int *d_maxIterations,
                    int *d_output_cuda ) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y; // HEIGHT
    int col = blockIdx.x * blockDim.x + threadIdx.x; // WIDTH
    
    int index = (row * (*d_width)) + col;
    
    if( col >= (*d_width) ) return;
    if( row >= (*d_height) ) return;
    if( index >= ( (*d_height)*(*d_width) ) ) return;
    
    double dx = ( (*d_x1) - (*d_x0) ) / (*d_width);
    double dy = ( (*d_y1) - (*d_y0) ) / (*d_height);
    
    double c_re = (*d_x0) + col * dx;
    double c_im = (*d_y0) + row * dy;
    
    double z_re = c_re;
    double z_im = c_im;
    
    int i = 0;
    for ( i = 0 ; i < *d_maxIterations ; ++i ) {
    
        if( z_re * z_re + z_im * z_im > 4.0 ) 
            break;
            
        double new_re = z_re*z_re - z_im*z_im;
        double new_im = 2.0 * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }

    d_output_cuda[index] = i;
    
}

int main(int argc, char *argv[])
{

    
    if(argc < 2) {
        printf("\n Give number of threads per block dimension value.");
        return 1;
    }
    
    int thread_dimension = atoi(argv[1]);
    int size_factor = atoi(argv[2]);
    
    /* Height , width of the image */
    const int width = 1200*size_factor;
    const int height = 800*size_factor;
    
    /* Max iterations to do */
    const int maxIterations = 256;
    
    /* The value of x0,x1,y0,y1 */
    double x0 = -2;
    double x1 = 1;
    double y0 = -1;
    double y1 = 1;


    int* output_serial = (int*)malloc(width*height*sizeof(int));
    int* output_cuda = (int*)malloc(width*height*sizeof(int));
    
    
    
    int *d_output_cuda;
    double *d_x0;
    double *d_y0;
    double *d_x1;
    double *d_y1;
    int *d_width, *d_height;
    int *d_maxIterations;
    
    
    double cuda_start_time,cuda_end_time,cuda_time;
    double serial_start_time,serial_end_time,serial_time;
    double data_communication_time = 0;
    double d_start, d_end;
    double c_start, c_end, c_time;
    
    
    cuda_start_time = usecs();
    d_start = usecs();
    
    cudaMalloc((void **)&d_output_cuda, sizeof(int)*width*height);
    cudaMalloc((void **)&d_x0, sizeof(double));
    cudaMalloc((void **)&d_x1, sizeof(double));
    cudaMalloc((void **)&d_y0, sizeof(double));
    cudaMalloc((void **)&d_y1, sizeof(double));
    cudaMalloc((void **)&d_width, sizeof(int));
    cudaMalloc((void **)&d_height, sizeof(int));
    cudaMalloc((void **)&d_maxIterations, sizeof(int));
    
    cudaMemcpy(d_output_cuda, output_cuda, sizeof(int)*width*height, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x0, &x0, sizeof(double) , cudaMemcpyHostToDevice);
    cudaMemcpy(d_x1, &x1, sizeof(double) , cudaMemcpyHostToDevice);
    cudaMemcpy(d_y0, &y0, sizeof(double) , cudaMemcpyHostToDevice);
    cudaMemcpy(d_y1, &y1, sizeof(double) , cudaMemcpyHostToDevice);
    cudaMemcpy(d_width, &width, sizeof(int) , cudaMemcpyHostToDevice);
    cudaMemcpy(d_height, &height, sizeof(int) , cudaMemcpyHostToDevice);
    cudaMemcpy(d_maxIterations, &maxIterations, sizeof(int) , cudaMemcpyHostToDevice);
    
    d_end = usecs();
    
    data_communication_time = d_end - d_start; 
    
    dim3 block_size(thread_dimension, thread_dimension);
    dim3 grid_size( ceil(width / block_size.x), ceil(height / block_size.y));
    
    c_start = usecs();
    mandelbrotCUDA<<<grid_size,block_size>>>(d_x0,d_y0,d_x1,d_y1,d_width,d_height,d_maxIterations,d_output_cuda);
    c_end = usecs();
    
    c_time = ((double)(c_end-c_start))/1000000;
    
    d_start = usecs();
    
    cudaMemcpy(output_cuda, d_output_cuda, sizeof(int)*width*height, cudaMemcpyDeviceToHost);
    
    cudaFree(d_output_cuda);
    cudaFree(d_x0);
    cudaFree(d_x1);
    cudaFree(d_y0);
    cudaFree(d_y1);
    cudaFree(d_width);
    cudaFree(d_height);
    cudaFree(d_maxIterations);

    d_end = usecs();
    
    data_communication_time += (d_end - d_start);
    data_communication_time = (data_communication_time)/1000000;

    cuda_end_time = usecs();
    
    cuda_time = ((double)(cuda_end_time-cuda_start_time))/1000000;


    serial_start_time = usecs();
    mandelbrotSerial(x0, y0, x1, y1, width, height, 0, height, maxIterations, output_serial);
    serial_end_time = usecs();
    
    serial_time = ((double)(serial_end_time-serial_start_time))/1000000;

    if (! verifyResult (output_serial, output_cuda, width, height)) {
        printf ("\n Error : Output from threads does not match serial output\n");
    }
    else {
        printf("\n\"The output from the CUDA matches the serial output\"\n\n");
    }
    
    writePPMImage(output_cuda, width, height, "mandelbrot-cuda.ppm", maxIterations);

    printf("\n Serial Computation time = %fs\n", serial_time);
    printf("\n CUDA Total time = %fs\n", cuda_time);
    printf("\n Data Transfer time = %fs\n", data_communication_time);
    printf("\n Computation time = %fs\n", c_time);
    
    printf("\n Speedup Achieved is : %fx \n\n",(serial_time/cuda_time));


    return 0;
}