#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* 
 *  Used in Serial Implementation of the mandelbrot 
 */
int mandel(float c_re, float c_im, int count)
{
    float z_re = c_re, z_im = c_im;
    int i;
    for (i = 0; i < count; ++i) {

        if (z_re * z_re + z_im * z_im > 4.f)
            break;

        float new_re = z_re*z_re - z_im*z_im;
        float new_im = 2.f * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }

    return i;
}


/* 
 *  Mandelbrot Serial Function 
 */
void mandelbrotSerial(
    float x0, float y0, float x1, float y1,
    int width, int height,
    int startRow, int totalRows,
    int maxIterations,
    int output[])
{
    float dx = (x1 - x0) / width;
    float dy = (y1 - y0) / height;

    int endRow = startRow + totalRows;

    for (int j = startRow; j < endRow; j++) {
        for (int i = 0; i < width; ++i) {
            float x = x0 + i * dx;
            float y = y0 + j * dy;

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

    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            if (gold[i * width + j] != result[i * width + j]) {
                printf ("Mismatch : [%d][%d], Expected : %d, Actual : %d\n",
                            i, j, gold[i * width + j], result[i * width + j]);
                continue;
            }
        }
    }

    return 1;
}

__global__ void mandelbrotCUDA(
                    float *d_x0, float *d_y0, float *d_x1, float *d_y1,
                    int *d_width, int *d_height,
                    int *d_maxIterations,
                    int *d_output_cuda ) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y; // HEIGHT
    int col = blockIdx.x * blockDim.x + threadIdx.x; // WIDTH
    
    if( row == 0 && col == 800 ) {
        printf("\n block dimensions are : (%d,%d)",blockDim.x,blockDim.y) ;
        printf("\n block Ids are : (%d,%d)",blockIdx.x,blockIdx.y) ;
        printf("\n thread Ids are : (%d,%d)",threadIdx.x,threadIdx.y) ;
        printf("\n x0, x1, y0, y1 : %f, %f, %f, %f",*d_x0,*d_x1,*d_y0,*d_y1);
        printf("\n height and width : %d, %d",*d_height,*d_width);
    }
    
    int index = row * (*d_width) + col;
    
    if( col >= (*d_width) ) return;
    if( row >= (*d_height) ) return;
    if( index >= ( (*d_height)*(*d_width) ) ) return;
    
    float dx = ( (*d_x1) - (*d_x0) ) / (*d_width);
    float dy = ( (*d_y1) - (*d_y0) ) / (*d_height);
    
    float c_re = (*d_x0) + col * dx;
    float c_im = (*d_y0) + row * dy;
    
    float z_re = c_re;
    float z_im = c_im;
    
    int i;
    for ( i = 0 ; i < *d_maxIterations ; i++ ) {
    
        if( z_re * z_re + z_im * z_im > 4.f ) 
            break;
            
        float new_re = z_re*z_re - z_im*z_im;
        float new_im = 2.f * z_re * z_im;
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
    
    
    /* Height , width of the image */
    const int width = 1200;
    const int height = 800;
    
    /* Max iterations to do */
    const int maxIterations = 256;
    
    /* The value of x0,x1,y0,y1 */
    float x0 = -2;
    float x1 = 1;
    float y0 = -1;
    float y1 = 1;


    int* output_serial = (int*)malloc(width*height*sizeof(int));
    int* output_cuda = (int*)malloc(width*height*sizeof(int));
    
    int *d_output_cuda;
    float *d_x0;
    float *d_y0;
    float *d_x1;
    float *d_y1;
    int *d_width, *d_height;
    int *d_maxIterations;
    
    cudaMalloc((void **)&d_output_cuda, sizeof(int)*width*height);
    cudaMalloc((void **)&d_x0, sizeof(float));
    cudaMalloc((void **)&d_x1, sizeof(float));
    cudaMalloc((void **)&d_y0, sizeof(float));
    cudaMalloc((void **)&d_y1, sizeof(float));
    cudaMalloc((void **)&d_width, sizeof(int));
    cudaMalloc((void **)&d_height, sizeof(int));
    cudaMalloc((void **)&d_maxIterations, sizeof(int));
    
    cudaMemcpy(d_output_cuda, output_cuda, sizeof(int)*width*height, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x0, &x0, sizeof(float) , cudaMemcpyHostToDevice);
    cudaMemcpy(d_x1, &x1, sizeof(float) , cudaMemcpyHostToDevice);
    cudaMemcpy(d_y0, &y0, sizeof(float) , cudaMemcpyHostToDevice);
    cudaMemcpy(d_y1, &y1, sizeof(float) , cudaMemcpyHostToDevice);
    cudaMemcpy(d_width, &width, sizeof(int) , cudaMemcpyHostToDevice);
    cudaMemcpy(d_height, &height, sizeof(int) , cudaMemcpyHostToDevice);
    cudaMemcpy(d_maxIterations, &maxIterations, sizeof(int) , cudaMemcpyHostToDevice);
    
    dim3 block_size(thread_dimension, thread_dimension);
    dim3 grid_size( ceil(width / block_size.x), ceil(height / block_size.y));
    
    mandelbrotCUDA<<<grid_size,block_size>>>(d_x0,d_y0,d_x1,d_y1,d_width,d_height,d_maxIterations,d_output_cuda);
    
    cudaMemcpy(output_cuda, d_output_cuda, sizeof(int)*width*height, cudaMemcpyDeviceToHost);
    
    mandelbrotSerial(x0, y0, x1, y1, width, height, 0, height, maxIterations, output_serial);
    
    cudaFree(d_output_cuda);
    cudaFree(d_x0);
    cudaFree(d_x1);
    cudaFree(d_y0);
    cudaFree(d_y1);
    cudaFree(d_width);
    cudaFree(d_height);
    cudaFree(d_maxIterations);

    if (! verifyResult (output_serial, output_cuda, width, height)) {
        printf ("Error : Output from threads does not match serial output\n");
        return 1;
    }
    else {
        printf("\n\"The output from the CUDA matches the serial output\"\n\n");
    }


    return 0;
}