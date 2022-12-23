#include <iostream>
#include <omp.h>
#include <stdlib.h>
#include <windows.h>
#include<fstream>

using namespace std;
const int N = 8; // 8,000 is a good number for testing

// dft
// two pi
#define PI2 6.28318530718
// this for the rounding error, increasing N rounding error increases
// 0.01 precision good for N > 8000
#define R_ERROR 0.01

//dct
//#define SZ 8000
# define M_PI           3.14159265358979323846  /* pi */
float in[8000];
float dct[8000], idct[8000];

// d4
double h0, h1, h2, h3;
/** forward transform wave coefficients */
double g0, g1, g2, g3;

double Ih0, Ih1, Ih2, Ih3;
double Ig0, Ig1, Ig2, Ig3;

// DFT/IDFT routine
// idft: 1 direct DFT, -1 inverse IDFT (Inverse DFT)
void DFT(int idft, double* xr, double* xi, double* Xr_o, double* Xi_o, int N) {
    //#pragma omp parallel for collapse(2)
    //#pragma omp parallel for
    for (int k = 0; k < N; k++)
    {
        //#pragma omp parallel for
        for (int n = 0; n < N; n++) {
            // Real part of X[k]
            Xr_o[k] += xr[n] * cos(n * k * PI2 / N) + idft * xi[n] * sin(n * k * PI2 / N);
            // Imaginary part of X[k]
            Xi_o[k] += -idft * xr[n] * sin(n * k * PI2 / N) + xi[n] * cos(n * k * PI2 / N);
        }
    }
    //for (int n = 0; n < N; n++) {
    //    printf("% f \n +", Xr_o[n]);
    //}
    printf("Invers");

    // normalization for IDFT
    if (idft == -1) {
        //#pragma omp parallel for
        //#pragma omp parallel for collapse(1)
        for (int n = 0; n < N; n++) {
            //printf("% f \n +", Xr_o[n]);

            Xr_o[n] /= N;
            Xi_o[n] /= N;
            //printf("% f \n -", Xr_o[n]);

        }
        //for (int n = 0; n < N; n++) {
        //    printf("% f \n", Xi_o[n]);
        //}
    }
}

// set the initial signal 
// careful rand() is NOT thread safe
void fillInputDFT(double* xr, double* xi, int N) {
    //srand(GetTickCount64());
    for (int n = 0; n < N; n++) {
        // Generate random discrete-time signal x in range (-1,+1)
        //xr[n] = ((double)(2.0 * rand()) / RAND_MAX) - 1.0;
        //xi[n] = ((double)(2.0 * rand()) / RAND_MAX) - 1.0;

        // constant real signal for check purpose
        // for this signal Xr_o[0] in checkResults should be equal to N
        xr[n] = n;
        xi[n] = n;
    }
}

// set to zero the output vector
void setOutputZero(double* Xr_o, double* Xi_o, int N) {
    for (int n = 0; n < N; n++) {
        Xr_o[n] = 0.0;
        Xi_o[n] = 0.0;
    }
}

// check if x = IDFT(DFT(x))
void checkResultsDFT(double* xr, double* xi, double* xr_check, double* xi_check, double* Xr_o, double* Xi_r, int N) {
    // x[0] and x[1] have typical rounding error problem
    // interesting there might be a theorem on this
    for (int n = 0; n < N; n++) {
        if (fabs(xr[n] - xr_check[n]) > R_ERROR)
            printf("ERROR - x[%d] = %f, inv(X)[%d]=%f \n", n, xr[n], n, xr_check[n]);
        if (fabs(xi[n] - xi_check[n]) > R_ERROR)
            printf("ERROR - x[%d] = %f, inv(X)[%d]=%f \n", n, xi[n], n, xi_check[n]);
    }
    printf("Xre[0] = %f \n", Xr_o[0]);
}
void checkResults(double* x, double* x_transf, int N) {
    // x[0] and x[1] have typical rounding error problem
    // interesting there might be a theorem on this
    for (int n = 0; n < N; n++) {
        if (fabs(x[n] - x_transf[n]) > R_ERROR)
            //printf("ERROR - x[%d] = %f, inv(X)[%d]=%f \n", n, xr[n], n, xr_check[n]);
            printf("!");
    }
}

// print the results of the DFT
void printResults(double* xr, double* xi, int N) {
    for (int n = 0; n < N; n++)
        printf("Xre[%d] = %f, Xim[%d] = %f \n", n, xr[n], n, xi[n]);
}

void DCT(double in[], int m, int N) {
    float csum = 0;
    float k;
    for (int i = 0; i < m; i++) {
        for (int u = 0; u < m; u++) {
            csum += in[u] * cos((2 * u + 1) * i * M_PI / (2 * m));
        }
        if (!i) k = 0.5;
        else if ((!i) || (i)) k = 1 / sqrt(2);
        else k = 1;
        dct[i] = 2 * csum * k;
        //printf("%.3f ", dct[i]);
        csum = 0;
        //}
        //printf("\n");
    }
}
void IDCT(double in[], int m, int N) {
    float csum = 0;
    float k;
    for (int i = 0; i < m; i++) {
        for (int u = 0; u < m; u++) {
            if (!u) k = 0.5;
            else if ((!u) || (u)) k = 1 / sqrt(2);
            else k = 1;
            csum += k * dct[u] * cos((2 * i + 1) * M_PI * u / (2 * m));
        }
        idct[i] = csum * 2 / m;
        csum = 0;
        //printf("%0.3f ", idct[i]);
    }
}

void haar_1d(int n, double x[])

//****************************************************************************80
//
//  Purpose:
//
//    HAAR_1D computes the Haar transform of a vector.
//
//  Discussion:
//
//    For the classical Haar transform, N should be a power of 2.
//    However, this is not required here.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    06 March 2014
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, int N, the dimension of the vector.
//
//    Input/output, double X[N], on input, the vector to be transformed.
//    On output, the transformed vector.
//
{
    printf("\nHaar is:\n");
    int i;
    int k;
    double s;
    double* y;

    s = sqrt(2.0);

    y = new double[n];
    //
    //  Initialize.
    //
    for (i = 0; i < n; i++)
    { // 0.0 0.02960650
        y[i] = 0.0;
        //printf("% 0.1f", y[i]);

    }
    //
    //  Determine K, the largest power of 2 such that K <= N.
    //
    k = 1;
    while (k * 2 <= n)
    {
        k = k * 2;
        //printf("% 0.2f", k);
    }

    while (1 < k * 2)
    {
        k = k / 2;
        for (i = 0; i < k; i++)
        {
            y[i] = (x[2 * i] + x[2 * i + 1]) / s;
            y[i + k] = (x[2 * i] - x[2 * i + 1]) / s;
            //printf("%0.1f ", y[i+k]);

        }
        for (i = 0; i < k * 2; i++)
        {
            x[i] = y[i];
        }

    }
    //for (int i = 0; i < 15; i++) {
    //    printf("B[%d] = %.5f:\n", i, y[i]);
    //}
    delete[] y;

    return;
}
void haar_1d_inverse(int n, double x[])

//****************************************************************************80
//
//  Purpose:
//
//    HAAR_1D_INVERSE computes the inverse Haar transform of a vector.
//
//  Discussion:
//
//    For the classical Haar transform, N should be a power of 2.
//    However, this is not required here.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license.
//
//  Modified:
//
//    06 March 2014
//
//  Author:
//
//    John Burkardt
//
//  Parameters:
//
//    Input, int N, the dimension of the vector.  
//
//    Input/output, double X[N], on input, the vector to be transformed.
//    On output, the transformed vector.
//
{
    printf("\nInv is:\n");
    int i;
    int k;
    double s;
    double* y;

    s = sqrt(2.0);

    y = new double[n];
    //
    //  Initialize.
    //
    for (i = 0; i < n; i++)
    {
        y[i] = 0.0;
    }

    k = 1;
    while (k * 2 <= n)
    {
        for (i = 0; i < k; i++)
        {
            y[2 * i] = (x[i] + x[i + k]) / s;
            y[2 * i + 1] = (x[i] - x[i + k]) / s;
        }
        for (i = 0; i < k * 2; i++)
        {
            x[i] = y[i];

        }
        k = k * 2;
    }
    //for (int i = 0; i < 15; i++) {
    //    printf("B[%d] = %.5f:\n", i, y[i]);
    //}
    delete[] y;

    return;
}

void walsh_hadamard(double A[], int n_qbits, int N) {
    A[0] = 1.0; // Initialized to |000..00> 
    double* B = new double[N];

    printf("Number of qubits: %d\n", n_qbits);
    printf("Number of states: %d\n", N);
    double isq2 = 1. / sqrt(2);
    for (int qbit = 0; qbit < n_qbits; qbit++) {
        //#pragma omp parallel shared(A,B,qbit,n_qbits)
        {
            //#pragma omp for
            for (int i = 0; i < N; ++i) { B[i] = 0; }
            //#pragma omp for
            for (int j = 0; j < N; ++j) {
                if (A[j] != 0) {
                    int bit_parity = (j >> qbit) % 2;
                    if (bit_parity == 0) {
                        B[j] += isq2 * A[j];
                        int set_bit = j | (1 << qbit);
                        B[set_bit] += isq2 * A[j];
                    }
                    else if (bit_parity == 1) {
                        B[j] += -isq2 * A[j];
                        int clear_bit = j & ~(1 << qbit);
                        B[clear_bit] += isq2 * A[j];
                    }
                }
            }
#pragma omp for
            for (int i = 0; i < N; ++i) { A[i] = B[i]; }
        }
    }
    //for (int i = 0; i < N; i++) {
    //    printf("B[%d] = %.5f:\n", i, B[i]);
    //}
}

void dw4_transform(double* a, const int n)
{
    if (n >= 4) {
        int i, j;
        const int half = n >> 1;

        double* tmp = new double[n];

        for (i = 0, j = 0; j < n - 3; j += 2, i++) {
            tmp[i] = a[j] * h0 + a[j + 1] * h1 + a[j + 2] * h2 + a[j + 3] * h3;
            tmp[i + half] = a[j] * g0 + a[j + 1] * g1 + a[j + 2] * g2 + a[j + 3] * g3;
        }

        tmp[i] = a[n - 2] * h0 + a[n - 1] * h1 + a[0] * h2 + a[1] * h3;
        tmp[i + half] = a[n - 2] * g0 + a[n - 1] * g1 + a[0] * g2 + a[1] * g3;

        for (i = 0; i < n; i++) {
            a[i] = tmp[i];
        }
        for (int i = 0; i < n; i++) {
            printf("B[%d] = %.5f:\n", i, tmp[i]);
        }
        delete[] tmp;
    }
}
void dw4_invTransform(double* a, const int n)
{
    if (n >= 4) {
        int i, j;
        const int half = n >> 1;
        const int halfPls1 = half + 1;

        double* tmp = new double[n];

        //      last smooth val  last coef.  first smooth  first coef
        tmp[0] = a[half - 1] * Ih0 + a[n - 1] * Ih1 + a[0] * Ih2 + a[half] * Ih3;
        tmp[1] = a[half - 1] * Ig0 + a[n - 1] * Ig1 + a[0] * Ig2 + a[half] * Ig3;
        for (i = 0, j = 2; i < half - 1; i++) {
            //     smooth val     coef. val       smooth val    coef. val
            tmp[j++] = a[i] * Ih0 + a[i + half] * Ih1 + a[i + 1] * Ih2 + a[i + halfPls1] * Ih3;
            tmp[j++] = a[i] * Ig0 + a[i + half] * Ig1 + a[i + 1] * Ig2 + a[i + halfPls1] * Ig3;
        }
        for (i = 0; i < n; i++) {
            a[i] = tmp[i];
        }
        for (int i = 0; i < n; i++) {
            printf("B[%d] = %.5f:\n", i, tmp[i]);
        }
        delete[] tmp;
    }
}

void fillInput(double* xr, double* x_check, int N) {
    //srand(GetTickCount64());
    for (int n = 0; n < N; n++) {
        //xr[n] = (rand() % 255 + 1);
        xr[n] = n;
        x_check[n] = n;
        //printf("%0.1f ", xr[n]);

    }

}

struct three_alloc_val {
    double* x;
    double* x_transf;
    double* x_check;
};

three_alloc_val doubleAlloc()
{
    three_alloc_val return_me;

    return_me.x = new double[N];
    return_me.x_transf = new double[N];
    return_me.x_check = new double[N];
    return return_me;
}

void new_arr(double* x, double* x_transf, int N) {
    for (int i = 0; i < N; i++) {
        x_transf[i] = x[i];
    }
}

void write_in_files(double* x, double* x_transf, string name_transf, double run_time, int N) {
    // this code add values to top 
    string nameTransf = name_transf + string("_transform.csv");
    string nameNewTransf = name_transf + string("_new_transform.csv");

    ofstream myFile;
    if (!myFile)
    {
        myFile.open(nameTransf, ios::app);
        for (int i = 0; i < N; i++)
        {
            myFile << x_transf[i] << ";" << x[i] << endl;
        }
        myFile.close();
    }
    else
    {
        ofstream myFileN(nameNewTransf);
        ifstream myFile(nameTransf);

        myFileN << name_transf + '\n';
        myFileN << "Time - " << ";" << run_time << endl;

        for (int i = 0; i < N; i++)
        {
            myFileN << x_transf[i] << ";" << x[i] << endl;
        }  // Write new data after header
        myFileN << myFile.rdbuf(); // Write rest of data

        myFile.close();
        myFileN.close();/*
        remove(name);
        rename("test_fresh_transform.csv", "test_transform.csv");*/
    }


    // for time results
    string nameRes = name_transf + string("_res.csv");
    string nameNewRes = name_transf + string("_new_res.csv");

    ofstream myFileRes;
    if (!myFileRes)
    {
        myFileRes.open(nameRes, ios::app);
        myFileRes << run_time << endl;
        myFileRes.close();
    }
    else
    {
        ofstream myFileResN(nameNewRes);
        ifstream myFileRes(nameRes);


        myFileResN << run_time << endl;// Write new data after header
        myFileResN << myFileRes.rdbuf(); // Write rest of data

        myFileRes.close();
        myFileResN.close();
        /* remove("test_res.csv");
         rename("test_fresh_res.csv", "test_res.csv");*/
    }
}

void DFT_call() {
    printf("DFTW calculation with N = %d \n", N);
    
        // Allocate array for input vector
        double* xr = new double[N];
        double* xi = new double[N];
        fillInputDFT(xr, xi, N);
    
        // for checking purposes
        double* xr_check = new double[N];
        double* xi_check = new double[N];
        setOutputZero(xr_check, xi_check, N);
    
        // Allocate array for output vector
        double* Xr_o = new double[N];
        double* Xi_o = new double[N];
        setOutputZero(Xr_o, Xi_o, N);
    
        // start timer
        double start_time = omp_get_wtime();
    
        // DFT
        int idft = 1;
        DFT(idft, xr, xi, Xr_o, Xi_o, N);
        // IDFT
        idft = -1;
        DFT(idft, Xr_o, Xi_o, xr_check, xi_check, N);
    
        // stop timer
        double run_time = omp_get_wtime() - start_time;
        printf("DFTW computation in %f seconds\n", run_time);
    
        // check the results: easy to make correctness errors with openMP
        checkResultsDFT(xr, xi, xr_check, xi_check, Xr_o, Xi_o, N);
        write_in_files(Xr_o, xr, "dft", run_time, N);
        //print the results of the DFT
    #ifdef DEBUG
        printResults(Xr_o, Xi_o, N);
    #endif
    
        // take out the garbage
        delete[] xr; delete[] xi;
        delete[] Xi_o; delete[] Xr_o;
        delete[] xr_check; delete[] xi_check;
   


}
void DCT_call() {
    int m;

    printf("Enter the size of the blocks - space separated\t");
    scanf_s("%d", &m);
    if (m >= N) {
        printf("Out of bounds\n");
        exit(1);
}
    three_alloc_val var = doubleAlloc();


    fillInput(var.x, var.x_check, N);

    // start timer
    double start_time = omp_get_wtime();
    DCT(var.x, m, N);
    // stores intermediate values for later access and output
    new_arr(var.x, var.x_transf, N);
    IDCT(var.x, m, N);

    // stop timer, print duration
    double run_time = omp_get_wtime() - start_time;
    printf("DFTW computation in %f seconds\n", run_time);

    checkResults(var.x, var.x_check, N);
    write_in_files(var.x_transf, var.x, "cos", run_time, N);

}
void dw2_haar_call() {
    three_alloc_val var = doubleAlloc();
    fillInput(var.x, var.x_check, N);

   // start timer
   double start_time = omp_get_wtime();
   haar_1d(N, var.x);
   // stores intermediate values for later access and output
   new_arr(var.x, var.x_transf, N);
   haar_1d_inverse(N, var.x);
   // stop timer
   double run_time = omp_get_wtime() - start_time;
   printf("DFTW computation in %f seconds\n", run_time);

   checkResults(var.x, var.x_check, N);
   write_in_files(var.x_transf, var.x, "haar", run_time, N);
}
void walsh_hadamard_call() {
    int n_qbits = 3;
    int N_wh = pow(2, n_qbits);
    double* x = new double[N_wh];
    double* x_transf = new double[N_wh];
    fillInput(x, x_transf, N_wh);

    // start timer
    double start_time = omp_get_wtime();
    walsh_hadamard(x, n_qbits, N_wh);
    // stop timer
    double run_time = omp_get_wtime() - start_time;
    printf("DFTW computation in %f seconds\n", run_time);

    //checkResults(var.x, var.x_check, N);
    write_in_files(x_transf, x, "wh", run_time, N);
}
void dw4_call() {
    const double sqrt_3 = sqrt(3);
    const double denom = 4 * sqrt(2);
    //
    // forward transform scaling (smoothing) coefficients
    //
    h0 = (1 + sqrt_3) / denom;
    h1 = (3 + sqrt_3) / denom;
    h2 = (3 - sqrt_3) / denom;
    h3 = (1 - sqrt_3) / denom;
    //
    // forward transform wavelet coefficients
    //
    g0 = h3;
    g1 = -h2;
    g2 = h1;
    g3 = -h0;

    Ih0 = h2;
    Ih1 = g2;  // h1
    Ih2 = h0;
    Ih3 = g0;  // h3

    Ig0 = h3;
    Ig1 = g3;  // -h0
    Ig2 = h1;
    Ig3 = g1;  // -h2

    three_alloc_val var = doubleAlloc();
    fillInput(var.x, var.x_check, N);
    // start timer
    double start_time = omp_get_wtime();
    dw4_transform(var.x, N-1);
    // stores intermediate values for later access and output
    new_arr(var.x, var.x_transf, N);
    dw4_invTransform(var.x, N-1);
    // stop timer
    double run_time = omp_get_wtime() - start_time;
    printf("DFTW computation in %f seconds\n", run_time);

    //checkResults(var.x, var.x_check, N);
    write_in_files(var.x_transf, var.x, "dw4", run_time, N);
}

int main()
{
    setlocale(0, "");
    SetConsoleOutputCP(1251);
    SetConsoleCP(1251);

    int transform_number;
    char ans = 'N';
    //scanf_s("%", &ans);

    do
    {
    printf("Enter the transform number (choose transform):\t");
    scanf_s("%d", &transform_number);

    switch (transform_number)
    {
    case 0:
        DFT_call();
        break;
    case 1:
        DCT_call();
        break;
    case 2:
        dw2_haar_call();
        break;
    case 3:
        walsh_hadamard_call();
        break;
    case 4:
        dw4_call();
        break;
    default:
        break;
    }

    printf("Do you want to continue (Y/N)?):\n");
    cin>>ans;
} while ((ans == 'Y') || (ans == 'y'));
    //system("pause");
    return 0;
}