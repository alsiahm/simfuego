#include "auxiliares.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>
#include <numeric>
#include <stdlib.h>
#include <stdio.h>
#include <cuda/cmath>

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)
 
/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err) {

	if (err == cudaSuccess)
		return;
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (EXIT_FAILURE);
}


static const double biomasa_inicial[] = {0, 0, .35, .65, 1};
__device__ double factor_ignicion[] = {0, 0, 1.40, 1.15, .90};
__device__ double factor_combustion[] = {0, 0, 1.40, 1, .70};

__device__ double limitar_factor(double valor) {
    return fmin(1.5, fmax(.5, valor));
}

void inicializar(const Escenario *e, Rejilla actual) {
    for (int i = 0; i < e->n; ++i) {
        actual.biomasa[i] = biomasa_inicial[e->terreno[i]];
        actual.estado[i] = INTACTA;
        if (actual.biomasa[i] == 0) actual.estado[i] = INERTE;
        if (e->foco[i]) actual.estado[i] = ARDIENDO;
    }
}

void calcular_viento(const Escenario *e, double viento[6]) {
    for (int k = 0; k < 6; ++k) {
        double angulo = (60*k - e->grados) * acos(-1.) / 180;
        viento[k] = viento[k] = fmin(1.5, fmax(.5, 1 + .05*e->velocidad*cos(angulo)));
    }
}

__device__ double influencia_vecinal(const Escenario *e, Rejilla actual,
                                 const double viento[6], int i) {
    int columna = i % e->columnas, fila = i / e->columnas;
    int impar = fila % 2;
    // Del destino a la fuente: E, NE, NO, O, SO, SE.
    int dx[6] = {1, impar, impar-1, -1, impar-1, impar};
    const int dy[6] = {0, 1, 1, 0, -1, -1};
    double suma = 0;
    for (int k = 0; k < 6; ++k) {
        int x = columna + dx[k], y = fila + dy[k];
        if (x < 0 || x >= e->columnas || y < 0 || y >= e->filas) continue;
        int j = y*e->columnas + x;
        if (actual.estado[j] != ARDIENDO) continue;
        double pendiente = limitar_factor(1 + e->altitud[i] - e->altitud[j]);
        int hacia_destino = (k + 3) % 6;
        suma += actual.biomasa[j] * viento[hacia_destino] * pendiente;
    }
    return suma;
}

/**
 * Función kernel (versión base) de la función evolucionar. Cada hilo procesa una celda de la rejilla.
 */
__global__ void evolucionar_kernel(Escenario e, const double *viento, const double *sorteo, Rejilla actual, Rejilla siguiente) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= e.n) return; // Evita que los hilos fuera del rango accedan a memoria no válida
    int tipo = e.terreno[i];
    double biomasa = actual.biomasa[i];
    siguiente.estado[i] = actual.estado[i];
    siguiente.biomasa[i] = biomasa;

    if (actual.estado[i] == ARDIENDO) {
        double consumo = factor_combustion[tipo] * (.01 + .04*biomasa);
        siguiente.biomasa[i] = fmax(0, biomasa - consumo);
        if (siguiente.biomasa[i] == 0) siguiente.estado[i] = QUEMADA;
    } else if (actual.estado[i] == INTACTA) {
        double influencia = influencia_vecinal(&e, actual, viento, i);
        double p = .20 * (1 - e.humedad[i]) * factor_ignicion[tipo] * influencia;
        if (sorteo[i] < fmin(1, p)) siguiente.estado[i] = ARDIENDO;
    }
}

Estadisticas calcular_estadisticas(int n, Rejilla actual, Rejilla anterior, int paso) {
    Estadisticas s = {0};
    for (int i = 0; i < n; ++i) {
        ++s.cuentas[actual.estado[i]];
        s.biomasa_total += actual.biomasa[i];
        if (actual.estado[i] == ARDIENDO) {
            s.intensidad_total += actual.biomasa[i];
            s.intensidad_maxima = fmax(s.intensidad_maxima, actual.biomasa[i]);
            if (paso == 0 || anterior.estado[i] == INTACTA) ++s.nuevas;
        }
    }
    return s;
}

#ifndef SIN_MAIN
int main(int argc, char **argv) {
    if (argc != 5) error("uso: simular escenario.txt pasos semilla directorio_nuevo");
    Escenario h_escenario = leer_escenario(argv[1]);
    int n = h_escenario.n;
    // Copia del struct en la CPU: se quedan n, columnas, filas, velocidad y grados.
    // Después se cambian los punteros por arrays reservados en la GPU.
    Escenario d_escenario = h_escenario;
    CUDA_CHECK_RETURN(cudaMalloc(&d_escenario.terreno, n * sizeof *d_escenario.terreno));
    CUDA_CHECK_RETURN(cudaMalloc(&d_escenario.humedad, n * sizeof *d_escenario.humedad));
    CUDA_CHECK_RETURN(cudaMalloc(&d_escenario.altitud, n * sizeof *d_escenario.altitud));
    CUDA_CHECK_RETURN(cudaMemcpy(d_escenario.terreno, h_escenario.terreno, n * sizeof *h_escenario.terreno, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(d_escenario.humedad, h_escenario.humedad, n * sizeof *h_escenario.humedad, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(d_escenario.altitud, h_escenario.altitud, n * sizeof *h_escenario.altitud, cudaMemcpyHostToDevice));
    d_escenario.foco = NULL; // foco solo se usa en inicializar (CPU), no se copia a la GPU

    int hilos = 256;
    int bloques = cuda::ceil_div(n, hilos);

    
    int pasos = entero(argv[2], 0, 1000000);
    int semilla = entero(argv[3], 0, 2147483647);
    srand(semilla);
    Salidas salidas = abrir_salidas(argv[4]);

    double h_viento[6];
    double *d_viento[6];
    CUDA_CHECK_RETURN(cudaMalloc(&d_viento, 6 * sizeof *d_viento));
    calcular_viento(&h_escenario, h_viento);
    CUDA_CHECK_RETURN(cudaMemcpy(d_viento, h_viento, 6 * sizeof *h_viento, cudaMemcpyHostToDevice));
    Rejilla h_actual = asignar_rejilla(n);
    Rejilla h_siguiente = asignar_rejilla(n);
    // Rejillas de la GPU: el struct está en la CPU, pero sus dos arrays están en la GPU.
    Rejilla d_actual, d_siguiente;
    CUDA_CHECK_RETURN(cudaMalloc(&d_actual.estado, n * sizeof *d_actual.estado));
    CUDA_CHECK_RETURN(cudaMalloc(&d_actual.biomasa, n * sizeof *d_actual.biomasa));
    CUDA_CHECK_RETURN(cudaMalloc(&d_siguiente.estado, n * sizeof *d_siguiente.estado));
    CUDA_CHECK_RETURN(cudaMalloc(&d_siguiente.biomasa, n * sizeof *d_siguiente.biomasa));

    double *h_sorteo = malloc(n * sizeof *h_sorteo);
    double *d_sorteo;
    CUDA_CHECK_RETURN(cudaMalloc(&d_sorteo, n * sizeof *d_sorteo));

    if (!h_sorteo) error("memoria insuficiente");
    inicializar(&h_escenario, h_actual);

    CUDA_CHECK_RETURN(cudaMemcpy(d_actual.estado, h_actual.estado, n * sizeof *h_actual.estado, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy(d_actual.biomasa, h_actual.biomasa, n * sizeof *h_actual.biomasa, cudaMemcpyHostToDevice));


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    float milliseconds_total = 0;
    float ms_estadisticas_total = 0;

    Estadisticas h_resumen;
    Estadisticas *d_resumen = NULL;
    CUDA_CHECK_RETURN(cudaMalloc(&d_resumen, sizeof *d_resumen));
    Estadisticas resumen = calcular_estadisticas(escenario.n, actual, actual, 0);
    escribir_paso(salidas, 0, escenario.n, actual, resumen);
    for (int paso = 1; paso <= pasos && h_resumen.cuentas[ARDIENDO] > 0; ++paso) {
        for (int i = 0; i < escenario.n; ++i) sorteo[i] = uniforme();
        CUDA_CHECK_RETURN(cudaMemcpy(d_sorteo, h_sorteo, n * sizeof *h_sorteo, cudaMemcpyHostToDevice));
        cudaEventRecord(*start);
        evolucionar_kernel<<<bloques, hilos>>>(d_escenario, d_viento, d_sorteo, d_actual, d_siguiente);
        CUDA_CHECK_RETURN(cudaGetLastError());
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        milliseconds_total += milliseconds;

        cudaEventRecord(*start);
        estadisticas_kernel<<<bloques, hilos>>>(d_escenario, d_actual, d_siguiente, paso, d_resumen);
        CUDA_CHECK_RETURN(cudaGetLastError());
        cudaEventRecord(*stop);
        cudaEventSynchronize(*stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        ms_estadisticas_total += milliseconds;
        CUDA_CHECK_RETURN(cudaMemcpy(h_resumen, d_resumen, sizeof h_resumen, cudaMemcpyDeviceToHost));
        CUDA_CHECK_RETURN(cudaMemcpy(h_siguiente.estado, d_siguiente.estado, n * sizeof *h_siguiente.estado, cudaMemcpyDeviceToHost));
        CUDA_CHECK_RETURN(cudaMemcpy(h_siguiente.biomasa, d_siguiente.biomasa, n * sizeof *h_siguiente.biomasa, cudaMemcpyDeviceToHost));
        escribir_paso(salidas, paso, escenario.n, h_siguiente, h_resumen);

        // Intercambiamos los punteros a los arrays.
        Rejilla temporal = d_actual;
        d_actual = d_siguiente;
        d_siguiente = temporal;
    }

    free(h_sorteo);
    cerrar_salidas(salidas);
    liberar_rejilla(h_actual);
    liberar_rejilla(h_siguiente);
    liberar_escenario(h_escenario);
    cudaFree(d_escenario.terreno);
    cudaFree(d_escenario.humedad);
    cudaFree(d_escenario.altitud);
    cudaFree(d_viento);
    cudaFree(d_sorteo);
    cudaFree(d_resumen);
    cudaFree(d_actual.estado);
    cudaFree(d_actual.biomasa);
    cudaFree(d_siguiente.estado);
    cudaFree(d_siguiente.biomasa);
    CUDA_CHECK_RETURN(cudaEventDestroy(*start));
    CUDA_CHECK_RETURN(cudaEventDestroy(*stop));
    return 0;
}
#endif
