#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <semaphore.h>
#include <time.h>

#define DEFAULT_THREADS 4
#define TOTAL_POINTS 1000000000LL

int num_threads;
const char* sem_name = "/task3-sem";
FILE* out_file = NULL;
double global_sum = 0.0;

#ifdef USE_SEMAPHORE
sem_t* sem;
#endif // USE_SEMAPHORE

static int init()
{
    out_file = fopen("data.txt", "a");
    if (out_file == NULL) {
        fprintf(stderr, "failed to open data file");
        return -1;
    }

#ifdef USE_SEMAPHORE
    sem = sem_open(sem_name, O_CREAT, 0644, 1);
    if (sem == SEM_FAILED) {
        fprintf(stderr, "failed to create semaphore");
        return -1;
    }
#endif // USE_SEMAPHORE

    return 0;
}

static int deinit()
{
    fclose(out_file);

#ifdef USE_SEMAPHORE
    sem_close(sem);
    sem_unlink(sem_name);
#endif // USE_SEMAPHORE

    return 0;
}

typedef struct {
    int thread_id;
    long long points;
} thread_data;

static void* monte_carlo(void* arg)
{
    thread_data* data = (thread_data*)arg;

    unsigned int seed = time(NULL) + data->thread_id;
    double* local_sum = malloc(sizeof(double));
    *local_sum = 0.0;

    for (long long i = 0; i < data->points; i++) {
        double x = ((double)rand_r(&seed) / RAND_MAX) * M_PI;
        double y = (double)rand_r(&seed) / RAND_MAX;

        if (y <= sin(x)) {
            *local_sum += x * y;
        }
    }

#ifdef USE_SEMAPHORE
    sem_wait(sem);
    global_sum += *local_sum;
    sem_post(sem);

    return NULL;
#else
    pthread_exit(local_sum);
#endif // USE_SEMAPHORE
}

static long long points_for_thread(int thread_id,
                                   int num_threads,
                                   long long total_points)
{
    long long base = total_points / num_threads;
    long long remainder = total_points % num_threads;

    if (thread_id < remainder) {
        return base + 1;
    }

    return base;
}

int main(int argc, const char* argv[])
{
    if (argc < 2) {
        num_threads = DEFAULT_THREADS;
    } else {
        num_threads = atoi(argv[1]);
    }

    if (init() < 0) {
        return EXIT_FAILURE;
    }

    pthread_t threads[num_threads];
    thread_data data[num_threads];

    struct timespec begin, end;
    double elapsed;

    clock_gettime(CLOCK_REALTIME, &begin);

    for (int i = 0; i < num_threads; i++) {
        data[i].thread_id = i;
        data[i].points = points_for_thread(i, num_threads,
                                           TOTAL_POINTS);

        pthread_create(&threads[i], NULL, monte_carlo, &data[i]);
    }

#ifdef USE_SEMAPHORE
    for(int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
#else
    for (int i = 0; i < num_threads; i++) {
        double* local_sum;
        pthread_join(threads[i], (void**)&local_sum);

        if (local_sum != NULL) {
            global_sum += *local_sum;
            free(local_sum);
        }
    }
#endif // USE_SEMAPHORE

    double area = M_PI;
    double integral = area * global_sum / TOTAL_POINTS;

    clock_gettime(CLOCK_REALTIME, &end);

    elapsed = end.tv_sec - begin.tv_sec;
    elapsed += (end.tv_nsec - begin.tv_nsec) / 1e9;

    printf("Integral = %.10f\n", integral);
    printf("Time = %lf seconds\n", elapsed);

    fprintf(out_file, "%lf\n", elapsed);

    if (deinit() < 0) {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
