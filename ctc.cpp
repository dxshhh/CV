#include <cstdio>
#include <cmath>
#define N1 1000
#define N2 500
#define M 500
double a[N1][M], b[N2][M];
double dis(double a[M], double b[M])
{
    double ans = 0;
    for (int i = 0; i < M; ++i)
        ans += (a[i] - b[i]) * (a[i] - b[i]);
    return pow(ans, 0.5);
}
//double c[N1][N2];
int main()
{
    FILE *f1 = fopen("car.in", "r");
    FILE *f2 = fopen("cubes.in", "r");
    for (int i = 0; i < N1; ++i)
        for (int j = 0; j < M; ++j)
            fscanf(f1, "%lf", &a[i][j]);
    for (int i = 0; i < N2; ++i)
        for (int j = 0; j < M; ++j)
            fscanf(f2, "%lf", &b[i][j]);
    fclose(f1);
    fclose(f2);
    FILE *f3 = fopen("cc.in", "w");
    for (int i = 0; i < N1; ++i)
    {
        for (int j = 0; j < N2; ++j)
        {
            double c = dis(a[i], b[j]);
            fprintf(f3, "%lf ", c);
        }
        fprintf(f3, "\n");
    }
    fclose(f3);
    return 0;
}
