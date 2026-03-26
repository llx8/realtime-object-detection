#pragma once
#include <cstdio>

#define LOGI(...) do { \
    printf("[INFO] "); \
    printf(__VA_ARGS__); \
    printf("\n"); \
} while(0)

#define LOGE(...) do { \
    fprintf(stderr, "[ERROR] "); \
    fprintf(stderr, __VA_ARGS__); \
    fprintf(stderr, "\n"); \
} while(0)
