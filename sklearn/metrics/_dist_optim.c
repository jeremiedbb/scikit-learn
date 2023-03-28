#ifdef __SSE3__
    #define HAS_SIMD 1

    #include <emmintrin.h>
    #include <xmmintrin.h>
    #include <pmmintrin.h>

    typedef __m128d simd_float64_t;
    typedef __m128 simd_float32_t;

    inline simd_float32_t abs_ps(simd_float32_t x) {
        const simd_float32_t sign_mask = _mm_set1_ps(-0.f); // -0.f = 1 << 31
        return _mm_andnot_ps(sign_mask, x);
    }

    // NUMPY VARIANT
    inline simd_float64_t abs_pd(simd_float64_t x) {
        // !sign_mask & x where sign_mask = -0. = 1 << 63
        return _mm_and_pd(
            x,
            _mm_castsi128_pd(_mm_set1_epi64x(0x7fffffffffffffffLL))
        );
    }

    float simd_manhattan32(const float* x, const float* y, ssize_t t) {
        simd_float32_t simd_x;
        simd_float32_t simd_y;
        ssize_t remainder = t % 4; // SIMD register can hold four floats
        ssize_t n_iter = t - remainder;
        ssize_t idx;

        simd_float32_t sum = _mm_setzero_ps();
        for(int k = 0; k < t; k+=2) {
            simd_x = _mm_set_ps(x[k], x[k+1], x[k+2], x[k+3]);
            simd_y = _mm_set_ps(y[k], y[k+1], y[k+2], y[k+3]);
            sum += abs_ps(_mm_sub_ps(simd_x, simd_y));
        }

        simd_float32_t hsum = _mm_hadd_ps(sum, sum);
        hsum = _mm_hadd_ps(hsum, hsum);
        float output_sum = *(float*)&hsum;
        for(idx = n_iter; idx < t; idx++){
            output_sum += fabs(x[idx] - y[idx]);
        }

        return output_sum;
    }

    double simd_manhattan(const double* x, const double* y, ssize_t t) {
        simd_float64_t simd_x;
        simd_float64_t simd_y;
        ssize_t remainder = t % 2; // SIMD register can hold two doubles
        ssize_t n_iter = t - remainder;
        ssize_t idx;

        simd_float64_t sum = _mm_setzero_pd();
        for(idx = 0; idx < n_iter; idx+=2) {
            simd_x = _mm_set_pd(x[idx], x[idx+1]);
            simd_y = _mm_set_pd(y[idx], y[idx+1]);
            sum += abs_pd(_mm_sub_pd(simd_x, simd_y));
        }
        simd_float64_t hsum = _mm_hadd_pd(sum, sum);
        double output_sum = *(double*)&hsum;
        for(idx = n_iter; idx < t; idx++){
            output_sum += fabs(x[idx] - y[idx]);
        }

        return output_sum;
    }
#else
    #define HAS_SIMD 0

    double simd_manhattan(double* x, double* y, ssize_t t) {return -1.f;}
    float simd_manhattan32(float* x, float* y, ssize_t t) {return -1.;}
#endif
