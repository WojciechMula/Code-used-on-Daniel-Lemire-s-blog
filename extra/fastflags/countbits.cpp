#include "benchmark.h"
#include "mklarqvist.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <x86intrin.h>

void scalar_naive(const uint16_t *data, size_t n, uint32_t *flags) {
  memset(flags, 0, 16 * sizeof(uint32_t));
  for (uint32_t i = 0; i < n; ++i) {
    for (int j = 0; j < 16; ++j) {
      flags[j] += ((data[i] & (1 << j)) >> j);
    }
  }
}

void scalar_morenaive(const uint16_t *data, size_t n, uint32_t *flags) {
  memset(flags, 0, 16 * sizeof(uint32_t));
  for (uint32_t i = 0; i < n; ++i) {
    for (int j = 0; j < 16; ++j) {
      flags[j] += (data[i] >> j) & 1;
    }
  }
}

void fastavx2(uint16_t *array, size_t len, uint32_t *flags) {
  for (size_t i = 0; i < 16; i++)
    flags[i] = 0;
  uint16_t buffer[16];
  __m256i bits = _mm256_set_epi16(-32768, 16384, 8192, 4096, 2048, 1024, 512,
                                  256, 128, 64, 32, 16, 8, 4, 2, 1);
  // we do the first part
  if (len < 64) {
    // don't bother with handcrafted SIMD
    scalar_naive(array, len, flags);
    return;
  }
  // handle the start (naively)
  __m256i count16 = _mm256_setzero_si256();

  {
    uint16_t startbuffer[32];
    memset(startbuffer, 0, 32 * 2);
    memcpy(startbuffer + 16, array, 16 * 2);
    for (size_t i = 1; i < 16; i++) {
      __m256i input = _mm256_loadu_si256((__m256i *)(startbuffer + i));
      __m256i m = _mm256_and_si256(input, bits);
      __m256i eq = _mm256_cmpeq_epi16(bits, m);
      count16 = _mm256_sub_epi16(count16, eq);
    }
  }
  {
    uint16_t startbuffer[32];
    memset(startbuffer, 0, 32 * 2);
    memcpy(startbuffer, array + len - 16, 16 * 2);
    for (size_t i = 1; i < 16; i++) {
      __m256i input = _mm256_loadu_si256((__m256i *)(startbuffer + i));
      __m256i m = _mm256_and_si256(input, bits);
      __m256i eq = _mm256_cmpeq_epi16(bits, m);
      count16 = _mm256_sub_epi16(count16, eq);
    }
  }
  _mm256_storeu_si256((__m256i *)buffer, count16);
  for (size_t k = 0; k < 16; k++) {
    flags[k] += buffer[k];
  }

  // main loop starts here
  for (size_t i = 0; i + 16 <= len;) {
    count16 = _mm256_setzero_si256();
    size_t j = 0;
    size_t maxj = 65535;
    if (maxj + i + 16 >= len)
      maxj = len - i - 15;
    for (; j < maxj; j++) {
      __m256i input = _mm256_loadu_si256((__m256i *)(array + i + j));
      __m256i m = _mm256_and_si256(input, bits);
      __m256i eq = _mm256_cmpeq_epi16(bits, m);
      count16 = _mm256_sub_epi16(count16, eq);
    }
    i += j;
    _mm256_storeu_si256((__m256i *)buffer, count16);
    for (size_t k = 0; k < 16; k++) {
      flags[k] += buffer[k];
    }
  }
}

void morefastavx2(uint16_t *array, size_t len, uint32_t *flags) {
  for (size_t i = 0; i < 16; i++)
    flags[i] = 0;
  uint16_t buffer[16];
  __m256i bits = _mm256_set_epi16(-32768, 16384, 8192, 4096, 2048, 1024, 512,
                                  256, 128, 64, 32, 16, 8, 4, 2, 1);
  // we do the first part
  if (len < 64) {
    // don't bother with handcrafted SIMD
    scalar_naive(array, len, flags);
    return;
  }
  // handle the start (naively)
  __m256i count16 = _mm256_setzero_si256();

  {
    uint16_t startbuffer[32];
    memset(startbuffer, 0, 32 * 2);
    memcpy(startbuffer + 16, array, 16 * 2);
    for (size_t i = 1; i < 16; i++) {
      __m256i input = _mm256_loadu_si256((__m256i *)(startbuffer + i));
      __m256i m = _mm256_and_si256(input, bits);
      __m256i eq = _mm256_cmpeq_epi16(bits, m);
      count16 = _mm256_sub_epi16(count16, eq);
    }
  }
  {
    uint16_t startbuffer[32];
    memset(startbuffer, 0, 32 * 2);
    memcpy(startbuffer, array + len - 16, 16 * 2);
    for (size_t i = 1; i < 16; i++) {
      __m256i input = _mm256_loadu_si256((__m256i *)(startbuffer + i));
      __m256i m = _mm256_and_si256(input, bits);
      __m256i eq = _mm256_cmpeq_epi16(bits, m);
      count16 = _mm256_sub_epi16(count16, eq);
    }
  }
  _mm256_storeu_si256((__m256i *)buffer, count16);
  for (size_t k = 0; k < 16; k++) {
    flags[k] += buffer[k];
  }

  // main loop starts here
  for (size_t i = 0; i + 16 <= len;) {
    count16 = _mm256_setzero_si256();
    size_t j = 0;
    size_t maxj = 65535;
    if (maxj + i + 16 >= len)
      maxj = len - i - 15;
    if (maxj > 8) {
      for (; j < maxj - 7; j += 8) {
        __m256i input1 = _mm256_loadu_si256((__m256i *)(array + i + j));
        __m256i m1 = _mm256_and_si256(input1, bits);
        __m256i eq1 = _mm256_cmpeq_epi16(bits, m1);
        count16 = _mm256_sub_epi16(count16, eq1);
        __m256i input2 = _mm256_loadu_si256((__m256i *)(array + i + j + 1));
        __m256i m2 = _mm256_and_si256(input2, bits);
        __m256i eq2 = _mm256_cmpeq_epi16(bits, m2);
        count16 = _mm256_sub_epi16(count16, eq2);
        __m256i input3 = _mm256_loadu_si256((__m256i *)(array + i + j + 2));
        __m256i m3 = _mm256_and_si256(input3, bits);
        __m256i eq3 = _mm256_cmpeq_epi16(bits, m3);
        count16 = _mm256_sub_epi16(count16, eq3);
        __m256i input4 = _mm256_loadu_si256((__m256i *)(array + i + j + 3));
        __m256i m4 = _mm256_and_si256(input4, bits);
        __m256i eq4 = _mm256_cmpeq_epi16(bits, m4);
        count16 = _mm256_sub_epi16(count16, eq4);
        __m256i input5 = _mm256_loadu_si256((__m256i *)(array + i + j + 4));
        __m256i m5 = _mm256_and_si256(input5, bits);
        __m256i eq5 = _mm256_cmpeq_epi16(bits, m5);
        count16 = _mm256_sub_epi16(count16, eq5);
        __m256i input6 = _mm256_loadu_si256((__m256i *)(array + i + j + 5));
        __m256i m6 = _mm256_and_si256(input6, bits);
        __m256i eq6 = _mm256_cmpeq_epi16(bits, m6);
        count16 = _mm256_sub_epi16(count16, eq6);
        __m256i input7 = _mm256_loadu_si256((__m256i *)(array + i + j + 6));
        __m256i m7 = _mm256_and_si256(input7, bits);
        __m256i eq7 = _mm256_cmpeq_epi16(bits, m7);
        count16 = _mm256_sub_epi16(count16, eq7);
        __m256i input8 = _mm256_loadu_si256((__m256i *)(array + i + j + 7));
        __m256i m8 = _mm256_and_si256(input8, bits);
        __m256i eq8 = _mm256_cmpeq_epi16(bits, m8);
        count16 = _mm256_sub_epi16(count16, eq8);
      }
    }

    if (maxj > 4) {
      for (; j + 3 < maxj; j += 4) {
        __m256i input1 = _mm256_loadu_si256((__m256i *)(array + i + j));
        __m256i m1 = _mm256_and_si256(input1, bits);
        __m256i eq1 = _mm256_cmpeq_epi16(bits, m1);
        count16 = _mm256_sub_epi16(count16, eq1);
        __m256i input2 = _mm256_loadu_si256((__m256i *)(array + i + j + 1));
        __m256i m2 = _mm256_and_si256(input2, bits);
        __m256i eq2 = _mm256_cmpeq_epi16(bits, m2);
        count16 = _mm256_sub_epi16(count16, eq2);
        __m256i input3 = _mm256_loadu_si256((__m256i *)(array + i + j + 2));
        __m256i m3 = _mm256_and_si256(input3, bits);
        __m256i eq3 = _mm256_cmpeq_epi16(bits, m3);
        count16 = _mm256_sub_epi16(count16, eq3);
        __m256i input4 = _mm256_loadu_si256((__m256i *)(array + i + j + 3));
        __m256i m4 = _mm256_and_si256(input4, bits);
        __m256i eq4 = _mm256_cmpeq_epi16(bits, m4);
        count16 = _mm256_sub_epi16(count16, eq4);
      }
    }
    for (; j < maxj; j++) {
      __m256i input = _mm256_loadu_si256((__m256i *)(array + i + j));
      __m256i m = _mm256_and_si256(input, bits);
      __m256i eq = _mm256_cmpeq_epi16(bits, m);
      count16 = _mm256_sub_epi16(count16, eq);
    }
    i += j;
    _mm256_storeu_si256((__m256i *)buffer, count16);
    for (size_t k = 0; k < 16; k++) {
      flags[k] += buffer[k];
    }
  }
}

// fixme
void fastavx2mula(uint16_t *array, size_t len, uint32_t *flags) {
  for (size_t i = 0; i < 16; i++)
    flags[i] = 0;
  for (size_t i = 0; i + 32 <= len; i += 32) {
    __m256i v0 = _mm256_loadu_si256((__m256i *)(array + i));
    __m256i v1 = _mm256_loadu_si256((__m256i *)(array + i + 16));

    __m256i input0 =
        _mm256_or_si256(_mm256_and_si256(v0, _mm256_set1_epi16(0x00ff)),
                        _mm256_slli_epi16(v1, 8));
    __m256i input1 =
        _mm256_or_si256(_mm256_and_si256(v0, _mm256_set1_epi16(0xff00)),
                        _mm256_srli_epi16(v1, 8));
    for (int i = 0; i < 8; i++) {
      flags[7 - i] += __builtin_popcount(_mm256_movemask_epi8(input0));
      flags[15 - i] += __builtin_popcount(_mm256_movemask_epi8(input1));
      input0 = _mm256_add_epi8(input0, input0);
      input1 = _mm256_add_epi8(input1, input1);
    }
  }
}

// fixme
void fastavx2mula2(uint16_t *array, size_t len, uint32_t *flags) {
  __m256i counters[16];

  for (size_t i = 0; i < 16; i++) {
    counters[i] = _mm256_setzero_si256();
  }

  for (size_t i = 0; i + 16 <= len; i += 16) {
    __m256i input = _mm256_loadu_si256((__m256i *)(array + i));

    for (int j = 0; j < 16; j++) {
      __m256i bit = _mm256_and_si256(input, _mm256_set1_epi16(1));
      counters[j] = _mm256_add_epi16(counters[j], bit);
      input = _mm256_srli_epi16(input, 1);
    }
  }

  uint16_t tmp[16];
  for (size_t i = 0; i < 16; i++) {
    _mm256_storeu_si256((__m256i*)tmp, counters[i]);
    flags[i] = 0;
    for (int j=0; j < 16; j++)
      flags[i] += tmp[j];
  }
}

void demo(size_t len) {
  printf("\n Using array size = %zu \n", len);
  uint32_t counter[16];
  uint32_t truecounter[16];

  uint16_t *array = (uint16_t *)malloc(len * sizeof(uint16_t));
  memset(array, 0, len * sizeof(uint16_t));
  for (size_t i = 0; i < len; i++) {
    array[i] = i & 0xFFFF;
  }
  scalar_naive(array, len, truecounter);
  for (size_t i = 0; i < 16; i++) {
    printf("%u ", truecounter[i]);
  }
  printf("\n");
  fastavx2(array, len, counter);
  for (size_t i = 0; i < 16; i++) {
    if(counter[i] != truecounter[i]) {
      printf("fastavx2 is buggy\n");
      break;
    }
  }
  int repeat = 10;
  BEST_TIME_NOCHECK(scalar_naive(array, len, counter), , repeat, len, true);
  for (size_t i = 0; i < 16; i++) {
    if(counter[i] != truecounter[i]) {
      printf("scalar_naive is buggy\n");
      break;
    }
  }
  BEST_TIME_NOCHECK(scalar_morenaive(array, len, counter), , repeat, len, true);
  for (size_t i = 0; i < 16; i++) {
    if(counter[i] != truecounter[i]) {
      printf("fastavx2 is buggy\n");
      break;
    }
  }
  BEST_TIME_NOCHECK(fastavx2(array, len, counter), , repeat, len, true);
  for (size_t i = 0; i < 16; i++) {
    if(counter[i] != truecounter[i]) {
      printf("fastavx2 is buggy\n");
      break;
    }
  }
  BEST_TIME_NOCHECK(morefastavx2(array, len, counter), , repeat, len, true);
  for (size_t i = 0; i < 16; i++) {
    if(counter[i] != truecounter[i]) {
      printf("morefastavx2 is buggy\n");
      break;
    }
  }
  BEST_TIME_NOCHECK(flag_stats_avx2(array, len, counter), , repeat, len, true);
  for (size_t i = 0; i < 16; i++) {
    if(counter[i] != truecounter[i]) {
      printf("flag_stats_avx2 is buggy\n");
      break;
    }
  }
  BEST_TIME_NOCHECK(flag_stats_avx2_naive_counter(array, len, counter), ,
                    repeat, len, true);
  for (size_t i = 0; i < 16; i++) {
    if(counter[i] != truecounter[i]) {
      printf("flag_stats_avx2_naive_counter is buggy\n");
      break;
    }
  }
  BEST_TIME_NOCHECK(flag_stats_avx2_single(array, len, counter), , repeat, len,
                    true);
  for (size_t i = 0; i < 16; i++) {
    if(counter[i] != truecounter[i]) {
      printf("flag_stats_avx2_single is buggy\n");
      break;
    }
  }
  BEST_TIME_NOCHECK(fastavx2mula(array, len, counter), , repeat, len, true);
  for (size_t i = 0; i < 16; i++) {
    if(counter[i] != truecounter[i]) {
      printf("fastavx2mula is buggy\n");
      break;
    }
  }

  BEST_TIME_NOCHECK(fastavx2mula2(array, len, counter), , repeat, len, true);
  for (size_t i = 0; i < 16; i++) {
    if(counter[i] != truecounter[i]) {
      printf("fastavx2mula2 is buggy\n");
      break;
    }
  }
  free(array);

}

int main() {
  demo(1000000);
  demo(2000000);
  demo(4000000);
  demo(100000000);
  return EXIT_SUCCESS;
}
