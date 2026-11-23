/* x86_xcr0.c -- Read XCR0 only when CPUID confirms XSAVE and OSXSAVE.
 * Usage: x86_xcr0. Output: xsave=<0|1> osxsave=<0|1> xcr0=<hex|unknown>.
 * Requires GCC/Clang CPUID support on x86; uses no AVX instructions.
 */
#include <cpuid.h>
#include <stdint.h>
#include <stdio.h>

int main(void) {
    unsigned int a, b, c, d;
    if (!__get_cpuid(1, &a, &b, &c, &d)) {
        puts("xsave=unknown osxsave=unknown xcr0=unknown");
        return 0;
    }
    unsigned int xsave = (c >> 26) & 1U;
    unsigned int osxsave = (c >> 27) & 1U;
    if (!xsave || !osxsave) {
        printf("xsave=%u osxsave=%u xcr0=unknown\n", xsave, osxsave);
        return 0;
    }
    uint32_t lo, hi;
    __asm__ volatile ("xgetbv" : "=a" (lo), "=d" (hi) : "c" (0));
    printf("xsave=1 osxsave=1 xcr0=0x%llx\n",
           (unsigned long long)(((uint64_t)hi << 32) | lo));
    return 0;
}
