#include <stdio.h>

double test(double a);

double print(double a) {
  printf("%f\n", a);
  return 0;
}

int main() {
  printf("%f\n", test(10));
  return 0;
}
