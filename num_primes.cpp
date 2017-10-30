#include<iostream>
using namespace std;

int check(int a)
{
  for(int t = 2; t*t <= a; t++)
    if(a%t == 0)
      return 0;
  return 1;
}

int main ()
{
  int n, k = 0;
  cin>>n;
  for(int i = 1; i <= n; i++)
    k += (check(i) == 1)?1:0;
  cout<<k;
  return 0;
}