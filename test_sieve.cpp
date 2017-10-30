#include<iostream>
using namespace std;

int main ()
{
	int n;
	int a[n];
	cin>>n;
	a[0] = 0;
	for(int i = 1; i <=n; i++)
		a[i-1] = 1;
	int count = n-1;
	for(int i = 2; i*i <= n; i++)
	{
		for(int j = i+1; j<=n; j++)
		{
			if(a[j-1] == 1)
				if(j%i == 0)
					if(a[i-1] == 1)
					{
						count--;
						a[i-1] = 0;
					}
		}
	} 
	cout<<count;
	return 0;
}