function [c,b]=coeffsgauss(alpha,n)

     b=(n*gamma(n/2/alpha)/ ...
            gamma((n+2)/2/alpha));

     c=gamma(n/2/alpha)*(b^(n/2))/alpha*pi^(n/2)/gamma(n/2);
 
     %d=2; a=alpha;
     %c=(2^n * b^(n/d) * n * gamma(n/(a*d)) * gamma(1/d)^n)  /   (a * d^(n+1) * gamma(n/d+1));

return
