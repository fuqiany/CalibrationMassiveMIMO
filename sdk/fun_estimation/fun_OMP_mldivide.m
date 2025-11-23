function xout = fun_OMP_mldivide(y,A,k)
[M,N] = size(A);
x = zeros(N,1);
r=y;
omega = [];
for i=1:k
    c = abs(A'*r);
    [m,t] = sort(c,'descend');
    l = t(1);
    omega = [omega l];
    x = mldivide(A(:,omega),y);
    r = y-A(:,omega)*x;
end
xout = zeros(N,1);
xout(omega) = x;
end