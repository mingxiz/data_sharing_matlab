%% Figure of PCG with different preconditioning matrix 
function test_cg

p = 500; b = 2*p;
x1 = randn(b, p); x2 = 5*rand(b, p);
A = x1'*x1; B = x2'*x2;
M = A + B;
rhs = rand(p, 1);

tol = 1e-10;
maxit = 100;
PM1 = eye(p);
[x,fl0,rr0,it0,rv0] = pcg(M,rhs,tol,maxit,PM1);
PM2 = A;
[x,fl1,rr1,it1,rv1] = pcg(M,rhs,tol,maxit,PM2);
PM3 = B;
[x,fl2,rr2,it2,rv2] = pcg(M,rhs,tol,maxit,PM3);
PM4 = inv(inv(A)+inv(B));
[x,fl3,rr3,it3,rv3] = pcg(M,rhs,tol,maxit,PM4);

pct = 0.01;
num_select = floor(pct*b);
index_select = randperm(b, num_select);
index_1 = index_select;
PM5 = 0.5*(A+1/pct*x2(index_1,:)'*x2(index_1,:))+0.5*(B+1/pct*x1(index_1,:)'*x1(index_1,:));
[x,fl4,rr4,it4,rv4] = pcg(M,rhs,tol,maxit,PM5);

semilogy(0:length(rv0)-1,rv0/norm(rhs),'-o')
hold on
semilogy(0:length(rv1)-1,rv1/norm(rhs),'-o')
semilogy(0:length(rv2)-1,rv2/norm(rhs),'-o')
semilogy(0:length(rv3)-1,rv3/norm(rhs),'-o')
semilogy(0:length(rv4)-1,rv4/norm(rhs),'-o')
yline(tol,'r--');
legend('No Preconditioner','precondition from center 1','precondition from center 2', 'local precondtion', 'global precondtion with data share','Tolerance','Location','East')
xlabel('Iteration number')
ylabel('Relative residual')