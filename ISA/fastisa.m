function fastica(dim, rdim, t, ss, maxiter, seed)

% Code for FastISA fixed point algorithm
%
% Input args:  rdim - reduced rdimensionality
%              t - number of data points
%              n - rdimensionality
%              ss - subspace size
%              maxiter - number of iterations
%
% examples usage: fastisa(20,20,10000,4,100,0)
%
% function calls:
%   gen_data.m to generate artifical subspace data
%   coeffsgauss.m gives normalization constant for generalized gaussian density

% Author: Urs Koster and Aapo Hyvarinen (2006-2012)
% Web: http://www.cs.helsinki.fi/u/koster/ 
%      http://www.cs.helsinki.fi/u/ahyvarin/
% This software is made available under the Creative Commons
% Attribution-Noncommercial License.
% (http://creativecommons.org/licenses/by-nc/3.0/)



rand('seed', seed); % initialize random number generator

B = rand(rdim);    
B = B*real((B'*B)^(-0.5));
ne = size(B,2);
obj = [];
objiter = [];


% Create subspace structure for ISA

disp('creating subspace structure for ISA')
ISAmatrix=zeros(dim);
for i=1:(dim/ss)
    for j=1:ss
        for k=1:ss
            offset=(i-1)*ss;
            ISAmatrix(offset+j,offset+k)=1;
        end
    end
end

% Collect artificial data with subspace dependencies

[X, whiteningMatrix, dewhiteningMatrix, M,S] = gen_data(dim,rdim, t, ss, seed ); 

% Initialize algorithm

W = rand(dim,rdim); 
E= zeros(dim,rdim); E(1,rdim)=.0001;
W = real((W*W')^(-0.5))*W; % orthoganlize

% FastISA algorithm main loop

for iter=1:maxiter 
    fprintf('(%d)',iter);

    %compute norms of projections to subspaces
    Y=W*X; 
    Y2=Y.^2;
    K=ISAmatrix*Y2; 

    % log-density and gradient for generalized exponential density
    epsilon=0.1;
    gK =  (epsilon+K).^(-0.5);       % nonlinearity
    gpK = -0.5.*(epsilon+K).^(-1.5); % gradient of the nonlinearity 

    %  fixed-point update step
    W = (Y.*gK)*X'/t - W.*(mean((gK+2*gpK.*Y2)')'*ones(1,size(W,2)));

    % Orthogonalize 
    W = real((W*W')^(-0.5))*W;

    %compute likelihood for each subspace
    loglikelihood(iter)=0; %this is gradually filled up within this loop
    for firstcomp=1:ss:(dim-1)
        indices=[firstcomp:(firstcomp+ss-1)]; %indices to one subspace
        alpha=0.5;
        [Z,b]=coeffsgauss(alpha,ss);
        r2=sum(Y(indices,:).^2,1); % sum inside space
        logLs=log(1/Z) - r2.^alpha/(b^(alpha));
        logL=sum(logLs); % sum over t
        loglikelihood(iter)=loglikelihood(iter)+logL; % sum up subspaces
    end 

    % write results to disk
    if rem(iter,10)==0  | iter==maxiter   
        A = dewhiteningMatrix * W'; % Filters in original space
        B = W * whiteningMatrix;    % Basis functions in original space
        save isarun.mat A B W 
    end
    
end 
fprintf('\n')

% data we work on is X = wM*M*S where S are sources
% estimate W such that W*X=S, check for W*wM*M permuted identity

% Plot the results
figure(1)
subplot(2,2,1)
imagesc(whiteningMatrix*M), title('True mixing matrix')
subplot(2,2,2)
imagesc(W), title('Recovered mixing matrix')
subplot(2,2,3)
imagesc(W*whiteningMatrix*M, [-1 1]), title('Mixing * recovered demixing')
subplot(2,2,4)
plot(loglikelihood), xlabel('iteration'), ylabel('log-likelihood')
drawnow

%keyboard

