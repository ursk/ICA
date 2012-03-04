function [X,whiteningMatrix,dewhiteningMatrix, M,S]=fakeX(dim,rdim,t,ss, seed)
% Generate artificial ISA data to test demixing
% input args:    dim - full data dimensionality
%                rdim - dimensionality after PCA dim. reduction
%                t - number of data points to generate
%                ss - subspace size, dim must be a multiple of this
%                seed - random seed to be reproducible
% output args:   X - data matrix
%                whiteningMatrix - projects to PCA space
%                dewhiteningMatrix - projects back to pixel space
%                M - mixing matrix
%                S - unmixed data
% example usage: [X,wM,dwM,M,S] = gen_data(160,159,10000,4, 0);

rand('seed', seed); % initialize random number generator
A=randn(dim,t); % create the individual gaussians 
J=floor(dim/ss); % number of subspaces

% ------------------------------
% first we generate the sources: 
% loop through subspaces and create independent uniform
% distribution for each subspace which gets mixed with  
% common samples from a gaussian. Gives supergaussian data
% with the requred subspace structure.
% -------------------------------

for i=1:J
   
   % create samples from uniform distribution for each subspace
   flat=ones(ss,t);
   for nn=1:ss 
       flat(nn,:)=rand(1,t);
   end
   % create supergaussian ss distributions by scaling gaussian with flat
   cols = (i-1)*ss+(1:ss); % the colums to add in this iteration
   A(cols,:)= A(cols,:).*flat;     
   
end

norm_const=sum(sum(A.^2))^.5/(dim*t)^.5;  % find normalisation constant
S=  A./norm_const;   % apply normalisation, these are our unknown sources

% -------------------------------
% now generate a mixing matrix and
% and create the "observed data" by
% multiplying it with the sources
% -------------------------------

M=rand(dim);  % mixing matrix
X=M*S; % and mix

% -------------------------------
% ICA and ISA work better in whitened
% space, so we use PCA to whiten the data
% -------------------------------

covarianceMatrix = X*X'/size(X,2);
[E, D] = eig(covarianceMatrix);
[dummy,order] = sort(diag(-D));

% -------------------------------
% optionally we can perform dimensionality reduction
% here, e.g. to remove DC value. Useful with image data.
% -------------------------------

E = E(:,order(1:rdim));
d = diag(D); 
d = real(d.^(-0.5));
D = diag(d(order(1:rdim)));

whiteningMatrix =D*E'; % projection to whitened space
dewhiteningMatrix = E*D' ; % projection back to original space

% and whiten the data
X = whiteningMatrix*X; % 


fprintf('Generated %d samples of %d-dimensional artificial data.\n', t, dim);