%%Perform jPCA as described in Churchland et al 2012.
%Edited by Adam Rouse, 8/6/20
%This code is taken directly from the code that accompanies:
%Churchland et al 2012. Neural population dynamics during reaching. Nature, 487: 51-56. 
% Accessed 3/20/18 from http://stat.columbia.edu/~cunningham/pdf/ChurchlandNature2012_code.zip

%PreState is t x n  (n is neurons or more likely PCs of reduced neural data)
%dState is t x n  dState should be PreState(t+1) - PreState(t), it must have the same number of datapoints as PreState

function jPCs = perform_jPCA(preState, dState)

numPCs = size(preState,2);
% now compute Mskew using John's method
% Mskew expects time to run vertically, transpose result so Mskew in the same format as M
% (that is, Mskew will transform a column state vector into dx)
Mskew = skewSymRegress(dState,preState)';  % this is the best Mskew for the same equation

%% USE Mskew to get the jPCs

% get the eigenvalues and eigenvectors
[V,D] = eig(Mskew); % V are the eigenvectors, D contains the eigenvalues
evals = diag(D); % eigenvalues

% the eigenvalues are usually in order, but not always.  We want the biggest
[~,sortIndices] = sort(abs(evals),1,'descend');
evals = evals(sortIndices);  % reorder the eigenvalues
evals = imag(evals);  % get rid of any tiny real part
V = V(:,sortIndices);  % reorder the eigenvectors (base on eigenvalue size)

jPCs = zeros(size(V));
for pair = 1:numPCs/2
    vi1 = 1+2*(pair-1);
    vi2 = 2*pair;
    
    VconjPair = V(:,[vi1,vi2]);  % a conjugate pair of eigenvectors
    evConjPair = evals([vi1,vi2]); % and their eigenvalues
    VconjPair = getRealVs(VconjPair,evConjPair);
    
    %%Test for which dimension has the most variance
    testProj = preState*VconjPair;  %Project PC variables into the two dimensions of the jPC plane
    rotV = pca(testProj);           %Perform PCA on the test projection to find the dimension in the plane with the most variance
    crossProd = cross([rotV(:,1);0], [rotV(:,2);0]);
    if crossProd(3) < 0, rotV(:,2) = -rotV(:,2); end   % make sure the second vector is 90 degrees clockwise from the first
    VconjPair = VconjPair*rotV;    %Rotate the jPC eigenvectors so jPC1 has the most variance
    % Flip both axes if necessary so that a majority of the excursion in jPC1 is in positive direction
    testProj = preState*VconjPair;    % Project PC variables into the new two dimensions of the jPC plan
    if nanmean(testProj(:,1)) < 0     %If mean jPC1 is less than zero, than flip, Note-this is actually tested again and potentially flipped in find_jPC in the current implementation 
        VconjPair = -VconjPair;
    end
    
    jPCs(:,[vi1,vi2]) = VconjPair;
end

% %% Get the projections
% 
% proj = Ared * jPCs;
% projAllTimes = bigAred * jPCs;
% tradPCA_AllTimes = bsxfun(@minus, bigA, mean(smallA)) * PCvectors;  % mean center in exactly the same way as for the shorter time period.
% crossCondMeanAllTimes = meanAred * jPCs;


end

%% Inline function that gets the real analogue of the eigenvectors
function Vr = getRealVs(V,evals)

    % get real vectors made from the eigenvectors
    
    % by paying attention to this order, things will always rotate CCW
    if abs(evals(1))>0  % if the eigenvalue with negative imaginary component comes first
        Vr = [V(:,1) + V(:,2), (V(:,1) - V(:,2))*1i]; 
    else
        Vr = [V(:,2) + V(:,1), (V(:,2) - V(:,1))*1i];
    end
    Vr = Vr / sqrt(2);

        %% The original jPCA orients the jPC plane so jPC1 contains most of the neural activity during planning  
        %  This has been commented out and replaced above by code above (starting at line 39) that orients the jPC plane so the most variance (regardless of time, occurs in jPC1)
%     % now get axes aligned so that plan is spread mostly along the horizontal axis
%     testProj = (Vr'*Ared(1:numAnalyzedTimes:end,:)')'; % just picks out the plan times
%     rotV = princomp(testProj);
%     crossProd = cross([rotV(:,1);0], [rotV(:,2);0]);
%     if crossProd(3) < 0, rotV(:,2) = -rotV(:,2); end   % make sure the second vector is 90 degrees clockwise from the first
%     Vr = Vr*rotV; 
% 
%     % flip both axes if necessary so that the maximum move excursion is in the positive direction
%     testProj = (Vr'*Ared')';  % all the times
%     if max(abs(testProj(:,2))) > max(testProj(:,2))  % 2nd column is the putative 'muscle potent' direction.
%         Vr = -Vr;
%     end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%
% John P Cunningham
% 2010
%
% skewSymRegress
%
% This function does least squares regression between a matrix dX and X.
% It finds a matrix M such that dX = XM (as close as possible in the 
% least squares sense).  Unlike regular least squares regression (M = X\dX)
% this function find M such that M is a skew symmetric matrix, that is, 
% M = -M^T.
%
% Put another way, this is least squares regression over the constrained set
% of skew-symmetric matrices.  
%
% This can be solved by treating M as a vector of the unique elements that
% exist in a skew-symmetric matrix. A skew-symmetric matrix M of size n by n really only
% has n*(n-1)/2 unique entries.  That is, the diagonal is 0, and the
% upper/lower triangle is the negative transpose of the lower/upper.  So,
% we can just think of such a matrix as a vector x of size n(n-1)/2.
%
% Corresponding to this change in M, we would have to change X to be quite
% big and quite redundant.  That can be done, but an easier and 
% faster and more stable thing to do is to use an iterative solver
% that takes a function and gradient evaluation. 
% This iterative procedure is numerically accurate, etc, etc.
% So, this allows us to never make a big X skew matrix, and we just have to
% reshape M between vector and skew-symmetric matrix form as appropriate.
%%%%%%%%%%%%%%%%%%%%%%%%%%%

function M = skewSymRegress(dX,X)
    
    % check that dX and X are appropriately sized.
    if ~isequal(size(dX),size(X))
        fprintf('ERROR: dX and X are not matched in size... a skew symmetric matrix (which must be square) can not result.  Check the size of dX and X, correct, and rerun the code.  This is a fatal error.\n');
        keyboard
    end
    if size(dX,2) > 20 || size(X,2) > 20 || size(dX,1) < 20 || size(X,1) < 20
        fprintf('ERROR (likely): dX and X should be ct by k, where ct is big and k is small, like 6.  Currently your ct is small or k is big.  You have probably put in the transpose of the matrices this code expects.  Check the size of dX and X, correct, and rerun the code.  This is a fatal error. \n');
        keyboard
    end
    if size(dX,1) < size(dX,2) || size(X,1) < size(X,2)
        fprintf('ERROR (almost definitely): dX and/or X are fat, not skinny.  This will result in a subrank solution.  Put differently, it is important that you have more data points than dimensions. Unless you really know what you are doing, check the size of dX and X, correct, and rerun the code. \n');
        keyboard
    end
   
    % initialize m0 somehow. It does not matter, as this function is
    % convex... ie there is provable one unique minimizer.  But a good
    % initialization will help the optimization converge faster in theory
    % (though in practice this is so fast anyway that it is unnoticed).
    M0 = X\dX; % the non-skew-symmetric matrix.
    M0k = 0.5*(M0 - M0'); % the skew-symmetric matrix component... this is the same as M (the result of this code) if the data is white, ie X'X=I.
    m0 = reshapeSkew(M0k);
    %m0 = zeros(size(m0));
    %m0 = 100*randn(size(m0)); 
    % any of the above are fine.
    
    %%%%%%%
    % the following call does all the work
    %%%%%%%
    
    % just call minimize.m with the appropriate function...
    [m, fM, i] = minimize( m0 , 'skewSymLSeval' , 1000, dX , X );
    
    
    
    % check to make sure that nothing was funky.
    if i > 500
        fprintf('Warning: more than 500 line searches were required for minimize to complete.  It should complete much more quickly.  Check the code or the conditioning of the matrices.\n');
        keyboard
    end
    
    % return the matrix
    M = reshapeSkew(m);
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%
% John P Cunningham
% 2010
%
% reshapeSkew.m
%
% jPCA2 support function
%
% A skew-symmetric matrix xMat of size n by n really only
% has n*(n-1)/2 unique entries.  That is, the diagonal is 0, and the
% upper/lower triangle is the negative transpose of the lower/upper.  So,
% we can just think of such a matrix as a vector x of size n(n-1)/2.  This
% function reshapes such a vector into the appropriate skew-symmetric
% matrix.  
% 
% The required ordering in x is row-minor, namely that xMat(1,1) = 0,
% xMat(2,1) = x(1), xMat(3,1) = x(2), and so on.
%
% This function goes either from vector to matrix or vice versa, depending
% on what the argument x is.
%
% In short, this function just reindexes a vector to a matrix or vice
% versa.
%%%%%%%%%%%%%%%%%%%%%%%%


function [ Z ] = reshapeSkew( x )

    % this reshapes a n(n-1)/2 vector to a n by n skew symmetric matrix, or vice versa.
    % First we must check if x is a matrix or a vector.
    if isvector(x)
        % then we are making a matrix
        
        % first get the size of the appropriate matrix
        % this should be n(n-1)/2 entries.
        % this is the positive root
        n = (1 + sqrt(1 + 8*length(x)))/2;
        % error check
        if n~=round(n) % if not an integer
            % this is a bad argument
            fprintf('ERROR... the size of the x vector prevents it from being shaped into a skew symmetric matrix.\n');
            keyboard;
        end
        
        % now make the matrix
        % initialize the return matrix
        Z = zeros(n);
        % and the marker index
        indMark = 1;
        
        for i = 1 : n-1
            % add the elements as appropriate.
            Z(i+1:end,i) = x(indMark:indMark+(n-i)-1);
            % now update the index Marker
            indMark = indMark + (n-i);
        end
        
        % now add the skew symmetric part
        Z = Z - Z';
        
    else
        % then we are making a vector from a matrix (note that the 
        % standard convention of lower case being a vector and upper case
        % being a matrix is now reversed).
        
        % first check that everything is appropriately sized and skew
        % symmetric
        if size(x) ~= size(x')
            % this is not symmetric
            fprintf('ERROR... the matrix x is not square, let alone skew-symmetric.\n');
            keyboard;
        end
        % now check for skew symmetry
        if abs(norm(x + x'))>1e-8
            % this is not skew symmetric.
            fprintf('ERROR... the matrix x is not skew-symmetric.\n');
            keyboard;
        end
        % everything is ok, so take the size
        n = size(x,1);
       
        
        % now make the vector Z
        indMark = 1;
        for i = 1 : n-1
            % add the elements into a column vector as appropriate.
            Z(indMark:indMark+(n-i)-1,1) = x(i+1:end,i);
            % now update the index Marker
            indMark = indMark + (n-i);
        end
        
    end
                
end

function [X, fX, i] = minimize(X, f, length, varargin)

% Minimize a differentiable multivariate function. 
%
% Usage: [X, fX, i] = minimize(X, f, length, P1, P2, P3, ... )
%
% where the starting point is given by "X" (D by 1), and the function named in
% the string "f", must return a function value and a vector of partial
% derivatives of f wrt X, the "length" gives the length of the run: if it is
% positive, it gives the maximum number of line searches, if negative its
% absolute gives the maximum allowed number of function evaluations. You can
% (optionally) give "length" a second component, which will indicate the
% reduction in function value to be expected in the first line-search (defaults
% to 1.0). The parameters P1, P2, P3, ... are passed on to the function f.
%
% The function returns when either its length is up, or if no further progress
% can be made (ie, we are at a (local) minimum, or so close that due to
% numerical problems, we cannot get any closer). NOTE: If the function
% terminates within a few iterations, it could be an indication that the
% function values and derivatives are not consistent (ie, there may be a bug in
% the implementation of your "f" function). The function returns the found
% solution "X", a vector of function values "fX" indicating the progress made
% and "i" the number of iterations (line searches or function evaluations,
% depending on the sign of "length") used.
%
% The Polack-Ribiere flavour of conjugate gradients is used to compute search
% directions, and a line search using quadratic and cubic polynomial
% approximations and the Wolfe-Powell stopping criteria is used together with
% the slope ratio method for guessing initial step sizes. Additionally a bunch
% of checks are made to make sure that exploration is taking place and that
% extrapolation will not be unboundedly large.
%
% See also: checkgrad 
%
% Copyright (C) 2001 - 2006 by Carl Edward Rasmussen (2006-09-08).

INT = 0.1;    % don't reevaluate within 0.1 of the limit of the current bracket
EXT = 3.0;                  % extrapolate maximum 3 times the current step-size
MAX = 20;                         % max 20 function evaluations per line search
RATIO = 10;                                       % maximum allowed slope ratio
SIG = 0.1; RHO = SIG/2; % SIG and RHO are the constants controlling the Wolfe-
% Powell conditions. SIG is the maximum allowed absolute ratio between
% previous and new slopes (derivatives in the search direction), thus setting
% SIG to low (positive) values forces higher precision in the line-searches.
% RHO is the minimum allowed fraction of the expected (from the slope at the
% initial point in the linesearch). Constants must satisfy 0 < RHO < SIG < 1.
% Tuning of SIG (depending on the nature of the function to be optimized) may
% speed up the minimization; it is probably not worth playing much with RHO.

% The code falls naturally into 3 parts, after the initial line search is
% started in the direction of steepest descent. 1) we first enter a while loop
% which uses point 1 (p1) and (p2) to compute an extrapolation (p3), until we
% have extrapolated far enough (Wolfe-Powell conditions). 2) if necessary, we
% enter the second loop which takes p2, p3 and p4 chooses the subinterval
% containing a (local) minimum, and interpolates it, unil an acceptable point
% is found (Wolfe-Powell conditions). Note, that points are always maintained
% in order p0 <= p1 <= p2 < p3 < p4. 3) compute a new search direction using
% conjugate gradients (Polack-Ribiere flavour), or revert to steepest if there
% was a problem in the previous line-search. Return the best value so far, if
% two consecutive line-searches fail, or whenever we run out of function
% evaluations or line-searches. During extrapolation, the "f" function may fail
% either with an error or returning Nan or Inf, and minimize should handle this
% gracefully.

if max(size(length)) == 2, red=length(2); length=length(1); else red=1; end
if length>0, S='Linesearch'; else S='Function evaluation'; end 

i = 0;                                            % zero the run length counter
ls_failed = 0;                             % no previous line search has failed
[f0 df0] = feval(f, X, varargin{:});          % get function value and gradient
fX = f0;
i = i + (length<0);                                            % count epochs?!
s = -df0; d0 = -s'*s;           % initial search direction (steepest) and slope
x3 = red/(1-d0);                                  % initial step is red/(|s|+1)

while i < abs(length)                                      % while not finished
  i = i + (length>0);                                      % count iterations?!

  X0 = X; F0 = f0; dF0 = df0;                   % make a copy of current values
  if length>0, M = MAX; else M = min(MAX, -length-i); end

  while 1                             % keep extrapolating as long as necessary
    x2 = 0; f2 = f0; d2 = d0; f3 = f0; df3 = df0;
    success = 0;
    while ~success && M > 0
      try
        M = M - 1; i = i + (length<0);                         % count epochs?!
        [f3 df3] = feval(f, X+x3*s, varargin{:});
        if isnan(f3) || isinf(f3) || any(isnan(df3)+isinf(df3)), error(''), end
        success = 1;
      catch                                % catch any error which occured in f
        x3 = (x2+x3)/2;                                  % bisect and try again
      end
    end
    if f3 < F0, X0 = X+x3*s; F0 = f3; dF0 = df3; end         % keep best values
    d3 = df3'*s;                                                    % new slope
    if d3 > SIG*d0 || f3 > f0+x3*RHO*d0 || M == 0  % are we done extrapolating?
      break
    end
    x1 = x2; f1 = f2; d1 = d2;                        % move point 2 to point 1
    x2 = x3; f2 = f3; d2 = d3;                        % move point 3 to point 2
    A = 6*(f1-f2)+3*(d2+d1)*(x2-x1);                 % make cubic extrapolation
    B = 3*(f2-f1)-(2*d1+d2)*(x2-x1);
    x3 = x1-d1*(x2-x1)^2/(B+sqrt(B*B-A*d1*(x2-x1))); % num. error possible, ok!
    if ~isreal(x3) || isnan(x3) || isinf(x3) || x3 < 0 % num prob | wrong sign?
      x3 = x2*EXT;                                 % extrapolate maximum amount
    elseif x3 > x2*EXT                  % new point beyond extrapolation limit?
      x3 = x2*EXT;                                 % extrapolate maximum amount
    elseif x3 < x2+INT*(x2-x1)         % new point too close to previous point?
      x3 = x2+INT*(x2-x1);
    end
  end                                                       % end extrapolation

  while (abs(d3) > -SIG*d0 || f3 > f0+x3*RHO*d0) && M > 0  % keep interpolating
    if d3 > 0 || f3 > f0+x3*RHO*d0                         % choose subinterval
      x4 = x3; f4 = f3; d4 = d3;                      % move point 3 to point 4
    else
      x2 = x3; f2 = f3; d2 = d3;                      % move point 3 to point 2
    end
    if f4 > f0           
      x3 = x2-(0.5*d2*(x4-x2)^2)/(f4-f2-d2*(x4-x2));  % quadratic interpolation
    else
      A = 6*(f2-f4)/(x4-x2)+3*(d4+d2);                    % cubic interpolation
      B = 3*(f4-f2)-(2*d2+d4)*(x4-x2);
      x3 = x2+(sqrt(B*B-A*d2*(x4-x2)^2)-B)/A;        % num. error possible, ok!
    end
    if isnan(x3) || isinf(x3)
      x3 = (x2+x4)/2;               % if we had a numerical problem then bisect
    end
    x3 = max(min(x3, x4-INT*(x4-x2)),x2+INT*(x4-x2));  % don't accept too close
    [f3 df3] = feval(f, X+x3*s, varargin{:});
    if f3 < F0, X0 = X+x3*s; F0 = f3; dF0 = df3; end         % keep best values
    M = M - 1; i = i + (length<0);                             % count epochs?!
    d3 = df3'*s;                                                    % new slope
  end                                                       % end interpolation

  if abs(d3) < -SIG*d0 && f3 < f0+x3*RHO*d0          % if line search succeeded
    X = X+x3*s; f0 = f3; fX = [fX' f0]';                     % update variables
    %fprintf('%s %6i;  Value %4.6e\r', S, i, f0);
    s = (df3'*df3-df0'*df3)/(df0'*df0)*s - df3;   % Polack-Ribiere CG direction
    df0 = df3;                                               % swap derivatives
    d3 = d0; d0 = df0'*s;
    if d0 > 0                                      % new slope must be negative
      s = -df0; d0 = -s'*s;                  % otherwise use steepest direction
    end
    x3 = x3 * min(RATIO, d3/(d0-realmin));          % slope ratio but max RATIO
    ls_failed = 0;                              % this line search did not fail
  else
    X = X0; f0 = F0; df0 = dF0;                     % restore best point so far
    if ls_failed || i > abs(length)         % line search failed twice in a row
      break;                             % or we ran out of time, so we give up
    end
    s = -df0; d0 = -s'*s;                                        % try steepest
    x3 = 1/(1-d0);                     
    ls_failed = 1;                                    % this line search failed
  end
end
%fprintf('\n');
end

%%%%%%%%%%%%%%%%%%%%%%
% John P Cunningham
% 2010
%
% skewSymLSeval.m
%
% This function evaluates the least squares function and derivative.
% The derivative is taken with respect to the vector m, which is the vector
% of the unique values in the skew-symmetric matrix M.
%
% A typical least squares regression of Az = b is z = A\b.  
% That is equivalent to an optimization: minimize norm(Az-b)^2.
%
% In matrix form, if we are doing a least squares regression of AM = B,
% that becomes: minimize norm(AM-B,'fro')^2, where 'fro' means frobenius
% norm.  Put another way, define error E = AM-B, then norm(E,'fro') is the
% same as norm(E(:)).
%
% Here, we want to evaluate the objective AM-B, where we call A 'X' and B
% 'dX'.  That is, we want to solve: minimize norm(dX - XM,'fro')^2.
% However, we want to constrain our solutions to just those M that are
% skew-symmetric, namely M = -M^T.  
%
% So, instead of just using M = X\dX, we here evaluate the objective and
% the derivative with respect to the unique elements of M (the vector m),
% for use in an iterative minimizer.
%
% See notes p80-82 for the derivation of the derivative.  
%%%%%%%%%%%%%%%%%%%%%%%%

function [f, df] = skewSymLSeval(m , dX , X)

% since this function is internal, we do very little error checking.  Also
% the helper functions and internal functions should throw errors if any of
% these shapes are wrong.

%%%%%%%%%%%%%
% Evaluate objective function
%%%%%%%%%%%%%

f = norm( dX - X*reshapeSkew(m) , 'fro')^2;

%%%%%%%%%%%%%
% Evaluate derivative
%%%%%%%%%%%%%
D = (dX - X*reshapeSkew(m))'*X;

df = 2*reshapeSkew( D - D' );
end
