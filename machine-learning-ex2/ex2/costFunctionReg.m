function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


htheta = sigmoid(X*theta);
logOfHtheta=log(htheta);
logOfOneminushtheta = log(1-htheta);
firstExp = (-y'*logOfHtheta);
secondExp = ((1-y)'*logOfOneminushtheta);
sumfact = firstExp-secondExp;
thetaWithoutFirst = theta(2:end);
thetaWeightage = (thetaWithoutFirst'*thetaWithoutFirst);
J = ((1/m)*sumfact) + ((lambda/(2*m))*(thetaWeightage));
thetalength = length(theta);

%grad=sum(htheta-y)*xji%
%grad 100*1%
grad = ((1/m)*(X'*(htheta-y)+(lambda*[0;thetaWithoutFirst])));


% =============================================================

end
