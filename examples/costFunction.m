function J = costFunctionJ(X, y, theta)

  % X is the *design matrix* containing our training examples
  % y is the class lables

  m = size(X, 1); % number of traing exmaples
  predictions = X*theta; % predictions of the hypothesis on all m examples
  sqrErrors = (predictions-y).^2;

  J = 1/(2*m) * sum(sqrErrors);
