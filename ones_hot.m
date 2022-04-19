function yp = ones_hot(flag, y)
    % This function will be used to convert between integer labels and
    % ones-hot labels. If the labels are integers and there are C classes,
    % then the labels will be one of {1, 2, ..., C}. If the labels are in
    % ones-hot form, then each label will be a Cx1 vector containing all
    % zeros except one element whose index corresponds to the class. Note
    % that in the case of binary classification (i.e., either zero or one),
    % then this won't touch the output labels.
    %
    % INPUTS:
    %       flag (str)              Flag toggling between integer labels or
    %                               ones-hot labels
    %       y (Mx1 array)           Original output of integer labels
    %         (CxM array)           Original output of ones-hot labels
    % 
    % OUTPUT:
    %       yp (Mx1 array)          New output of integer labels
    %          (CxM array)          New output of ones-hot labels
    
    % Handle the case of conversion from integer to ones-hot labels. This 
    % will be used primarily when training the neural network since the 
    % output layer activations will be a CxM matrix.
    if (strcmp(flag, 'ones-hot'))
        
        % Handle the case of binary classification - the new output array
        % should equal the transpose of the old output array. The transpose
        % is taken since keeping track of all the dimensions in this is
        % difficult.
        if (max(y) == 1)
            yp = y';
            
        % Handle the case of multi-class classification.
        else

            % Get the number of training examples and initialize the new
            % output matrix. Note that since the mathematics are annoying,
            % I am flipping the output matrix so that it is a CxM matrix
            % instead of an MxC matrix.
            M = size(y, 1);
            yp = zeros(max(y), M);

            % Iterate over each training example and get its corresponding
            % integer label. Use this to find the corresponding element in
            % its ones-hot vector representation to set equal to one.
            for i = 1: M
                yp(y(i), i) = 1;
            end
        end

    % Handle the case of conversion from ones-hot to integer labels. This 
    % will be used primarily when making predictions with the neural 
    % network.
    elseif (strcmp(flag, 'integer'))
        
        % Handle the case of binary classification - the new output array
        % should equal the transpose of the old output array. The transpose
        % is taken since keeping track of all the dimensions in this is
        % difficult.
        if (size(y, 1) == 1)
            yp = y';
            
        % Handle the case of multi-class classification.
        else

            % Get the number of training examples and initialize the new
            % output matrix. Note that in order to jibe with the original
            % output matrix, I am once again taking the transpose of the
            % results (similar to binary classification).
            M = size(y, 2);
            yp = zeros(M, 1);

            % Iterate over each training example and find where the
            % ones-hot label is equal to one. The index corresponding to
            % this point will be the integer label.
            for i = 1: M
                yp(i) = find(y(:, i) == 1);
            end
        end
    end
end