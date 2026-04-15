function [Xn, mu, sigma] = normalize(X, method, mu_in, sigma_in)
    if nargin < 2 || isempty(method)
        method = '';
    end

    if nargin < 3
        mu_in = [];
    end

    if nargin < 4
        sigma_in = [];
    end

    if isempty(method)
        mu = zeros(1, size(X,2));
        sigma = ones(1, size(X,2));
        Xn = X;
        return;
    end

    switch lower(method)
        case 'zscore'
            if isempty(mu_in)
                mu = mean(X, 1);
            else
                mu = mu_in;
            end

            if isempty(sigma_in)
                sigma = std(X, 0, 1);
            else
                sigma = sigma_in;
            end

            sigma(sigma == 0) = 1;
            Xn = (X - mu) ./ sigma;

        otherwise
            error('Metodo de normalizacao nao suportado: %s', method);
    end
end
