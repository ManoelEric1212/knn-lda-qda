function results = lda(X, y, use_normalization, normalization_method, class_names, feature_names, best_pair, save_prefix)

    if nargin < 3 || isempty(use_normalization)
        use_normalization = false;
    end
    if nargin < 4 || isempty(normalization_method)
        normalization_method = '';
    end
    if nargin < 7 || isempty(best_pair)
        best_pair = [1 2];
    end
    if nargin < 8 || isempty(save_prefix)
        save_prefix = 'lda';
    end

    n = size(X,1);
    y_pred = zeros(n,1);

    for i = 1:n
        train_idx = true(n,1);
        train_idx(i) = false;
        test_idx = ~train_idx;

        Xtr = X(train_idx, :);
        ytr = y(train_idx);
        Xte = X(test_idx, :);

        if use_normalization
            [Xtrn, mu, sigma] = normalize(Xtr, normalization_method);
            Xten = normalize(Xte, normalization_method, mu, sigma);
        else
            Xtrn = Xtr;
            Xten = Xte;
        end

        model = lda_train(Xtrn, ytr);
        y_pred(i) = lda_predict(model, Xten);
    end

    accuracy = mean(y_pred == y);

    % Modelo treinado em todos os dados para visualizacao
    if use_normalization
        [Xall, mu_all, sigma_all] = normalize(X, normalization_method);
    else
        Xall = X;
        mu_all = zeros(1, size(X,2));
        sigma_all = ones(1, size(X,2));
    end

    model_all = lda_train(Xall, y);
    plot_lda_model_2d(model_all, Xall, y, class_names, feature_names, best_pair, save_prefix);

    results = struct();
    results.accuracy = accuracy;
    results.y_pred = y_pred;
    results.use_normalization = use_normalization;
    results.normalization_method = normalization_method;
    results.mu = mu_all;
    results.sigma = sigma_all;
    results.model = model_all;
end


function model = lda_train(X, y)
    classes = unique(y);
    C = numel(classes);
    [n, d] = size(X);

    priors = zeros(C,1);
    means = zeros(C,d);
    Sigma = zeros(d,d);

    for i = 1:C
        c = classes(i);
        Xc = X(y == c, :);
        priors(i) = size(Xc,1) / n;
        means(i,:) = mean(Xc, 1);
    end

    for i = 1:C
        c = classes(i);
        Xc = X(y == c, :);
        centered = Xc - means(i,:);
        Sigma = Sigma + centered' * centered;
    end

    Sigma = Sigma / (n - C);
    Sigma = Sigma + 1e-6 * eye(d);  % regularizacao
    invSigma = inv(Sigma);

    model = struct();
    model.classes = classes;
    model.priors = priors;
    model.means = means;
    model.Sigma = Sigma;
    model.invSigma = invSigma;
end


function yhat = lda_predict(model, Xtest)
    if size(Xtest,1) == 1
        Xtest = reshape(Xtest, 1, []);
    end

    C = numel(model.classes);
    n = size(Xtest,1);
    scores = zeros(n, C);

    for i = 1:C
        mu = model.means(i,:);
        scores(:,i) = Xtest * model.invSigma * mu' ...
                    - 0.5 * (mu * model.invSigma * mu') ...
                    + log(model.priors(i));
    end

    [~, idx] = max(scores, [], 2);
    yhat = model.classes(idx);
end


function plot_lda_model_2d(model, X, y, class_names, feature_names, best_pair, save_prefix)
    f1 = best_pair(1);
    f2 = best_pair(2);

    x1_min = min(X(:,f1)) - 0.5;
    x1_max = max(X(:,f1)) + 0.5;
    x2_min = min(X(:,f2)) - 0.5;
    x2_max = max(X(:,f2)) + 0.5;

    grid_n = 200;
    [xx, yy] = meshgrid(linspace(x1_min, x1_max, grid_n), ...
                        linspace(x2_min, x2_max, grid_n));

    Xref = mean(X, 1);
    Xgrid = repmat(Xref, numel(xx), 1);
    Xgrid(:,f1) = xx(:);
    Xgrid(:,f2) = yy(:);

    Z = lda_predict(model, Xgrid);
    Z = reshape(Z, size(xx));

    fig = figure('visible', 'off');
    hold on;
    contourf(xx, yy, Z, numel(unique(y)), 'LineColor', 'none');
    plot_class_points(X, y, f1, f2);

    xlabel(feature_names{f1});
    ylabel(feature_names{f2});
    title('LDA - fronteira de decisao 2D');
    legend(class_names, 'Location', 'bestoutside');
    grid on;
    print(fig, sprintf('%s_model.png', save_prefix), '-dpng');
    close(fig);
end


function plot_class_points(X, y, f1, f2)
    markers = {'o', 's', '^', 'd', 'v', '>'};
    for c = 1:numel(unique(y))
        idx = (y == c);
        plot(X(idx,f1), X(idx,f2), markers{c}, 'MarkerSize', 6, 'LineWidth', 1.2);
    end
end