function results = knn(X, y, k_values, use_normalization, normalization_method, class_names, feature_names, best_pair, save_prefix)

    if nargin < 4 || isempty(use_normalization)
        use_normalization = false;
    end
    if nargin < 5 || isempty(normalization_method)
        normalization_method = '';
    end
    if nargin < 8 || isempty(best_pair)
        best_pair = [1 2];
    end
    if nargin < 9 || isempty(save_prefix)
        save_prefix = 'knn';
    end

    n = size(X,1);
    num_classes = numel(unique(y));
    accuracies = zeros(numel(k_values), 1);

    for kk = 1:numel(k_values)
        k = k_values(kk);
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

            y_pred(i) = knn_predict_single(Xtrn, ytr, Xten, k, num_classes);
        end

        accuracies(kk) = mean(y_pred == y);
    end

    [best_accuracy, best_idx] = max(accuracies);
    best_k = k_values(best_idx);

    % Grafico de acuracia
    fig1 = figure('visible', 'off');
    plot(k_values, accuracies * 100, '-o', 'LineWidth', 2, 'MarkerSize', 8);
    grid on;
    xlabel('k');
    ylabel('Acuracia (%)');
    title('k-NN com Leave-One-Out');
    xlim([min(k_values), max(k_values)]);
    print(fig1, sprintf('%s_accuracy.png', save_prefix), '-dpng');
    close(fig1);

    % Normalizacao para visualizacao do modelo
    if use_normalization
        [Xall, mu_all, sigma_all] = normalize(X, normalization_method);
    else
        Xall = X;
        mu_all = zeros(1, size(X,2));
        sigma_all = ones(1, size(X,2));
    end

    plot_knn_model_2d(Xall, y, best_k, num_classes, class_names, feature_names, best_pair, save_prefix);

    results = struct();
    results.k_values = k_values(:);
    results.accuracies = accuracies(:);
    results.best_k = best_k;
    results.best_accuracy = best_accuracy;
    results.use_normalization = use_normalization;
    results.normalization_method = normalization_method;
    results.mu = mu_all;
    results.sigma = sigma_all;
end


function yhat = knn_predict_single(Xtr, ytr, x, k, num_classes)
    dists = sqrt(sum((Xtr - x).^2, 2));
    [~, idx] = sort(dists, 'ascend');
    neighbors = ytr(idx(1:k));

    counts = zeros(num_classes, 1);
    for c = 1:num_classes
        counts(c) = sum(neighbors == c);
    end

    max_count = max(counts);
    candidates = find(counts == max_count);

    if numel(candidates) == 1
        yhat = candidates;
    else
        for ii = 1:k
            if any(candidates == ytr(idx(ii)))
                yhat = ytr(idx(ii));
                return;
            end
        end
    end
end


function plot_knn_model_2d(X, y, best_k, num_classes, class_names, feature_names, best_pair, save_prefix)
    f1 = best_pair(1);
    f2 = best_pair(2);

    x1_min = min(X(:,f1)) - 0.5;
    x1_max = max(X(:,f1)) + 0.5;
    x2_min = min(X(:,f2)) - 0.5;
    x2_max = max(X(:,f2)) + 0.5;

    grid_n = 200;
    [xx, yy] = meshgrid(linspace(x1_min, x1_max, grid_n),linspace(x2_min, x2_max, grid_n));

    Xref = mean(X, 1);
    Z = zeros(size(xx));

    for i = 1:size(xx,1)
        for j = 1:size(xx,2)
            xg = Xref;
            xg(f1) = xx(i,j);
            xg(f2) = yy(i,j);
            Z(i,j) = knn_predict_single(X, y, xg, best_k, num_classes);
        end
    end

    fig2 = figure('visible', 'off');
    hold on;
    contourf(xx, yy, Z, num_classes, 'LineColor', 'none');
    plot_class_points(X, y, f1, f2);

    xlabel(feature_names{f1});
    ylabel(feature_names{f2});
    title(sprintf('k-NN (k = %d) - fronteira de decisao 2D', best_k));
    legend(class_names, 'Location', 'bestoutside');
    grid on;
    print(fig2, sprintf('%s_best_model.png', save_prefix), '-dpng');
    close(fig2);
end


function plot_class_points(X, y, f1, f2)
    markers = {'o', 's', '^', 'd', 'v', '>'};
    for c = 1:numel(unique(y))
        idx = (y == c);
        plot(X(idx,f1), X(idx,f2), markers{c}, 'MarkerSize', 6, 'LineWidth', 1.2);
    end
end