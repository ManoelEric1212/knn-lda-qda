clc;
clear;
close all;
addpath('data');



filePath = 'data/column_3c.dat';
% filePath = 'data/column_3c.dat';
% Carrega os dados
feature_names = {'pelvic incidence', 'pelvic tilt', 'lumbar lordosis angle', 'sacral slope', 'pelvic radius', 'degree spondylolisthesis'};
% class_names = {'AB', 'NO'};
class_names = {'DH', 'SL','NO'}
[X, y, labels_text] = load_data(filePath,class_names,' ');


use_normalization = true;        
normalization_method = 'zscore';


fprintf('Carregando base de dados...\n');
fprintf('Arquivo: %s\n', filePath);
fprintf('Numero de amostras: %d\n', size(X,1));
fprintf('Numero de features: %d\n', size(X,2));
fprintf('Numero de classes: %d\n', numel(class_names));
fprintf('\nDistribuicao das classes:\n');




best_pair = best_feature_pair(X, y);

fprintf('\nMelhor par de atributos: (%d, %d)\n', best_pair(1), best_pair(2));
fprintf('Features: %s e %s\n', ...
    feature_names{best_pair(1)}, ...
    feature_names{best_pair(2)});

Xp = X(:, best_pair);

figure;
hold on;
grid on;

markers = {'o', 's', '^', 'd', 'v', '>'};
colors = lines(numel(unique(y)));

classes = unique(y);

for c = 1:numel(classes)
    idx = (y == classes(c));

    scatter(Xp(idx,1), Xp(idx,2), 60, ...
        'Marker', markers{c}, ...
        'MarkerEdgeColor', colors(c,:), ...
        'DisplayName', class_names{c});
end

xlabel(feature_names{best_pair(1)});
ylabel(feature_names{best_pair(2)});
title('Distribuicao das classes (melhor par de atributos)');
legend('Location', 'best');

for c = 1:numel(class_names)
    fprintf('  %s -> %d amostras\n', class_names{c}, sum(y == c));
end
fprintf('\n');


% KNN

k_values = 1:5;
fprintf('Executando k-NN com Leave-One-Out...\n');

knn_results = knn(X, y, k_values, use_normalization, normalization_method,class_names, feature_names, best_pair, 'outputs/knn');

fprintf('\n=== Resultado k-NN ===\n');
for i = 1:numel(knn_results.k_values)
    fprintf('k = %d -> acuracia = %.4f (%.2f%%)\n', ...
        knn_results.k_values(i), ...
        knn_results.accuracies(i), ...
        100 * knn_results.accuracies(i));
end
fprintf('Melhor k = %d\n', knn_results.best_k);
fprintf('Melhor acuracia = %.4f (%.2f%%)\n', ...
    knn_results.best_accuracy, 100 * knn_results.best_accuracy);


% LDA

fprintf('\nExecutando LDA com Leave-One-Out...\n');

lda_results = lda(X, y, use_normalization, normalization_method,class_names, feature_names, best_pair, 'outputs/lda');

fprintf('\n=== Resultado LDA ===\n');
fprintf('Acuracia = %.4f (%.2f%%)\n', ...
    lda_results.accuracy, 100 * lda_results.accuracy);


% QDA

fprintf('\nExecutando QDA com Leave-One-Out...\n');

qda_results = qda(X, y, use_normalization, normalization_method,class_names, feature_names, best_pair, 'outputs/qda');

fprintf('\n=== Resultado QDA ===\n');
fprintf('Acuracia = %.4f (%.2f%%)\n', ...
    qda_results.accuracy, 100 * qda_results.accuracy);


% RESUMO FINAL

fprintf('\n================ RESUMO FINAL ================\n');
fprintf('k-NN (melhor k = %d): %.4f (%.2f%%)\n', ...
    knn_results.best_k, knn_results.best_accuracy, 100 * knn_results.best_accuracy);
fprintf('LDA: %.4f (%.2f%%)\n', ...
    lda_results.accuracy, 100 * lda_results.accuracy);
fprintf('QDA: %.4f (%.2f%%)\n', ...
    qda_results.accuracy, 100 * qda_results.accuracy);

fprintf('\nArquivos de figura gerados:\n');
fprintf('  - knn_accuracy.png\n');
fprintf('  - knn_best_model.png\n');
fprintf('  - lda_model.png\n');
fprintf('  - qda_model.png\n');