function best_pair = best_feature_pair(X, y)

    d = size(X,2);
    best_score = -inf;
    best_pair = [1 2];

    classes = unique(y);
    C = numel(classes);

    for i = 1:d-1
        for j = i+1:d

            Xp = X(:, [i j]);

            % Media global
            mu = mean(Xp);

            Sw = zeros(2,2);
            Sb = zeros(2,2);

            for c = 1:C
                Xc = Xp(y == classes(c), :);
                nc = size(Xc,1);
                muc = mean(Xc);

                % intra-classe
                Sw = Sw + (Xc - muc)' * (Xc - muc);

                % inter-classe
                diff = (muc - mu)';
                Sb = Sb + nc * (diff * diff');
            end

            % Score
            J = trace(pinv(Sw) * Sb);

            if J > best_score
                best_score = J;
                best_pair = [i j];
            end
        end
    end
end