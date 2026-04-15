function [X, y, labels] = load_data(filePath, targetNames, delimiter)
    % filePath: String com o caminho do arquivo
    % targetNames: Cell array de strings, ex: {'AB', 'NO'}
    % delimiter: (Opcional) ',' ou ' ' ou '\t'. Se omitido, tenta detectar.

    fid = fopen(filePath, 'r');
    if fid == -1
        error('Could not open file: %s', filePath);
    end

    X = [];
    labels = {};

    line = fgetl(fid);
    while ischar(line)
        line = strtrim(line);
        if ~isempty(line)
            if nargin < 3 || isempty(delimiter)
                if contains(line, ',')
                    current_delimiter = ',';
                else
                    current_delimiter = ''; 
                end
            else
                current_delimiter = delimiter;
            end
            if isempty(current_delimiter)
                parts = strsplit(line); 
            else
                parts = strsplit(line, current_delimiter);
            end

            if numel(parts) >= 2
                features = str2double(parts(1:end-1));
                class_name = strtrim(parts{end});

                X = [X; features];
                labels{end+1, 1} = class_name;
            end
        end
        line = fgetl(fid);
    end

    fclose(fid);

    numSamples = length(labels);
    y = zeros(numSamples, 1);
    for i = 1:numSamples
        idx = find(strcmp(targetNames, labels{i}));
        if ~isempty(idx)
            %y(i) = idx -1;
            y(i) = idx;
        else
            error('Class found "%s" not present in targetNames', labels{i});
        end
    end
end