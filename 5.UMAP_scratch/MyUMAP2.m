classdef MyUMAP2
    % MyUMAP2: Implementación personalizada de UMAP con opción supervisada.
    %
    % Esta clase implementa UMAP tanto en versión no supervisada como
    % supervisada. En el modo supervisado, se considera la información de
    % las etiquetas durante el entrenamiento para mejorar la separación
    % entre clases en el espacio embebido.
    %
    % Propiedades:
    %   - X_train, y_train, X_test, y_test: Conjuntos de datos.
    %   - N_NEIGHBOR: Número de vecinos para la construcción de las probabilidades.
    %   - MIN_DIST: Distancia mínima para las proyecciones en baja dimensión.
    %   - LEARNING_RATE: Tasa de aprendizaje para la optimización.
    %   - MAX_ITER: Número máximo de iteraciones para el descenso de gradiente.
    %   - N_LOW_DIMS: Dimensionalidad del espacio proyectado.
    %   - ShowDiagnostics: Mostrar diagnósticos durante el entrenamiento.
    %   - numsub: Tamaño del subconjunto de datos utilizado.
    %   - CE_array: Matriz para almacenar la evolución de la entropía cruzada.
    %   - embedding: Resultado final de la proyección UMAP.
    %   - Supervised: Booleano para indicar si es supervisado o no.
    %
    % Métodos:
    %   - train_umap: Entrena el modelo UMAP.
    %   - show_diagnostics: Muestra diagnósticos de distancias y evolución de CE.
    %   - plot_umap: Dibuja el resultado del embedding.
    %   - CE_supervised: Función de coste modificada para la versión supervisada.

    properties
        X_train
        y_train
        X_test
        y_test
        N_NEIGHBOR = 15
        MIN_DIST = 0.25
        LEARNING_RATE = 0.25
        MAX_ITER = 500
        N_LOW_DIMS = 2
        ShowDiagnostics = false
        numsub = 1000; % Número de imágenes para seleccionar aleatoriamente
        CE_array % Propiedad para almacenar los valores de CE durante el entrenamiento
        embedding % Propiedad para almacenar el embedding final
        Supervised = false % Booleano para indicar si es supervisado
    end

    methods
        function obj = MyUMAP2(Xtr, Ytr, Xts, Yts, Supervised, ShowDiagnostics)
            % Constructor que inicializa los datos de entrenamiento y prueba
            
            if nargin < 4 || isempty(Xtr) || isempty(Ytr) || isempty(Xts) || isempty(Yts)
                % Generar datos por defecto usando myReadMinstStraight
                rng(2), close all,
                [X_train_full, ~, y_train_full, X_test_full, ~, y_test_full] = myReadMinstStraight();
                X_train_full = X_train_full';
                y_train_full = y_train_full';
                X_test_full = X_test_full';
                y_test_full = y_test_full';

                % Filtrar los datos para seleccionar solo los dígitos '1' y '7'
                selected_idx = y_train_full == 1 | y_train_full == 7;
                X_train_full = X_train_full(selected_idx, :);
                y_train_full = y_train_full(selected_idx);

                selected_idx = y_test_full == 1 | y_test_full == 7;
                X_test_full = X_test_full(selected_idx, :);
                y_test_full = y_test_full(selected_idx);
            else
                % Inicializar con datos proporcionados
                X_train_full = Xtr;
                y_train_full = Ytr;
                X_test_full = Xts;
                y_test_full = Yts;
            end

            % Ajustar obj.numsub si es mayor que el número de muestras disponibles
            if obj.numsub > size(X_train_full, 1)
                obj.numsub = size(X_train_full, 1);
            end

            rng(0);
            idxTrain = randperm(size(X_train_full, 1), obj.numsub);
            idxTest = randperm(size(X_test_full, 1), obj.numsub);

            obj.X_train = X_train_full(idxTrain, :);
            obj.y_train = y_train_full(idxTrain);
            obj.X_test = X_test_full(idxTest, :);
            obj.y_test = y_test_full(idxTest);

            if nargin >= 5 && ~isempty(Supervised)
                obj.Supervised = Supervised;
            end

            if nargin >= 6 && ~isempty(ShowDiagnostics)
                obj.ShowDiagnostics = ShowDiagnostics;
            end

            % Inicializar propiedades
            obj.CE_array = zeros(1, obj.MAX_ITER);
        end

        function obj = train_umap(obj)
            % Calcular distancias euclídeas entre las muestras
            dist = squareform(pdist(obj.X_train, 'euclidean'));

            % Calcular el segundo valor más pequeño de cada fila de distancias
            rho = zeros(size(dist, 1), 1);
            for i = 1:size(dist, 1)
                sorted_distances = sort(dist(i, :));
                rho(i) = sorted_distances(2);
            end

            % Construir probabilidades en alta dimensión
            n = size(obj.X_train, 1);
            prob = zeros(n, n);
            sigma_array = zeros(n, 1);
            for dist_row = 1:n
                func = @(sigma) obj.myk(obj.prob_high_dim(sigma, dist_row, dist, rho));
                binary_search_result = obj.sigma_binary_search(func, obj.N_NEIGHBOR);
                prob(dist_row, :) = obj.prob_high_dim(binary_search_result, dist_row, dist, rho);
                sigma_array(dist_row) = binary_search_result;
            end

            % Calcular la matriz P como el promedio de prob y su transpuesta
            P = (prob + prob') / 2;

            % Construir probabilidades en baja dimensión
            x = linspace(0, 3, 300);
            y_values = obj.myf(x, obj.MIN_DIST);

            % Ajustar la curva utilizando lsqcurvefit
            initial_guess = [1, 1];
            options = optimset('Display', 'off');
            [p, ~] = lsqcurvefit(@(p, x) obj.dist_low_dim(p, x), initial_guess, x, y_values, [], [], options);

            a = p(1);
            b = p(2);
            fprintf('Hyperparameters a = %f and b = %f\n', a, b);

            % Aprender incrustaciones en baja dimensión
            rng(12345);
            obj.embedding = obj.randomInitialization(obj.X_train, obj.N_LOW_DIMS);
            disp('Running Gradient Descent:');
            for i = 1:obj.MAX_ITER
                if obj.Supervised
                    obj.embedding = obj.embedding - obj.LEARNING_RATE * obj.CE_supervised(P, obj.embedding, a, b);
                else
                    obj.embedding = obj.embedding - obj.LEARNING_RATE * obj.CE_gradient(P, obj.embedding, a, b);
                end

                obj.CE_array(i) = sum(sum(obj.CE(P, obj.embedding, a, b))) / 1e5;

                if mod(i, 50) == 0
                    fprintf('Iteration %d of %d\n', i, obj.MAX_ITER);
                end
            end
        end

        function show_diagnostics(obj)
            % Calcular las distancias euclídeas entre las muestras
            dist = squareform(pdist(obj.X_train, 'euclidean'));

            % Calcular el segundo valor más pequeño de cada fila de distancias (rho)
            rho = zeros(size(dist, 1), 1);
            for i = 1:size(dist, 1)
                sorted_distances = sort(dist(i, :));
                rho(i) = sorted_distances(2);
            end

            % Construir la figura y los subplots
            figure(10);

            % Histograma de distancias
            subplot(3, 2, 1);
            hist_dist = histogram(dist(:), 100);
            title('Histogram of Distances');
            xlim([min(hist_dist.BinEdges), max(hist_dist.BinEdges)]); % Fijar los ejes para todos los histogramas

            % Histograma de rho
            subplot(3, 2, 3);
            hist_rho = histogram(rho, 50);
            title('Histogram of Rho');
            xlim([min(hist_dist.BinEdges), max(hist_dist.BinEdges)]); % Fijar los ejes para todos los histogramas

            % Calcular sigma_array
            N_NEIGHBOR = obj.N_NEIGHBOR;
            sigma_array = zeros(size(dist, 1), 1);
            for dist_row = 1:size(dist, 1)
                func = @(sigma) obj.myk(obj.prob_high_dim(sigma, dist_row, dist, rho));
                sigma_array(dist_row) = obj.sigma_binary_search(func, N_NEIGHBOR);
            end

            % Histograma de sigma
            subplot(3, 2, 5);
            hist_sigma = histogram(sigma_array, 100);
            title('Histogram of Sigmas');
            xlim([min(hist_dist.BinEdges), max(hist_dist.BinEdges)]); % Fijar los ejes para todos los histogramas

            % Evolución de la entropía cruzada
            subplot(3, 2, [2 4 6]);
            plot(1:obj.MAX_ITER, obj.CE_array);
            title('Cross-Entropy Evolution');
            xlabel('Iteration');
            ylabel('Cross-Entropy');
        end

        function plot_umap(obj)
            % Dibujar el scatter plot del resultado de UMAP
            % Convertir y_train a un vector numérico si es necesario
            y_train_numeric = double(obj.y_train); % Asegúrate de que sea numérico

            % Verificar si la cantidad de etiquetas coincide con la cantidad de puntos
            if length(y_train_numeric) ~= size(obj.embedding, 1)
                error('El tamaño de y_train no coincide con el número de puntos en embedding.');
            end

            figure;
            scatter(obj.embedding(:, 1), obj.embedding(:, 2), 50, y_train_numeric, 'filled');
            colormap('jet'); xlabel('h_1'); ylabel('h_2'); colorbar;
        end

        function prob = prob_high_dim(~, sigma, dist_row, dist, rho)
            d = dist(dist_row, :) - rho(dist_row);
            d(d < 0) = 0;
            prob = exp(-d / sigma);
        end

        function n_neighbor = myk(~, prob)
            n_neighbor = 2 ^ sum(prob);
        end

        function approx_sigma = sigma_binary_search(~, k_of_sigma, fixed_k)
            sigma_lower_limit = 0;
            sigma_upper_limit = 1000;
            for i = 1:20
                approx_sigma = (sigma_lower_limit + sigma_upper_limit) / 2;
                if k_of_sigma(approx_sigma) < fixed_k
                    sigma_lower_limit = approx_sigma;
                else
                    sigma_upper_limit = approx_sigma;
                end
                if abs(fixed_k - k_of_sigma(approx_sigma)) <= 1e-5
                    break;
                end
            end
        end

        function y = myf(~, x, min_dist)
            y = zeros(size(x));
            for i = 1:length(x)
                if x(i) <= min_dist
                    y(i) = 1;
                else
                    y(i) = exp(-x(i) + min_dist);
                end
            end
        end

        function inv_distances = prob_low_dim(~, Y, a, b)
            distances = squareform(pdist(Y, 'euclidean'));
            inv_distances = (1 + a * distances.^b).^-1;
        end

        function ce = CE(obj, P, Y, a, b)
            Q = obj.prob_low_dim(Y, a, b);
            ce = -P .* log(Q + 0.01) - (1 - P) .* log(1 - Q + 0.01);
        end

        function grad = CE_gradient(~, P, Y, a, b)
            y_diff = permute(Y, [1, 3, 2]) - permute(Y, [3, 1, 2]);
            distances = squareform(pdist(Y, 'euclidean').^2);
            inv_dist = (1 + a * distances.^b).^-1;
            Q = (1 - P) * (0.001 + distances).^-1;
            Q(logical(eye(size(Q)))) = 0;
            Q = Q ./ sum(Q, 2);

            fact = a * P .* (1e-8 + distances).^(b-1) - Q;

            inv_dist = reshape(inv_dist, [size(inv_dist, 1), size(inv_dist, 2), 1]);

            grad = 2 * b * sum(fact .* y_diff .* inv_dist, 2);
            grad = squeeze(grad);
        end

        function grad = CE_supervised(obj, P, Y, a, b)
            % Función de coste supervisada para UMAP
            % Añade un término adicional para penalizar puntos con la misma etiqueta pero que están lejos.
            y_diff = permute(Y, [1, 3, 2]) - permute(Y, [3, 1, 2]);
            distances = squareform(pdist(Y, 'euclidean').^2);
            inv_dist = (1 + a * distances.^b).^-1;
            Q = (1 - P) * (0.001 + distances).^-1;
            Q(logical(eye(size(Q)))) = 0;
            Q = Q ./ sum(Q, 2);

            % Penalizar puntos con la misma etiqueta
            label_penalty = double(obj.y_train' == obj.y_train);
            label_penalty(logical(eye(size(label_penalty)))) = 0;

            fact = a * P .* (1e-8 + distances).^(b-1) - Q - label_penalty .* distances;

            inv_dist = reshape(inv_dist, [size(inv_dist, 1), size(inv_dist, 2), 1]);

            grad = 2 * b * sum(fact .* y_diff .* inv_dist, 2);
            grad = squeeze(grad);
        end

        function Y = randomInitialization(~, X, n_components)
            rng(12345);
            Y = randn(size(X, 1), n_components);
        end

        function y_fit = dist_low_dim(~, p, x)
            % Definir la función dist_low_dim que se ajustará
            a = p(1);
            b = p(2);
            y_fit = 1 ./ (1 + a * x.^(2 * b));
        end
    end
end

