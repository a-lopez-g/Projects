classdef MyUMAPv2
    properties
        X_train
        y_train
        X_test
        y_test
        N_NEIGHBOR = 15
        MIN_DIST = 0.25
        LEARNING_RATE = 0.01 
        MAX_ITER = 1000
        N_LOW_DIMS = 2
        ShowDiagnostics = false
        batch_size = 100 % (New) Batch Size SGD
        numsub = 1000; % Número de imágenes para seleccionar aleatoriamente
        CE_array % Propiedad para almacenar los valores de CE durante el entrenamiento
        embedding % Propiedad para almacenar el embedding final
        init = 'spectral' % (New) Property initial embedding
    end

    methods
        function obj = MyUMAPv2(Xtr, Ytr, Xts, Yts, ShowDiagnostics)
            % Constructor que inicializa los datos de entrenamiento y prueba
            if nargin == 0
                % Generar datos por defecto usando myReadMinstStraight
                rng(2), close all,
                [X_train_full, ~, y_train_full, X_test_full, ~, y_test_full] = myReadMinstStraight();
                X_train_full = X_train_full';
                y_train_full = y_train_full';
                X_test_full = X_test_full';
                y_test_full = y_test_full';
            else
                % Inicializar con datos proporcionados
                X_train_full = Xtr;
                y_train_full = Ytr;
                X_test_full = Xts;
                y_test_full = Yts;
            end

            % Seleccionar un subconjunto aleatorio de imágenes
            rng(0);
            idxTrain = randperm(size(X_train_full, 1), obj.numsub);
            idxTest = randperm(size(X_test_full, 1), obj.numsub);

            obj.X_train = X_train_full(idxTrain, :);
            obj.y_train = y_train_full(idxTrain);
            obj.X_test = X_test_full(idxTest, :);
            obj.y_test = y_test_full(idxTest);

            if nargin >= 5
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
            P = (prob + prob' - (prob.*prob')); % (New) Simetry Formula
            %P = (prob + prob') / 2;

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
            obj.embedding = obj.initialize_embedding(P); 

            % (New) Initilization Method
            obj.init = obj.embedding; %Store initialization embedding
            disp('Running Stochasctic Gradient Descent:');

            % (New) STOCHASTIC GRADIENT DESCENT
            for i = 1:obj.MAX_ITER
                 % Randomly shuffle indices for batch selection
                indices = randperm(size(obj.embedding, 1)); 
                
                for batch_start = 1:obj.batch_size:size(obj.embedding, 1)
                    % Select a batch of data
                    batch_indices = indices(batch_start:min(batch_start + obj.batch_size - 1, size(obj.embedding, 1)));
                    
                    % Compute gradient for the batch
                    obj.embedding(batch_indices,:)  = obj.embedding(batch_indices,:) - obj.LEARNING_RATE * obj.CE_gradient(P(batch_indices,batch_indices), obj.embedding(batch_indices,:) , a, b);

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
            sigma_array = zeros(size(dist, 1), 1);
            for dist_row = 1:size(dist, 1)
                func = @(sigma) obj.myk(obj.prob_high_dim(sigma, dist_row, dist, rho));
                sigma_array(dist_row) = obj.sigma_binary_search(func, obj.N_NEIGHBOR);
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

        function prob = prob_high_dim(obj, sigma, dist_row, dist, rho)
            % (New) High Dimensional probablity calculation. Nearest
            % k-neighbours
            N = size(dist, 1);
            prob = zeros(1, N);
            [ordered_dists,indices] = sort(dist(dist_row,:));
            for j = 2:obj.N_NEIGHBOR+1
                dist_diff = max(0, ordered_dists(j) - rho(indices(j)));
                if dist_diff <= dist_row
                    prob(1, indices(j)) = 1;
                else
                    prob(1, indices(j)) = exp(-dist_diff / sigma);
                end
            end
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
                if abs(fixed_k - k_of_sigma(approx_sigma)) <= 1e-8
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

        % (New) Initilization options
        function embedding = initialize_embedding(obj, graph)
            if strcmp(obj.init, 'spectral')
                embedding = obj.spectral_layout(graph);
            else
                embedding = rand(size(graph, 1), obj.N_LOW_DIMS);
            end
        end

        % (New) Spectral Layout Initialization
        function embedding = spectral_layout(obj, graph)
            diag_vals = 1 ./ sqrt(sum(graph, 2));
            D = spdiags(diag_vals, 0, size(graph, 1), size(graph, 1));
            L = speye(size(graph, 1)) - D * graph * D;
            k = obj.N_LOW_DIMS + 1;
            [eigenvecs, ~] = eigs(L, k, 'smallestabs');
            embedding = eigenvecs(:, 2:k);
        end


        function y_fit = dist_low_dim(~, p, x)
            % Definir la función dist_low_dim que se ajustará
            a = p(1);
            b = p(2);
            y_fit = 1 ./ (1 + a * x.^(2 * b));
        end
    end
end
