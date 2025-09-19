function pruebaMyUMAPv2()
    % Script para testear la clase MyUMAP
    tic
    % Crear una instancia de MyUMAP con los datos por defecto
    umapMinst = MyUMAPv2();

    % Entrenar el modelo UMAP y obtener la incrustación
    umapMinst = umapMinst.train_umap();
    toc
    % Mostrar diagnósticos si es necesario
    umapMinst.show_diagnostics();

    % Dibujar la incrustación obtenida con tu versión
    %umapMinst.plot_umap();

    % Obtener los mismos datos para la versión de MATLAB
    X_train = umapMinst.X_train;
    y_train = umapMinst.y_train;

    %Dibujar la incrustación obtenida con la versión de MATLAB
    figure('Position', [100, 100, 800, 600]);  % [x, y, width, height]
    subplot(131)
    scatter(umapMinst.init(:, 1), umapMinst.init(:, 2), 50, y_train, 'filled');
    colormap('jet');
    xlabel('h_1');
    ylabel('h_2');
    colorbar;
    title('Intialization embedding');


    subplot(132)
    scatter(umapMinst.embedding(:, 1), umapMinst.embedding(:, 2), 50, y_train, 'filled');
    colormap('jet');
    xlabel('h_1');
    ylabel('h_2');
    colorbar;
    title('Custom UMAP  embedding');

    
    % Correr la versión de UMAP de MATLAB
    % Especificar los parámetros que quieras comparar
    [embedding_matlab, umap_params] = run_umap(X_train,'init',umapMinst.init, 'n_neighbors', umapMinst.N_NEIGHBOR, 'min_dist', umapMinst.MIN_DIST,'verbose','text');

    % Mostrar los parámetros usados por run_umap
    disp('Parámetros usados por run_umap:');
    disp(umap_params);

    % Dibujar la incrustación obtenida con la versión de MATLAB
    subplot(133)
    scatter(embedding_matlab(:, 1), embedding_matlab(:, 2), 50, y_train, 'filled');
    colormap('jet');
    xlabel('h_1');
    ylabel('h_2');
    colorbar;
    title('UMAP Embedding - MATLAB Version');

    % Comparar parámetros
    fprintf('Comparación de parámetros:\n');
    fprintf('Número de vecinos (MyUMAP vs run_umap): %d vs %d\n', umapMinst.N_NEIGHBOR, umap_params.n_neighbors);
    fprintf('Distancia mínima (MyUMAP vs run_umap): %.2f vs %.2f\n', umapMinst.MIN_DIST, umap_params.min_dist);
end
