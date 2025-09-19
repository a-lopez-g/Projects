function pruebaMyUMAP()
    % Script para testear la clase MyUMAP

    % Crear una instancia de MyUMAP con los datos por defecto
    umapMinst = MyUMAP();

    % Entrenar el modelo UMAP y obtener la incrustación
    umapMinst = umapMinst.train_umap();

    % Mostrar diagnósticos si es necesario
    umapMinst.show_diagnostics();

    % Dibujar la incrustación obtenida con tu versión
    umapMinst.plot_umap();

    % Obtener los mismos datos para la versión de MATLAB
    X_train = umapMinst.X_train;
    y_train = umapMinst.y_train;

    % Correr la versión de UMAP de MATLAB
    % Especificar los parámetros que quieras comparar
    [embedding_matlab, umap_params] = run_umap(X_train, 'n_neighbors', umapMinst.N_NEIGHBOR, 'min_dist', umapMinst.MIN_DIST);

    % Mostrar los parámetros usados por run_umap
    disp('Parámetros usados por run_umap:');
    disp(umap_params);

    % Dibujar la incrustación obtenida con la versión de MATLAB
    figure;
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
