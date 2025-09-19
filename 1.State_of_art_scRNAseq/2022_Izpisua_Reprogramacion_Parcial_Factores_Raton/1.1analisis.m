<<<<<<< HEAD
%% --- Cargamos los datos: tenemos dos matrices --- %%
%una de Dox - (control) y otra Dox + (reprogramadas). 
% Son archivos .h5, tiene la matriz (matrix) que a su vex tienen 1 grupo
% (features) y 5 datasets (barcodes, data, indprt, indices, shape)

% El id de los genes está en "features", "id"
% La matriz de conteos en "data"
%gunzip("C:\matlab scripts\Yamanaka_Parcial_Raton_Izpisua\X_filtered_dox_plus.txt.gz")
X_reprog = load("C:\matlab scripts\Yamanaka_Parcial_Raton_Izpisua\X_filtered_dox_plus.txt");
%% --- Estructura --- %%
% Dox +
h5info("C:\matlab scripts\Yamanaka_Parcial_Raton_Izpisua\GSE144600_liver4F_SC_plusDox_filtered_feature_bc_matrix.h5");
h5disp("C:\matlab scripts\Yamanaka_Parcial_Raton_Izpisua\GSE144600_liver4F_SC_plusDox_filtered_feature_bc_matrix.h5");

% Dox -
%h5info("C:\matlab scripts\Yamanaka_Parcial_Raton_Izpisua\GSE144600_liver4F_SC_minusDox_filtered_feature_bc_matrix.h5");
%h5disp("C:\matlab scripts\Yamanaka_Parcial_Raton_Izpisua\GSE144600_liver4F_SC_minusDox_filtered_feature_bc_matrix.h5");

%% --- Matriz de conteo -> data --- %%
X_dox_plus = h5read('./GSE144600_liver4F_SC_plusDox_filtered_feature_bc_matrix.h5','/matrix/data');
X_dox_minus = h5read('./GSE144600_liver4F_SC_minusDox_filtered_feature_bc_matrix.h5','/matrix/data');

%% --- Id de los genes -> features, id --- %%
gene_id_dox_plus = h5read('./GSE144600_liver4F_SC_plusDox_filtered_feature_bc_matrix.h5','/matrix/features/id');
gene_id_dox_minus = h5read('./GSE144600_liver4F_SC_minusDox_filtered_feature_bc_matrix.h5','/matrix/features/id');

writecell(gene_id_dox_plus,'gene_id_dox_plus.txt')
writecell(gene_id_dox_minus,'gene_id_dox_minus.txt')

%% --- Barcodes de las células -> barcodes --- %%
barcodes_dox_plus = h5read('./GSE144600_liver4F_SC_plusDox_filtered_feature_bc_matrix.h5','/matrix/barcodes');
barcodes_dox_minus = h5read('./GSE144600_liver4F_SC_minusDox_filtered_feature_bc_matrix.h5','/matrix/barcodes');

writecell(barcodes_dox_plus,'barcodes_dox_plus.txt')
writecell(barcodes_dox_minus,'barcodes_dox_minus.txt')

%% --- Comprimimos --- 
gzip('barcodes_dox_plus.txt');
gzip('barcodes_dox_minus.txt');
gzip('gene_id_dox_plus.txt');
gzip('gene_id_dox_minus.txt');
=======
%% --- Cargamos los datos: tenemos dos matrices --- %%
%una de Dox - (control) y otra Dox + (reprogramadas). 
% Son archivos .h5, tiene la matriz (matrix) que a su vex tienen 1 grupo
% (features) y 5 datasets (barcodes, data, indprt, indices, shape)

% El id de los genes está en "features", "id"
% La matriz de conteos en "data"
%gunzip("C:\matlab scripts\Yamanaka_Parcial_Raton_Izpisua\X_filtered_dox_plus.txt.gz")
X_reprog = load("C:\matlab scripts\Yamanaka_Parcial_Raton_Izpisua\X_filtered_dox_plus.txt");
%% --- Estructura --- %%
% Dox +
h5info("C:\matlab scripts\Yamanaka_Parcial_Raton_Izpisua\GSE144600_liver4F_SC_plusDox_filtered_feature_bc_matrix.h5");
h5disp("C:\matlab scripts\Yamanaka_Parcial_Raton_Izpisua\GSE144600_liver4F_SC_plusDox_filtered_feature_bc_matrix.h5");

% Dox -
%h5info("C:\matlab scripts\Yamanaka_Parcial_Raton_Izpisua\GSE144600_liver4F_SC_minusDox_filtered_feature_bc_matrix.h5");
%h5disp("C:\matlab scripts\Yamanaka_Parcial_Raton_Izpisua\GSE144600_liver4F_SC_minusDox_filtered_feature_bc_matrix.h5");

%% --- Matriz de conteo -> data --- %%
X_dox_plus = h5read('./GSE144600_liver4F_SC_plusDox_filtered_feature_bc_matrix.h5','/matrix/data');
X_dox_minus = h5read('./GSE144600_liver4F_SC_minusDox_filtered_feature_bc_matrix.h5','/matrix/data');

%% --- Id de los genes -> features, id --- %%
gene_id_dox_plus = h5read('./GSE144600_liver4F_SC_plusDox_filtered_feature_bc_matrix.h5','/matrix/features/id');
gene_id_dox_minus = h5read('./GSE144600_liver4F_SC_minusDox_filtered_feature_bc_matrix.h5','/matrix/features/id');

writecell(gene_id_dox_plus,'gene_id_dox_plus.txt')
writecell(gene_id_dox_minus,'gene_id_dox_minus.txt')

%% --- Barcodes de las células -> barcodes --- %%
barcodes_dox_plus = h5read('./GSE144600_liver4F_SC_plusDox_filtered_feature_bc_matrix.h5','/matrix/barcodes');
barcodes_dox_minus = h5read('./GSE144600_liver4F_SC_minusDox_filtered_feature_bc_matrix.h5','/matrix/barcodes');

writecell(barcodes_dox_plus,'barcodes_dox_plus.txt')
writecell(barcodes_dox_minus,'barcodes_dox_minus.txt')

%% --- Comprimimos --- 
gzip('barcodes_dox_plus.txt');
gzip('barcodes_dox_minus.txt');
gzip('gene_id_dox_plus.txt');
gzip('gene_id_dox_minus.txt');
>>>>>>> 8cd151ba1ef24b10d08902f35c3e6583971e2ffe
