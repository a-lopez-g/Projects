<<<<<<< HEAD
%% --- Descomprimimos --- 
%gunzip("C:\matlab scripts\Llorenc_Aging_Piel_Humano\X_filtered.txt.gz");
gunzip("C:\matlab scripts\Llorenc_Aging_Piel_Humano\GSE130973_barcodes_filtered.tsv.gz");

%% --- Cargamos la matriz X para ver que está todo bien ---
X = load("C:\matlab scripts\Llorenc_Aging_Piel_Humano\X_filtered.txt");

%% --- Genes --- 
gunzip("C:\matlab scripts\Llorenc_Aging_Piel_Humano\gene_names.txt.gz");
gunzip("C:\matlab scripts\Llorenc_Aging_Piel_Humano\gene_id.txt.gz");

%% --- Cargamos los datos de los genes --- 
gene_id = importdata("C:\matlab scripts\Llorenc_Aging_Piel_Humano\gene_id.txt");
gene_names = importdata("C:\matlab scripts\Llorenc_Aging_Piel_Humano\gene_names.txt");

%% --- Células, barcodes --- 
gunzip("C:\matlab scripts\Llorenc_Aging_Piel_Humano\barcodes.txt.gz");

%% --- Cargamos los datos de las células --- 
barcodes = importdata("C:\matlab scripts\Llorenc_Aging_Piel_Humano\barcodes.txt");

%% --- Etiqueta de salida ---
% Será un 0 si los hombres son jóvenes (-1,-2 en barcodes)
% Será un 1 si los hombres son adultos (-3,-4,-5 en barcodes)


%% --- Transponemos matriz X --- 
writematrix(X.', "X_filtered_transposed.txt");

%% --- Comprimimos la matriz X traspuesta ---
=======
%% --- Descomprimimos --- 
%gunzip("C:\matlab scripts\Llorenc_Aging_Piel_Humano\X_filtered.txt.gz");
gunzip("C:\matlab scripts\Llorenc_Aging_Piel_Humano\GSE130973_barcodes_filtered.tsv.gz");

%% --- Cargamos la matriz X para ver que está todo bien ---
X = load("C:\matlab scripts\Llorenc_Aging_Piel_Humano\X_filtered.txt");

%% --- Genes --- 
gunzip("C:\matlab scripts\Llorenc_Aging_Piel_Humano\gene_names.txt.gz");
gunzip("C:\matlab scripts\Llorenc_Aging_Piel_Humano\gene_id.txt.gz");

%% --- Cargamos los datos de los genes --- 
gene_id = importdata("C:\matlab scripts\Llorenc_Aging_Piel_Humano\gene_id.txt");
gene_names = importdata("C:\matlab scripts\Llorenc_Aging_Piel_Humano\gene_names.txt");

%% --- Células, barcodes --- 
gunzip("C:\matlab scripts\Llorenc_Aging_Piel_Humano\barcodes.txt.gz");

%% --- Cargamos los datos de las células --- 
barcodes = importdata("C:\matlab scripts\Llorenc_Aging_Piel_Humano\barcodes.txt");

%% --- Etiqueta de salida ---
% Será un 0 si los hombres son jóvenes (-1,-2 en barcodes)
% Será un 1 si los hombres son adultos (-3,-4,-5 en barcodes)


%% --- Transponemos matriz X --- 
writematrix(X.', "X_filtered_transposed.txt");

%% --- Comprimimos la matriz X traspuesta ---
>>>>>>> 8cd151ba1ef24b10d08902f35c3e6583971e2ffe
gzip("X_filtered_transposed.txt");