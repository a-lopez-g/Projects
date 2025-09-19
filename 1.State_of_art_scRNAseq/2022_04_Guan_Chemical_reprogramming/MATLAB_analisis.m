
%% --- Descomprimimos datos en bruto ---
%gunzip("C:\matlab scripts\Quimica_Pluripotente_Humano_Guan\fibroblastos_embrionarios\hCIPSCS-1117\matrix.mtx.gz")
%gunzip("C:\matlab scripts\Quimica_Pluripotente_Humano_Guan\fibroblastos_embrionarios\hCIPSCS-1117\features.tsv.gz")
%gunzip("C:\matlab scripts\Quimica_Pluripotente_Humano_Guan\fibroblastos_embrionarios\hCIPSCS-1117\barcodes.tsv.gz")


%gunzip("C:\matlab scripts\Quimica_Pluripotente_Humano_Guan\H1\matrix.mtx.gz")
%gunzip("C:\matlab scripts\Quimica_Pluripotente_Humano_Guan\H1\features.tsv.gz")
%gunzip("C:\matlab scripts\Quimica_Pluripotente_Humano_Guan\H1\barcodes.tsv.gz")

%features = tdfread('features.tsv');
%barcodes = tdfread('C:\matlab scripts\Quimica_Pluripotente_Humano_Guan\H1\barcodes.tsv');
%matrix2 = load('C:\matlab scripts\Quimica_Pluripotente_Humano_Guan\H1\matrix.mtx');

%gen_extra = features.MIR13020x2D10(1,:)
%gen = features.ENSG00000243485(1,:);

%% ---- FIBROBLASTOS EMRBIONARIOS: Descomprimimos los datos preprocesados en Jupyter Notebook  ---
% Datos de células reprogramadas 
gunzip("C:\matlab scripts\Quimica_Pluripotente_Humano_Guan\fibroblastos_embrionarios\hCIPSCS-1117\gene_names.txt.gz");
gunzip("C:\matlab scripts\Quimica_Pluripotente_Humano_Guan\fibroblastos_embrionarios\hCIPSCS-1117\gene_id.txt.gz");
gunzip("C:\matlab scripts\Quimica_Pluripotente_Humano_Guan\fibroblastos_embrionarios\hCIPSCS-1117\feature_types.txt.gz");
gunzip("C:\matlab scripts\Quimica_Pluripotente_Humano_Guan\fibroblastos_embrionarios\hCIPSCS-1117\barcodes.txt.gz");
%gunzip("C:\matlab scripts\Quimica_Pluripotente_Humano_Guan\fibroblastos_embrionarios\hCIPSCS-1117\X.txt.gz");

% Datos de células sin reprogramar
gunzip("C:\matlab scripts\Quimica_Pluripotente_Humano_Guan\fibroblastos_embrionarios\hEFS-0330\gene_names.txt.gz");
gunzip("C:\matlab scripts\Quimica_Pluripotente_Humano_Guan\fibroblastos_embrionarios\hEFS-0330\gene_id.txt.gz");
gunzip("C:\matlab scripts\Quimica_Pluripotente_Humano_Guan\fibroblastos_embrionarios\hEFS-0330\feature_types.txt.gz");
gunzip("C:\matlab scripts\Quimica_Pluripotente_Humano_Guan\fibroblastos_embrionarios\hEFS-0330\barcodes.txt.gz");
%gunzip("C:\matlab scripts\Quimica_Pluripotente_Humano_Guan\fibroblastos_embrionarios\hEFS-0330\X.txt.gz");


%% --- Cargamos los datos ----
% Células reprogramadas
%X_reprog = load("C:\matlab scripts\Quimica_Pluripotente_Humano_Guan\fibroblastos_embrionarios\hCIPSCS-1117\X.txt");
gene_id_reprog_0330 = importdata("C:\matlab scripts\Quimica_Pluripotente_Humano_Guan\fibroblastos_embrionarios\hCIPSCS-1117\gene_id.txt");
barcodes_reprog_0330 = importdata("C:\matlab scripts\Quimica_Pluripotente_Humano_Guan\fibroblastos_embrionarios\hCIPSCS-1117\barcodes.txt");

% Células sin reprogramar
%X_sin_reprog = load("C:\matlab scripts\Quimica_Pluripotente_Humano_Guan\fibroblastos_embrionarios\hEFS-0330\X.txt");
gene_id_sin_reprog_0330 = importdata("C:\matlab scripts\Quimica_Pluripotente_Humano_Guan\fibroblastos_embrionarios\hEFS-0330\gene_id.txt");
barcodes_sin_reprog_0330 = importdata("C:\matlab scripts\Quimica_Pluripotente_Humano_Guan\fibroblastos_embrionarios\hEFS-0330\barcodes.txt");


%Comparamos los barcodes y gene_id entre reprogramadas y no reprogramadas 
gene_diff_0330 = setdiff(gene_id_reprog_0330,gene_id_sin_reprog_0330); % Todos los genes coinciden
barcode_diff_0330 = setdiff(barcodes_reprog_0330,barcodes_sin_reprog_0330); % 18800 diferentes


%% ---- FIBROBLASTOS PIEL ADULTOS: Descomprimimos los datos preprocesados en Jupyter Notebook  ---
% Datos de células reprogramadas 
gunzip("C:\matlab scripts\Quimica_Pluripotente_Humano_Guan\fibroblastos_piel_adultos\hCIPSCS-1230\gene_names.txt.gz");
gunzip("C:\matlab scripts\Quimica_Pluripotente_Humano_Guan\fibroblastos_piel_adultos\hCIPSCS-1230\gene_id.txt.gz");
gunzip("C:\matlab scripts\Quimica_Pluripotente_Humano_Guan\fibroblastos_piel_adultos\hCIPSCS-1230\feature_types.txt.gz");
gunzip("C:\matlab scripts\Quimica_Pluripotente_Humano_Guan\fibroblastos_piel_adultos\hCIPSCS-1230\barcodes.txt.gz");
%gunzip("C:\matlab scripts\Quimica_Pluripotente_Humano_Guan\fibroblastos_piel_adultos\hCIPSCS-1230\X.txt.gz");

% Datos de células sin reprogramar
gunzip("C:\matlab scripts\Quimica_Pluripotente_Humano_Guan\fibroblastos_piel_adultos\hASFS-0605\gene_names.txt.gz");
gunzip("C:\matlab scripts\Quimica_Pluripotente_Humano_Guan\fibroblastos_piel_adultos\hASFS-0605\gene_id.txt.gz");
gunzip("C:\matlab scripts\Quimica_Pluripotente_Humano_Guan\fibroblastos_piel_adultos\hASFS-0605\feature_types.txt.gz");
gunzip("C:\matlab scripts\Quimica_Pluripotente_Humano_Guan\fibroblastos_piel_adultos\hASFS-0605\barcodes.txt.gz");
%gunzip("C:\matlab scripts\Quimica_Pluripotente_Humano_Guan\fibroblastos_piel_adultos\hASFS-0605\X.txt.gz");


%% --- Cargamos los datos ----
% Células reprogramadas
X_reprog = load("C:\matlab scripts\Quimica_Pluripotente_Humano_Guan\fibroblastos_embrionarios\hCIPSCS-1117\X.txt");
% Vamos a usar la traspuesta, así habrá genes en las columnas y células en
% las filas
X_reprog_t = X_reprog.';

gene_id_reprog_1230 = importdata("C:\matlab scripts\Quimica_Pluripotente_Humano_Guan\fibroblastos_piel_adultos\hCIPSCS-1230\gene_id.txt");
barcodes_reprog_1230 = importdata("C:\matlab scripts\Quimica_Pluripotente_Humano_Guan\fibroblastos_piel_adultos\hCIPSCS-1230\barcodes.txt");
%%
% Células sin reprogramar
%X_sin_reprog = load("C:\matlab scripts\Quimica_Pluripotente_Humano_Guan\fibroblastos_embrionarios\hEFS-0330\X.txt");
gene_id_sin_reprog_1230 = importdata("C:\matlab scripts\Quimica_Pluripotente_Humano_Guan\fibroblastos_piel_adultos\hASFS-0605\gene_id.txt");
barcodes_sin_reprog_1230 = importdata("C:\matlab scripts\Quimica_Pluripotente_Humano_Guan\fibroblastos_piel_adultos\hASFS-0605\barcodes.txt");

%Comparamos los barcodes y gene_id entre reprogramadas y no reprogramadas 
gene_diff_1230 = setdiff(gene_id_reprog_1230,gene_id_sin_reprog_1230); 
barcode_diff_1230 = setdiff(barcodes_reprog_1230,barcodes_sin_reprog_1230); 

%% ---- C.MADRE DERIVADAS DE ADIPOSAS 0618: Descomprimimos los datos preprocesados en Jupyter Notebook  ---
% Datos de células reprogramadas 
gunzip("C:\matlab scripts\Quimica_Pluripotente_Humano_Guan\madre_mesenquimales_derivadas_de_adiposas\hCIPSCS-0618\gene_names.txt.gz");
gunzip("C:\matlab scripts\Quimica_Pluripotente_Humano_Guan\madre_mesenquimales_derivadas_de_adiposas\hCIPSCS-0618\gene_id.txt.gz");
gunzip("C:\matlab scripts\Quimica_Pluripotente_Humano_Guan\madre_mesenquimales_derivadas_de_adiposas\hCIPSCS-0618\feature_types.txt.gz");
gunzip("C:\matlab scripts\Quimica_Pluripotente_Humano_Guan\madre_mesenquimales_derivadas_de_adiposas\hCIPSCS-0618\barcodes.txt.gz");

% Datos de células sin reprogramar
gunzip("C:\matlab scripts\Quimica_Pluripotente_Humano_Guan\madre_mesenquimales_derivadas_de_adiposas\hADSCS-0618\gene_names.txt.gz");
gunzip("C:\matlab scripts\Quimica_Pluripotente_Humano_Guan\madre_mesenquimales_derivadas_de_adiposas\hADSCS-0618\gene_id.txt.gz");
gunzip("C:\matlab scripts\Quimica_Pluripotente_Humano_Guan\madre_mesenquimales_derivadas_de_adiposas\hADSCS-0618\feature_types.txt.gz");
gunzip("C:\matlab scripts\Quimica_Pluripotente_Humano_Guan\madre_mesenquimales_derivadas_de_adiposas\hADSCS-0618\barcodes.txt.gz");


%% -- Cargamos los datos -- 

% Células sin reprogramar
gene_id_sin_reprog_0618 = importdata("C:\matlab scripts\Quimica_Pluripotente_Humano_Guan\madre_mesenquimales_derivadas_de_adiposas\hADSCS-0618\gene_id.txt");
barcodes_sin_reprog_0618 = importdata("C:\matlab scripts\Quimica_Pluripotente_Humano_Guan\madre_mesenquimales_derivadas_de_adiposas\hADSCS-0618\barcodes.txt");

% Células reprogramdas
gene_id_reprog_0618 = importdata("C:\matlab scripts\Quimica_Pluripotente_Humano_Guan\madre_mesenquimales_derivadas_de_adiposas\hCIPSCS-0618\gene_id.txt");
barcodes_reprog_0618 = importdata("C:\matlab scripts\Quimica_Pluripotente_Humano_Guan\madre_mesenquimales_derivadas_de_adiposas\hCIPSCS-0618\barcodes.txt");

%Comparamos los barcodes y gene_id entre reprogramadas y no reprogramadas 
gene_diff_0618 = setdiff(gene_id_reprog_0618,gene_id_sin_reprog_0618); 
barcode_diff_0618 = setdiff(barcodes_reprog_0618,barcodes_sin_reprog_0618); 

%% ---- C.MADRE DERIVADAS DE ADIPOSAS 0809: Descomprimimos los datos preprocesados en Jupyter Notebook  ---
% Datos de células reprogramadas 
gunzip("C:\matlab scripts\Quimica_Pluripotente_Humano_Guan\madre_mesenquimales_derivadas_de_adiposas\hCIPSCS-0809\gene_names.txt.gz");
gunzip("C:\matlab scripts\Quimica_Pluripotente_Humano_Guan\madre_mesenquimales_derivadas_de_adiposas\hCIPSCS-0809\gene_id.txt.gz");
gunzip("C:\matlab scripts\Quimica_Pluripotente_Humano_Guan\madre_mesenquimales_derivadas_de_adiposas\hCIPSCS-0809\feature_types.txt.gz");
gunzip("C:\matlab scripts\Quimica_Pluripotente_Humano_Guan\madre_mesenquimales_derivadas_de_adiposas\hCIPSCS-0809\barcodes.txt.gz");

% Datos de células sin reprogramar
gunzip("C:\matlab scripts\Quimica_Pluripotente_Humano_Guan\madre_mesenquimales_derivadas_de_adiposas\hADSCS-0809\gene_names.txt.gz");
gunzip("C:\matlab scripts\Quimica_Pluripotente_Humano_Guan\madre_mesenquimales_derivadas_de_adiposas\hADSCS-0809\gene_id.txt.gz");
gunzip("C:\matlab scripts\Quimica_Pluripotente_Humano_Guan\madre_mesenquimales_derivadas_de_adiposas\hADSCS-0809\feature_types.txt.gz");
gunzip("C:\matlab scripts\Quimica_Pluripotente_Humano_Guan\madre_mesenquimales_derivadas_de_adiposas\hADSCS-0809\barcodes.txt.gz");


%% -- Cargamos los datos -- 

% Células sin reprogramar
gene_id_sin_reprog_0809 = importdata("C:\matlab scripts\Quimica_Pluripotente_Humano_Guan\madre_mesenquimales_derivadas_de_adiposas\hADSCS-0809\gene_id.txt");
barcodes_sin_reprog_0809 = importdata("C:\matlab scripts\Quimica_Pluripotente_Humano_Guan\madre_mesenquimales_derivadas_de_adiposas\hADSCS-0809\barcodes.txt");
 
% Células reprogramdas
gene_id_reprog_0809 = importdata("C:\matlab scripts\Quimica_Pluripotente_Humano_Guan\madre_mesenquimales_derivadas_de_adiposas\hCIPSCS-0809\gene_id.txt");
barcodes_reprog_0809 = importdata("C:\matlab scripts\Quimica_Pluripotente_Humano_Guan\madre_mesenquimales_derivadas_de_adiposas\hCIPSCS-0809\barcodes.txt");

%Comparamos los barcodes y gene_id entre reprogramadas y no reprogramadas 
gene_diff_0809 = setdiff(gene_id_reprog_0809,gene_id_sin_reprog_0809); 
barcode_diff_0809 = setdiff(barcodes_reprog_0809,barcodes_sin_reprog_0809); 

%% H1 CÉLULAS PLURIPOTENTES ORIGINALES: Descomprimimos los datos preprocesados en Jupyter Notebook

gunzip("C:\matlab scripts\Quimica_Pluripotente_Humano_Guan\h1\gene_names.txt.gz");
gunzip("C:\matlab scripts\Quimica_Pluripotente_Humano_Guan\h1\gene_id.txt.gz");
gunzip("C:\matlab scripts\Quimica_Pluripotente_Humano_Guan\h1\feature_types.txt.gz");
gunzip("C:\matlab scripts\Quimica_Pluripotente_Humano_Guan\h1\barcodes.txt.gz");

% --Cargamos los datos -- 

gene_id_h1 = importdata("C:\matlab scripts\Quimica_Pluripotente_Humano_Guan\h1\gene_id.txt");
barcodes_h1 = importdata("C:\matlab scripts\Quimica_Pluripotente_Humano_Guan\h1\barcodes.txt");

%% -- COMPARAMOS LOS BARCODES CON LOS DE H1 PARA VER SI COINCIDEN -- 
barcodes_0330_reprog_h1 = ismember(barcodes_sin_reprog_0330,barcodes_h1); 
sum(barcodes_0330_reprog_h1==1)
barcodes_0330_sin_reprog_h1 = ismember(barcodes_sin_reprog_0330,barcodes_h1);
sum(barcodes_0330_sin_reprog_h1==1)
