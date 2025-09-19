<<<<<<< HEAD
load('datos_MSX1.mat','X','y');
names_T = readtable("names_MSX1.csv", 'ReadVariableNames', false);
names = table2cell(names_T)';

[vv,ii] = mode(X);
X(:,ii/size(X,1)>=0.95) = [];

names(:,ii/size(X,1)>=0.95) = [];

=======
load('datos_MSX1.mat','X','y');
names_T = readtable("names_MSX1.csv", 'ReadVariableNames', false);
names = table2cell(names_T)';

[vv,ii] = mode(X);
X(:,ii/size(X,1)>=0.95) = [];

names(:,ii/size(X,1)>=0.95) = [];

>>>>>>> 8cd151ba1ef24b10d08902f35c3e6583971e2ffe
writecell(names, 'names_msx1_filtrado.csv')