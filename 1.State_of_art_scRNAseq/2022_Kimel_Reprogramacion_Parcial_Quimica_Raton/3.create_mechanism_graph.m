load("info_MSX1.mat");
names_T = readtable("names_MSX1.csv", 'ReadVariableNames', false);
names = table2cell(names_T)';

NumVar = [];
NamVar = [];
thrDiff = 0.9:-0.1:0.1;
thrCW = 0.9:-0.1:0.1;
for i=1:length(thrDiff)
    disp("---- " + num2str(thrDiff(i)))
    for j=1:length(thrCW)
        disp("-- " + num2str(thrCW(j)))
        cW2 = cW; 
        cW2(~triu(((cW-cX)>thrDiff(i)) - cW>thrCW(j))) = 0;
        [ff,cc] = find(cW2~=0);
        if sum(sum(cW2))==0
            NumVar(i,j) = 0;
        else
            NumVar(i,j) = size(unique([ff;cc]),1);
        end
    end
end

save('NumVar.mat',"NumVar")

showInteractiveGraph_v2(cW,cX,importancias,names,relevantes)