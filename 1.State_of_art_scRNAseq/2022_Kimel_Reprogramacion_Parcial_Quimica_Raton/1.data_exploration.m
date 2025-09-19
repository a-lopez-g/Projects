load('datos_MSX1.mat','X','y');

[vv,ii] = mode(X);
X(:,ii/size(X,1)>=0.95) = [];
%{
params_ML=[];
N = size(X,1);
i = 1;
while true
    disp(i);
    parmhat = nbinfit(X(:,i));
    if parmhat(1)==Inf
        continue
    else
        params_ML = [params_ML; parmhat];
    end
    histogram(nbinrnd(parmhat(1),parmhat(2),N,1),'normalization','pdf')
    hold on; 
    histogram(X(:,i),'normalization','pdf'); 
    hold off;
    
    k = waitforbuttonpress;
    value = double(get(gcf,'CurrentCharacter'));
    if value==27
        close;
        break;
    elseif value==28
        if i==1
            continue;
        else
            i = i - 1;
        end
    elseif value==29
        if i==size(X,2)
            continue;
        else
            i = i + 1;
        end
    end
end
%}
% Correlations
C = corr(X);
lim_sup = 0.2;
lim_inf = 0.1;
[f,c]=find(C-diag(diag(C))>lim_inf & C-diag(diag(C))<lim_sup);

params_ML=[];
N = size(X,1);
i = 1;
while true
    disp(string(i)+"/"+string(size(f,1)));
    disp(string(f(i))+"-"+string(c(i)));
    disp("Corr: "+string(C(f(i),c(i))));
    parmhat1 = nbinfit(X(:,f(i)));
    parmhat2 = nbinfit(X(:,c(i)));
    if parmhat1(1)==Inf || parmhat2(1)==Inf
        continue;
    else
        params_ML = [params_ML; [f(i),parmhat1,c(i),parmhat2,C(f(i),c(i))]];
    end
    subplot(2,1,1)
    data_generated1 = nbinrnd(parmhat1(1),parmhat1(2),N,1);
    histogram(data_generated1,'normalization','pdf')
    hold on; 
    histogram(X(:,f(i)),'normalization','pdf'); 
    hold off;
    title(string(f(i)))

    subplot(2,1,2)
    data_generated2 = nbinrnd(parmhat2(1),parmhat2(2),N,1);
    histogram(data_generated2,'normalization','pdf')
    hold on; 
    histogram(X(:,c(i)),'normalization','pdf'); 
    hold off;
    title(string(c(i)))

    cc = corrcoef(data_generated1,data_generated2);
    disp("Corr2: "+string(cc(1,2)));
    
    k = waitforbuttonpress;
    value = double(get(gcf,'CurrentCharacter'));
    if value==27
        close;
        break;
    elseif value==28
        if i==1
            continue;
        else
            i = i - 1;
        end
    elseif value==29
        if i==size(f,1)
            continue;
        else
            i = i + 1;
        end
    end
end