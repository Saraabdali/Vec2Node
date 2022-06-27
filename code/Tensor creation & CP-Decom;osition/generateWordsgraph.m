function [A, linkwords] = generateWordsgraph(C,n_neighbors)
    Mdl = KDTreeSearcher(C);
	nn=knnsearch(Mdl,C,'k',n_neighbors+1);
    disp('knnsearch done!');
    linkwords = knnInd(nn); %function to generate links
    disp('knnInd!');
    wordsGraph= digraph(linkwords(:,1),linkwords(:,2));
    A = adjacency(wordsGraph); % no symmetric matrix
    A=A+A';
    A=spones(A); %symmetric
end

