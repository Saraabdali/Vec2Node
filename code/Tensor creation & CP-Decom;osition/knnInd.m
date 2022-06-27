function linkwords = knnInd(nn)
    ln=size(nn,2);
    lwords=size(nn,1);
    linkwords = zeros((ln-1)*lwords,2);
    l=1;
    for i=1:lwords
        for j=2:ln
            linkwords(l,:)=[i,nn(i,j)];
            l=l+1;
        end
    end
end
