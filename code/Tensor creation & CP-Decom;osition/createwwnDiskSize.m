function [ntexts,ldic]=createwwnDiskSize(uniqueWords,text,textF,dictionaryMap,k)
mkdir IndicesTemp_TTA;
ntexts = size(text,1); %number of texts
ldic= double(dictionaryMap.Count);
%get 10% of dataset size 
s=floor(ntexts*0.1);
c=s;
aux=s;
subIndtexts=cell(1,c); 
valsTexts=cell(1,c);
g=1;
length(uniqueWords)
ntexts
for i=1:ntexts-1
    uw=uniqueWords{i};
    subInd=cell(1,length(uw));
    vals=cell(1,length(uw));
    l=1;
    for j=1:length(uw)

        idxlist=find(strcmp(text{i},uw(j)));
        lidxlist=length(idxlist);
        co_oclist=cell(1,lidxlist);
        if lidxlist==1
            
            co_oclist{1,1} = findcooc(idxlist,text{i},textF(i),k);
        else
            for h=1:lidxlist
                
                co_oclist{1,h}=findcooc(idxlist(h),text{i},textF(i),k);
                
            end
        end
        co_oclist = horzcat(co_oclist{:});
        [uniqueCooc,~,idxCooc] = unique(co_oclist);
        fabs=accumarray(idxCooc,1);
        if ~isempty(uniqueCooc)
            idx=cell2mat(values(dictionaryMap,uniqueCooc));
            iduw = dictionaryMap(uw{j});
            indx=ones(1,length(idx))';
            subInd{l}=[(indx*iduw) idx' (i*indx)];
            vals{l} = fabs;
        end
        l=l+1;
        clear idx co_oclist uniqueCooc idxCooc fabs indx;
        %clear lidxlist;
    end
    subIndTexts{1,g}=vertcat(subInd{:});
    valsTexts{1,g}=vertcat(vals{:});
    clear subInd;
    clear vals;
    g=g+1;
    if i == aux
        subIndTexts=vertcat(subIndTexts{:});
        valsTexts=vertcat(valsTexts{:});
        file_name= strcat('./IndicesTemp_TTA/subIndTextsVals_', num2str(i),'.mat');
        save(file_name,'-v7.3','subIndTexts','valsTexts');
        clear subIndTexts;
        clear valsTexts;
        subIndTexts=cell(1,c); 
        valsTexts=cell(1,c);
        g=1;
        aux=aux+c;
    end
end
subIndTexts=vertcat(subIndTexts{:});
valsTexts=vertcat(valsTexts{:});
file_name= strcat('./IndicesTemp_TTA/subIndTextsVals_', num2str(ntexts),'.mat');
save(file_name,'-v7.3','subIndTexts','valsTexts');
end

