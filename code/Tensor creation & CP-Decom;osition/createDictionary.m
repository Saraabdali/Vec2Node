function [uniqueWords,dictionary,tf,n_tf,wordsSents] = createDictionary(wordsSents)
        % find words with length <= 40
        
        idswords=cellfun(@(x)find(strlength(x)<=40 & strlength(x)>0),wordsSents,'UniformOutput',false);
        % remove words with lenght >40
         wordsSents=cellfun(@(x,y) x(y),wordsSents,idswords,'UniformOutput',false);
        % unique words per Text
        [uniqueWords,~,idxWord] = cellfun(@unique,wordsSents,'UniformOutput',false);
        uniqueWords = cellfun(@transpose,uniqueWords,'UniformOutput',false);
        ln=length(wordsSents);
        tf=cell(1,ln);
        n_tf=cell(1,ln);
        for i=1:ln
            tf{i}=accumarray(idxWord{i},1); % frequency abs. of words
            n_tf{i}=tf{i}/length(wordsSents{i}); 
        end

         uniqueWords(any(cellfun(@isempty,uniqueWords),2),:) = [];
         dictionary  = unique(cat(1, uniqueWords{:}));% corpus
         
%          dictionary
         
end
