load('dictionary_SST2.mat');
keys=dictionaryMap.keys();
values=dictionaryMap.values();
fid = fopen('dictionary_SST2.tsv','wt');
 if fid>0
     fprintf(fid,'%s\t%s\n','key','value');
     for i=1:size(keys,2)
         if ~isnan(values{i})
            fprintf(fid,'%s\t%s\n',lower(keys{i}),string(values{i}-1));
         end
     end
     fclose(fid);
 end


