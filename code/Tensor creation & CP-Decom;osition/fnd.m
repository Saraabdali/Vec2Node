function fnd(R,nn,window)

addpath('./tensor_toolbox_2.6/')
addpath('./tensor_toolbox_2.6/met')
filename='./code/SST2/All_SST2.csv';

sents=readtable(filename,'Delimiter',',');

label3=sents.label3;

Ystring=cellfun(@strsplit,cellstr(sents.textT),'UniformOutput',false);
[uniqueWords,dictionary,~,~,wordsSents] = createDictionary(Ystring);
Ystring=wordsSents;
for i=1:size(sents,1)
  sents.textF(i)=length(wordsSents{i});
end

indices = 1:length(dictionary);
dictionaryMap = containers.Map(dictionary, indices);
save('dictionary_SST2.mat','-v7.3','dictionaryMap');
disp('dictionaryMap finished');
tic

textT=Ystring;
textF=sents.textF;
label3=sents.label3;

[ntexts,ldic]=createwwnDiskSize(uniqueWords,textT,textF,dictionaryMap,window);
[wxwxn,wxwxnF,wxwxnL] = buildCoocTensor_TTA(ntexts,ldic);
disp('tensor done!');
X=cp_als(wxwxn,R);
[A,~] = generateWordsgraph(X.u{1},nn); 
[B,~] = generateWordsgraph(X.u{3},nn);
C=X.u{1};
disp('nngraph generated');
save(strcat('KNN_SST2_CP_Rank',num2str(R),'.mat'),'A','B','C');
