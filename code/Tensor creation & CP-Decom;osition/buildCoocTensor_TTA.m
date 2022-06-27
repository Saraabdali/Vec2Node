function [wxwxn,wxwxnF,wxwxnL] =buildCoocTensor_TTA(ntexts,ldic)
myFolder = './IndicesTemp_TTA/';
filePattern = fullfile(myFolder, 'subIndTextsVals_*.mat');
matFiles = dir(filePattern);
subInd = cell( length(matFiles),1);
vals = cell( length(matFiles),1);
valslog = cell( length(matFiles),1);
for k = 1:length(matFiles)
  baseFileName = matFiles(k).name;
  fullFileName = fullfile(myFolder, baseFileName);
  fprintf(1, 'Reading %s\n', baseFileName);
  load(fullFileName);
  subInd{k,1}=subIndTexts;
  vals{k,1}=valsTexts; %remove log 
  valslog{k,1}=log(valsTexts);
  clear subIndTexts;
  clear valsTexts;
end
tic
subInd=vertcat(subInd{:});
vals=vertcat(vals{:});
valslog=vertcat(valslog{:});
disp('concatenation finished');
toc
tic
wxwxn=sptensor(subInd,1,[ldic,ldic,ntexts]);
wxwxnF=sptensor(subInd,vals,[ldic,ldic,ntexts]);
wxwxnL=sptensor(subInd,valslog,[ldic,ldic,ntexts]);
rmdir('IndicesTemp_TTA', 's');
disp('sptensor finished');
end
