function [words]=findcooc(idx,text,textF,k)
    i=idx-k;
    j=idx+k;
    len=textF;
    if (idx==1 && j>len && len==1)
            words=text(idx);
    else
            if (i<=0 && j>len)
               words= horzcat(text(1:(idx-1)),text((idx+1):end));
            else
                 if (i<=0 && j>0 && j<=len)
                words= horzcat(text(1:(idx-1)),text((idx+1):(j)));
                 else     
                    if (i>0 && j>0 && j<=len )
                        words= horzcat(text(i:(idx-1)),text((idx+1):(j)));
                    else
                        if (i>0 && idx<len && j>len)
                         words= horzcat(text((i):(idx-1)),text((idx+1):end));
                        else
                           if (i>0 && idx==len)
                                 words= text((i):(idx-1));
                           else
                                  i=idx-k
                                  j=idx+k 
                                  len=textF
        
                           end
                        end
                    end
                 end
            end
    end
 
    
%     if (idx == 1 && j<size(text,1))
%          words= text((idx+1):(j))
%     else
%         if (size(text,1)==1)
%          words= text((size(text,1)));
%         end
%          if (size(text,1)~=1 && idx == 1 && j>size(text,1))
%              words= text((idx+1):(size(text,1)));
%          end
% 
%         if (idx == size(text,1) && i>0)
%             words = text((i):(idx-1));
%         else
%             if ((j) > str2double(textF))
%                 if ((i)<1)
% 			words= horzcat(text(1:(idx-1)),text((idx+1):end));
% 		else
%                 	words= horzcat(text((i):(idx-1)),text((idx+1):end));
%             	end
% 	    else
%                 if ((i) < 1 && idx>1)
%                     words= horzcat(text(1:(idx-1)),text((idx+1):(j)));
%                 else
%                     if(j>size(text,1) && idx==1)
%                       words= text(1);
%                     else
%                     words= horzcat(text((i):(idx-1)),text((idx+1):(j)));
%                     end
%                 end
%             end
%         end
%     end  
end

