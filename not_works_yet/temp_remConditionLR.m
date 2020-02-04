def remConditionLR(y, fCondition, maxCon= None, X= None, varargin= None):
    """
    Find condition fCondition and try to exclude it by deleting elements of
  vector y.
    :param y: 
    :param fCondition: function(y, x) wich returns mask of elements which not satisfy condition. 
    :param maxCon: function(y, x) wich returns mask of elements which need to be removeed first
    :param X: additional argument for fCondition and maxCon of size same as y.
    coordinates to find maxCon by minimising distance(X) for 'diff(y)<0'
condition

    :param varargin: additional argument for fCondition and maxCon
    :return: (y_out, indGood)
    *y_out: result of all deleting of y
    *indGood: indexes of y_out in y

Algorithm Remarks:
If ~isempty(maxCon) effective for exclude huge number of small zones of 
condition: if maxCon==True  or  maxCon=='R' (right), then bad data in 
condition zone and also at next point, or only at next point if zone=1point.
 If maxCon==False  or  maxCon=='L' then do the same but with previous point

If condition coused by spyke, can delete mach after it if maxCon==True or 
'R', or before it if maxCon==False or 'L'. Use bSpyke, bSingleSpikes or 
FSFilter (which them includes) before apply remConditionLR or set maxCon 
None:
for condition 'diff(y)<d.ddd..' or 'diff(y)>d.ddd\', this  
apply effective algorithm for (number of bad zones)<(number of bad points)
by removing minimum poits keepeng one edge of each bad zone consequently.

Example:
[y ind]= remConditionLR(R.Depth, @(y) diff(y)<0, @(X) X, bIf(R.SpeedUp>0,0,R.SpeedUp))
    """
# \                        iS   iE[k] iS[k+1]
#  \_______________  b_bad: g  g  b     g  g  g
#  |\      /\              .     ?     ?
#  | \    /| \                .  
#  |  \  / |  \                  .     ^ 
#  |   \/__| __\_____                     .
#  |   |   |   |\                            .
#  | ? |del| ? | \
#X[1]X[2]X(3)X(4)
#1)    i0,k|   i1   
#2)i0      k,i1    

n= len(y)
indGood= np.arange(n)
bUseBCondition= False
str_func= getattr(fCondition.__code__,'co_code') # 'co_argcount', 'co_cellvars', 'co_code', 'co_consts', 'co_name', 'co_names'func2str(fCondition)
if(n==1) and (size(y,2)>1)
  k= size(y,2)
  if str_func.find('y[:,{:d}]'.format(k))<0
    warning('remConditionLR:mayBe_badArrayShape', \
      'size(y)==[1 %d] may be bad array shape!', k)
  end
  return
end

str_arg= getattr(fCondition.__code__,'co_code')
str_func

bUseX= (nargin>=4) and ~isempty(X)
if nargin<3 or isempty(maxCon) or isa(maxCon, 'function_handle'):
  b_EfAlg= True
  fCondMod= lambda y,P: fCondition(y-P) #'diff(y)'-> y-P

  if nargin>=3:
    if isa(maxCon, 'function_handle'):
      bUseBCondition= True
      if ~bUseX:
        X= np.arange(n)
    else:
      b= maxCon
else:
  b_EfAlg= False
  if nargin<3:
  	b= [] 
  elif maxCon==False or maxCon=='L':
    b= False
  elif maxCon==True  or maxCon=='R':
    b= True
  else:
  	b= maxCon

if b_EfAlg:
  while True:
    b_bad= fCondition(y,varargin);
    if any(b_bad):
      bShift= np.zeros_like(y, np.bool8)
      #plot(y); hold on
      
      #starts of intervals and its good identification:
      i_diff= np.flatnonzero(ediff1d(b_bad, to_begin= 0))
      bi_bad= b_bad[i_diff]
      #starts and ends of good intervals
      ilast= len(y)-1
      iE= np.append(i_diff[bi_bad], ilast)
      if b_bad[0]:
      	#if first bad then set first good interval empty?
        if b_bad[-1]:
          iS= np.hstack((0, i_diff[~bi_bad], ilast))
        else:
          iS= [1, i_diff[~bi_bad]]
        k= int32(1)
        bShift[1:(iS[2]-1)]= True
      else:
        if b_bad[-1]:
          iS=    [i_diff[~bi_bad], size(y,1)]
        else:
          iS=     i_diff[~bi_bad]
        k= int32(0)
      #i_diff= i_diff[bi_bad]
      kEnd=sum(bi_bad)
      while k<kEnd:
        k= k+1
        # Find possible bad bounds [i0 i1] from adjacent good fCondition intervals
        #i0: (last good + 1) from start (it is not 1st bad!)
        i0= int32(np.flatnonzero(fCondMod(y(iS[k]:iE[k],:), varargin, y(iS[k+1],:)), 1, 'last')); 
        if isempty(i0); if k>1; i0=iE[k-1]; else i0= iS[k]; end
        else i0= i0 + iS[k] - 1; end #convert to global indexes
        
        # Search to +inf:
        #i1: (first good - 1) from next start (it is not last bad!)
        i1= int32(np.flatnonzero(fCondMod(y(iE[k],:), varargin,y(iS[k+1]:end,:)), 1,'first'))
        if i1>1; i1= i1+iS[k+1]-2; else i1= iS[k+1]; end #convert to global indexes
        if bUseBCondition  and  i0<i1
          if i0==1; ind0= i0:(iE[k]); else ind0= (i0-1):(iE[k]); end
          n= int32(numel(ind0))
          w0= maxCon(X(ind0))**2
          
          ind1= iS[k+1]:i1; n1= int32(numel(ind1)); w1= maxCon(X(ind1))**2
          t1= ones(1,n,'int32')
          
          t0= np.flatnonzero(fCondMod(y[ind1,:], varargin,y(ind0(1),:)), 1,'last')
          if ~isempty(t0); t1[1]= t0; end #else t1[1]= 1
          for t0= 2:n
            t0Max= np.flatnonzero(fCondMod(y[ind1,:], varargin,y(ind0(t0),:)), 1,'last'); #last bad condition
            if isempty(t0Max); t1[t0]= n1
            else               t1[t0]= t0Max
            end
          end
          w0sum= cumsum(fliplr(w0))./(1:double(n))'
          w1sum= cumsum(w1)./(1:double(n1))'
          b= [diff(t1)>0, True]
          [~, t0Max]= max(w0sum(b)+w1sum(t1[b]))
          w0sum=np.flatnonzero(b); #temporary
          t0Max= w0sum(t0Max)
          t1Max= t1[t0Max]
          #figure(); hold on; plot([iE[k] iS[k+1]],-y([iE[k] iS[k+1]],1), 'r*');plot(ind0(1):(i1+5),-y(ind0(1):(i1+5),1), 'r', [i0 i1],-y[[i0 i1],1], 'b.'); plot([ind0(1)+t0Max-1 iS[k+1]+t1Max-1],-y([ind0(1)+t0Max-1 iS[k+1]+t1Max-1],1), 'g.')
          if i0==1; i0= ind0(1) + t0Max - 1
          else      i0= ind0(1) + t0Max
          end                   #not delete t0Max point if exists before
          i1= ind1(1) + t1Max - 1;  #delete t1Max
          bShift[i0:i1]= True
          
          if (k<kEnd) and (i1>=iS[k+2])  #(k<kEnd) and (i1>=iE[k+2])
            k= k+np.flatnonzero(i1>=iS((k+2):end), 1, 'last')
          end
        else
          if bUseX
            b= abs(X(iE[k])-X(i0))>abs(X(i1)-X(iS[k+1]))
          else
            b= (iE[k]-i0)>(i1-iS[k+1])
          end
          if b #figure(); plot(indGood, -y[:,1], '.b')
            bShift(iE[k]:i1)= True; #(iE[k]+1)
            #figure(); hold on; plot([iE[k] iS[k+1]],-y([iE[k] iS[k+1]],1), 'r*'); plot(ind0(1):(i1+5),-y(ind0(1):(i1+5),1), 'r'); plot([i0 i1],-y[[i0 i1],1], 'b.')
          else #plot([i0 i1],-y[[i0 i1],1], 'g.'); #plot(i0:(i1+5),X(i0:(i1+5)),'k')
            bShift(i0:(iS[k+1]-1))= True; #iS[k+1]
            #plot(iS[k+1],-y(iS[k+1]), 'r*'); plot(i0:(iS[k+1]-1),-y(i0:(iS[k+1]-1)), 'r')
          end #figure(); plot(-y[:,1], '.r'); hold on
        end
      end
      #hold off            #figure(); hold on; plot(indGood,y[:,1], 'c')
     y[bShift]=   []      #figure(); hold on; plot(indGood,y[:,1], '.k')
      indGood[bShift]= []  #plot(indGood,y[:,1], '.b'); plot(indGood,y[:,1], '.b')
      if bUseX:
       X[bShift]= []
    else: #exit cycle hold on
      return

elif nargin<3:
  error('cpecify algoritm: b') 
elif b==True:
  #Check and delete all conditions points + next point
  for k= indGood
    b_bad= fCondition(y,varargin)
    if any(b_bad)
      #bShift= [False, (diff([False b_bad])<0)]
      bShift= [False; b_bad]; #all conditions points + next point
     y[bShift,:]= []
      indGood[bShift]= []
    else
      return
    end
  end
  fprintf(1, '\nremConditionLR(#0.0f points)/ not success!', numel(y))
  return
elif(b==False) #Check and delete all conditions points + previous point
    for k= indGood
        b_bad= fCondition(y,varargin); #diff(y)<=0
        if any(b_bad)
           y[b_bad,:]= []
            indGood[b_bad]= []
        else
            return
        end
    end
    fprintf(1, '\nremConditionLR(#0.0f points)/ not success!', numel(y))
elif(~isempty(b)) # and (b~=True) and (b~=False)
  ##Delete points of condition interleavingly
  while(1)    
    b_bad= fCondition(y,varargin); #diff(y)<=0
    if any(b_bad)
      if b==True
        b= False
        bShift= [False; (diff([False; b_bad])>0)]
      else
        b= True
        bShift= diff([False; b_bad])>0
      end
     y[bShift]=    []
      indGood[bShift]=  []
    else
      return
    end
  end
end
indGood= []; y= []
    
