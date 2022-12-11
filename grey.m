D = 'database2/Honzo Chan';
S = dir(fullfile(D,'*.jpg'));
for k = 1:numel(S)
    F = fullfile(D,S(k).name);
    S(k).data = imread(F);
end





path = 'database2/Honzo Chan';
for k = 1:numel(S)
    
    
    my = S(k).data;
    J = imrotate(my,-180,'bilinear','crop');
    gmy = rgb2gray(J);
    NNN = imresize(gmy, [300 300]);
    
    imshow(NNN);
    
    xx = string(k);
    png = append(xx,'.jpg');
    newpath = append(path,png);
    imwrite(NNN, newpath);  
end


poop = imread('database2/Su han/1.jpg');

whos poop 

% 
% path = 'database2/BM.jpg';
% 
% I = imread(path);
% J = imrotate(I,-180,'bilinear','crop');
% 
% imshow(J)
% 

















