%% 
% vypis jakoby před kamerou
ii = imread("56.jpg");

imshow(ii)

label=classify(net,ii)

%%
%rozeznání osoby před kamerou

c=webcam;
%load RESnet;
faceDetector=vision.CascadeObjectDetector;
while true
    e=c.snapshot;
    bboxes =step(faceDetector,e);
    if(sum(sum(bboxes))~=0)
    es=imcrop(e,bboxes(1,:));
    es=imresize(es,[224 224]);
    label=classify(net,es);
    image(e);
    title(char(label));
    drawnow;
    else
        image(e);
        title('No Face Detected');
    end
end

