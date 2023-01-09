clc;
clear all
close all
cao=webcam;
faceDetector=vision.CascadeObjectDetector;
c=1;
temp=0;
personname = "Pablo";
mkdir ../datastorage Pablo

while true
    e=cao.snapshot;
    bboxes =step(faceDetector,e);
    if(sum(sum(bboxes))~=0)
    if(temp>=c)
        break;
    else
    es=imcrop(e,bboxes(1,:));
    es=imresize(es,[224 224]);
    
    filename=strcat(num2str(temp),'.jpg');
    af = "../facerecon/datastorage/";
    path0 = append(af,personname);
    path00 = append(path0, "/");
    path = append(path00, personname);
    newfilename = append(path,filename);
    
    imwrite(es,newfilename);
    temp=temp+1;
    imshow(es);
    drawnow;
    end
    else
        imshow(e);
        drawnow;
    end
end
