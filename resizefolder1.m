faceDetector=vision.CascadeObjectDetector;

c=201;
temp=1;
while true
    ff = append(num2str(temp),".jpg");
    filename = append("/Users/jankarasek/Documents/MATLAB/zpracovaniObrDat/projekt2/datastorage/Allfaces/", ff);
    e=imread(filename);
    bboxes =step(faceDetector,e);
    if(sum(sum(bboxes))~=0)
    if(temp>=c)
        break;
    else
    es=imcrop(e,bboxes(1,:));
    es=imresize(es,[224 224]);
    filename=strcat(num2str(temp),'.jpg');
    af = "/Users/jankarasek/Documents/MATLAB/zpracovaniObrDat/projekt2/datastorage/ALtered/";
    newfilename = append(af,filename);
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