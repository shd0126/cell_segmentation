function result = globalsg(f1)
original = f1;
[m,n,k] = size(original);
if k > 1
    original = rgb2gray(original); % 转换为灰度图片
end

object = original;

a = 20;% a值越大，高频信息滤除的越多，留下的为低频信息；
b = 1;% b值越小，保留高频信息越多
g1 = imgaussfilt(object,a);
% imwrite(g1,'result5/object79_3.tif')
% figure,mesh(g1)
g2 = imgaussfilt(object,b);
% imwrite(g2,'result5/object79_4.tif')
object2 = imsubtract(g1,g2); % 双重高斯滤波消除光照不均匀
% figure,mesh(object2),xlim([0,256]),ylim([0,256])
% figure,imshow(object2),title(strcat('a=',num2str(a),',b=',num2str(b)));
% imwrite(object2,'result5/object79_5.tif');

% std1(a,10) = std(double(object2),0,[1,2]);
% end
% std2 = std1(:,1:5);
% x = [1:50];
% x = x';
% x = repmat(x,[1,10]);
% x2 = x(:,1:5);
% figure
% plot(x2,std2)
% legend('Cell_79','Cell_33','Cell_23','Cell_21','Cell_64')

object2 = imgaussfilt(object,b);

%% 白顶帽:Itophat = I - Io,得到图像中灰度较亮的区域
object2_tophat = imtophat(object2,strel('disk',6));
object2_tophat_adjust = imadjust(object2_tophat);
% figure,mesh(object2_tophat_adjust),xlim([0,256]),ylim([0,256])
% figure,imshow(object2_tophat_adjust),title('白顶帽');
% imwrite(object2_tophat_adjust,'result5\object79_tophat.tif');

    object2_tophat_adjust_medfilt = medfilt2(object2_tophat_adjust,[5,5]); % 使用二维中值滤波器进行滤波消除椒盐噪声
%     imwrite(object2_tophat_adjust_medfilt,'result5\object79_medfilt.tif')
% figure,mesh(object2_tophat_adjust_medfilt),xlim([0,256]),ylim([0,256])
    %---求二维直方图---
    fxy = zeros(m,n);
    for i = 1:m
        for j = 1:n
             a = object2_tophat_adjust(i,j);
             b = object2_tophat_adjust_medfilt(i,j);
             fxy(a+1,b+1) = fxy(a+1,b+1) + 1;
        end
    end
%     fxy = fxy./(256*256);

%     figure
%     mesh(fxy)
%     xlim([0,256])
%     ylim([0,256])
%     figure,imshow(fxy),title('projection')

    %------------------ 

    level1 = graythresh(object2_tophat_adjust);
%     level1 = graythresh(object);
    level2 = graythresh(object2_tophat_adjust_medfilt);
    object2_tophat_adjust_normalization = double(object2_tophat_adjust)./255;
%     object2_tophat_adjust_normalization = double(object)./255;
    object2__tophat_adjust_medfilt_normalization = double(object2_tophat_adjust_medfilt)./255;
    
    temp = ones(m,n);
    for i = 1:m
        for j = 1:n
            if object2_tophat_adjust_normalization(i,j) <= level1 && object2__tophat_adjust_medfilt_normalization(i,j) <= level2
                temp(i,j) = 0;
            end
        end
    end
    object3 = temp;
%     imwrite(object3,'result5\object79_2Dotsu.tif')
    object3 = imbinarize(object3);



object4 = imclose(object3,strel('disk',10));
object5_1 = imfill(object4,"holes");
object6 = imopen(object5_1,strel("disk",1)); % 该项主要影响细长细胞的分割，会把其细胞细长部分给去除
object7 = bwareaopen(object6,500);
result = object7;
% figure
% imshow(result)
end

