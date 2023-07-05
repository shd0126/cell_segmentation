%% 获取整张明场图片的全部细胞轮廓:需要自己调整参数，方法一般
% 1 先得到一个正方形框，之后由细胞核位置质心定位明场细胞位置，对单个细胞进行分割
% 2 把全部分割结果叠加在明场细胞位置上
clear;close all;clc
prompt1 = '请输入计数荧光点的阈值: ';
threshold33 = input(prompt1);
threshold3 = threshold33
prompt2 = '请输入计数荧光强度的阈值: ';
threshold44 = input(prompt2);
threshold4 = threshold44
tic
timebar = waitbar(0,"正在运行中，请等待");

original = imread('原图/2022.6.1_4_original.tif');% 读入原始图像
nucleus1 = imread('原图/2022.6.1_4_nucleus.tif');    % 读入细胞核图片
fluo11 = imread('原图/11_r-fluo.tif'); % 读入红色荧光图片
fluo22 = imread('原图/11_b-fluo.tif'); % 读入青色荧光图片
fluo33 = imread('原图/11_p-fluo.tif'); % 读入紫色荧光图片
fluo44 = imread('原图/11_y-fluo.tif'); % 读入黄色荧光图片
gfluointensity11 = imread('原图/11_gfluo_intensity.tif'); % 读入绿色荧光强度图片
[m,n,k] = size(original);
[m2,n2,k2] = size(fluo11);
[m3,n3,k3] = size(nucleus1);
if k > 1
    original = rgb2gray(original); % 转换为灰度图片
end
if k2 > 1
    fluo1 = rgb2gray(fluo11); % 转换为灰度图片
    fluo2 = rgb2gray(fluo22); % 转换为灰度图片
    fluo3 = rgb2gray(fluo33); % 转换为灰度图片
    fluo4 = rgb2gray(fluo44); % 转换为灰度图片
    gfluointensity1 = rgb2gray(gfluointensity11); % 转换为灰度图片
end
if k3 > 1
    nucleus = nucleus1(:,:,3);    % 转换为灰度图片
end
        

result = original; % 保存分割结果
count = original; % 保存计数结果


% figure,imshow(original),title('original')
nucleus_binarize1 = nucleus;
nucleus_binarize2 = imbinarize(nucleus_binarize1); % 对细胞核图片二值化
% h = fspecial('disk',2);
% nucleus_binarize2 = imfilter(nucleus_binarize,h,'replicate');
% nucleus_binarize2 = medfilt2(nucleus_binarize,[5,5]); % 中值滤波
nucleus_binarize3 = bwareaopen(nucleus_binarize2,500); % 去除杂散点
nucleus_binarize4 = imfill(nucleus_binarize3,'holes'); % 空洞填充,存在3个包含两个细胞核粘连在一起的图
% imwrite(nucleus_binarize4,'result3/result1_nucleus.tif');


[L,N] = bwlabel(nucleus_binarize4); % 对连通域分量进行标记

mask1 = zeros(m,n,"logical");
mask2 = zeros(m,n,'logical');
method = 5; % method为1使用activecontour(花费时间最多)方法；method=2使用K-means方法；method为3使用阈值分割;method为4使用图割方法分割;method为5使用二维OTSU方法

% N = 100;


timebar = waitbar(0,timebar,"图像分割中，请等待");

% std1 = []; %标准差

for num = 1:N
    L2 = L;
    x = 0;
    y = 0;
    sum = 0;
    for i = 1:m
        for j = 1:n
            if L2(i,j) == num    % 找到特定细胞
                y = y + L2(i,j)*i;
                x = x + L2(i,j)*j;
                sum = sum + L2(i,j);
            else
                L2(i,j) = 0;
            end
        end
    end
average_y = round(y/sum);   % y轴质心
average_x = round(x/sum);   % x轴质心
% L2 = insertText(L2,[1064,500],strcat('质心(',num2str(average_x),',',num2str(average_y),')'),"FontSize",48);
% imwrite(L2,'L2.tif');

square = [];    % 框的长宽选择是个难点，需要使其包含整个需要分割的细胞；关系到粘连细胞的切割，若是框大小合适，对粘连细胞的切割很有帮助
square(1) = average_y - 128; % y轴初始
i_bd = m ;
j_bd = n;
if square(1) < 1
    square(1) = 1;
end
square(2) = average_y + 127; % y轴结束
if square(2) > i_bd
    square(2) = i_bd;
end
square(3) = average_x - 128; % x轴初始
if square(3) < 1
    square(3) = 1;
end
square(4) = average_x + 127; % x轴结束
if square(4) > j_bd
    square(4) = j_bd;
end

object = imcrop(original,[square(3),square(1),square(4)-square(3),square(2)-square(1)]); % 注意xmin，ymin，x轴width，y轴heigth顺序
[m4,n4] = size(object);

% if m4 == 236 && n4 == 210
%     imwrite(object,strcat('result11/','2022_6_1_original_1-原-',num2str(num),'.bmp'));
% end

%%-----%%


mask = imcrop(nucleus_binarize4,[square(3),square(1),square(4)-square(3),square(2)-square(1)]); % 把细胞核信息也加入进去

nucleus_object = imcrop(nucleus1,[square(3),square(1),square(4)-square(3),square(2)-square(1)]); % 没有灰度化的细胞核
% imwrite(nucleus_object,strcat('result12_4/','2022_6_1_original_4-原细胞核-',num2str(num),'.tif'));

%% 直方图均衡化
% test_he = histeq(object);
% imwrite(test_he,'result3/object79_he.tif')

%% 双重高斯滤波。通过调节a和b的参数值，一定可以较为完整的分割出细胞

% for a = 1:50

a = 20;% a值越大，高频信息滤除的越多，留下的为低频信息；
b = 1;% b值越小，保留高频信息越多
g1 = imgaussfilt(object,a);
% imwrite(g1,'result5/object79_3.tif')
% figure,mesh(g1)
g2 = imgaussfilt(object,b);
% imwrite(g2,'result5/object79_4.tif')
g1 = double(g1);
g2 = double(g2);
object2 = imsubtract(g2,g1); % 双重高斯滤波消除光照不均匀
% figure,mesh(object2),xlim([0,256]),ylim([0,256])
% figure,imshow(object2),title(strcat('a=',num2str(a),',b=',num2str(b)));
% imwrite(object2,strcat('result11/','2022_6_1_original_1-双重-',num2str(num),'.bmp'));

% std1(a,num) = std(double(object2),0,[1,2]);
% end
% end

% x = [1:50];
% x = x';
% x = repmat(x,[1,N]);
% x2 = x;
% figure
% plot(x2,std1)
% xlabel('$\sigma_2$',Interpreter='latex',FontSize=16)
% ylabel('$standard deviation of f_dg$',Interpreter='latex',FontSize=16)
% save('std_9.mat',"std1")
legend('Cell_1','Cell_2','Cell_3','Cell_4','Cell_5','Cell_6','Cell_7','Cell_8','Cell_9','Cell_{10}')
% tic

%% 单纯高斯滤波
% object2 = imgaussfilt(object,b);
%% 白顶帽:Itophat = I - Io,得到图像中灰度较亮的区域
object2_tophat = imtophat(object2,strel('disk',6));
object2_tophat_adjust = imadjust(object2_tophat,stretchlim(object2_tophat),[]);
% figure,mesh(object2_tophat_adjust),xlim([0,256]),ylim([0,256])
% figure,imshow(object2_tophat_adjust),title('白顶帽');
% imwrite(object2_tophat_adjust,strcat('result11/','2022_6_1_original_1-白顶帽-',num2str(num),'.bmp'));

switch method
    case 1
%% 主动轮廓方法分割
mask_activecontour = zeros(m4,n4);
mask_activecontour(10:end-10,10:end-10) = 1;
object3 = activecontour(object2_tophat_adjust,mask_activecontour,1000,'edge');
% figure,imshow(object3_activecontour),title('主动轮廓方法分割')
% imwrite(object3_activecontour,'result3\object79_activecontour.tif')
object8 = imadd(object3,mask);
    case 2
%% 基于K均值聚类的图像分割
object_kmeans = imsegkmeans(object2_tophat_adjust,2);
object_kmeans(object_kmeans==2) = 255;
object3 = imbinarize(object_kmeans);
% figure,imshow(object_kmeans2),title('基于K均值聚类的图像分割');
% imwrite(object_kmeans2,'result3\object79_kmeans.tif')

% figure,imshow(object3),title(strcat('a=',num2str(a),',b=',num2str(b)));
% imwrite(object3,'result3/object79_6.tif');
object8 = imadd(object3,mask);
    case 3
%% 阈值法分割
level = graythresh(object2_tophat_adjust);
object3 = imbinarize(object2_tophat_adjust,level); % OTSU阈值法分割
object8 = imadd(object3,mask);
    case 4
        %% 图割法：grabcut(original,label,roi)，官方给出的文档中标签矩阵label由超像素分割superpixel产生
        label = superpixels(object2_tophat_adjust,500);
        roi = false(size(object2_tophat_adjust)); % roi的选择也很重要，图割法是从roi开始向内压缩的，直到形成分割区域
        roi(3:end-3,3:end-3) = true;
        object3 = grabcut(object2_tophat_adjust,label,roi);
        object8 = imadd(object3,mask);

    case 5 
        %% 改进二维OTSU阈值分割
    object2_tophat_adjust_medfilt = medfilt2(object2_tophat_adjust,[5,5]); % 使用二维中值滤波器进行滤波消除椒盐噪声
%     imwrite(object2_tophat_adjust_medfilt,strcat('result11/','2022_6_1_original_1-中值滤波-',num2str(num),'.bmp'))
% figure,mesh(object2_tophat_adjust_medfilt),xlim([0,256]),ylim([0,256])
    %---求二维直方图---
%     fxy = zeros(256,256);
%     for i = 1:m4
%         for j = 1:n4
%              a = object2_tophat_adjust(i,j);
%              b = object2_tophat_adjust_medfilt(i,j);
%              fxy(a+1,b+1) = fxy(a+1,b+1) + 1;
%         end
%     end
%     fxy = fxy./(256*256);

%     figure,mesh(fxy),xlim([0,256]),ylim([0,256])
%     figure,imshow(fxy,[]),title('projection')

    %------------------ 

    level1 = graythresh(object2_tophat_adjust);
%     level1 = graythresh(object);
    level2 = graythresh(object2_tophat_adjust_medfilt);
    object2_tophat_adjust_normalization = double(object2_tophat_adjust)./255;
%     object2_tophat_adjust_normalization = double(object)./255;
    object2__tophat_adjust_medfilt_normalization = double(object2_tophat_adjust_medfilt)./255;
%%%%%%-------画图-------%%%%%%
%     fxy2 = fxy;
%     fxy2(round(level1*255),:) = 255; % 这两个步骤的作用是什么？画图作用的，画两条阈值分割线
%     fxy2(:,round(level2*255)) = 255; %
%     figure,imshow(fxy2)
%%%%%%--------------%%%%%%



    temp = ones(m4,n4);
    for i = 1:m4
        for j = 1:n4
            if object2_tophat_adjust_normalization(i,j) <= level1 && object2__tophat_adjust_medfilt_normalization(i,j) <= level2
                temp(i,j) = 0;
            end
        end
    end
    object3 = temp;
%     imwrite(object3,'result5\object79_2Dotsu.tif')
    object3 = imbinarize(object3);


object4 = imadd(object3,mask);
% imwrite(object4,strcat('result11/','2022_6_1_original_1-二维OTSU-',num2str(num),'.bmp'));

%% 形态学处理


% 1 闭运算
object5 = imclose(object4,strel('disk',8)); % 闭运算
% imwrite(object5,'result3/object79_8.tif')

% 1.1 填充空洞

% imwrite(object5_1,'result3/object79_9.tif')
object5_1 = object5;
% 2 开运算
object6 = imopen(object5_1,strel("disk",5)); % 关系到粘连细胞的切割，若是设置的值过大，容易把整个细胞切割开，如num=79；
% imwrite(object6,'result3/object79_10.tif')

% 3 从二值图像中删除小对象
object7 = bwareaopen(object6,200); % 从二值图像中删除小对象，针对不同明场图像需要选择一个合适的值;
% imwrite(object7,'result3/object79_11.tif')
% 第8组数据设置为1400；

% 4 填充孔洞
object8 = imfill(object7,"holes");

% if m4 == 236 && n4 == 210
%     imwrite(object8,strcat('result11/','2022_6_1_original_1-形态学处理-',num2str(num),'.bmp'));
% end

end
%%-----%%


%% 把单个细胞的分割结果合并在一张图上，方便后面使用分水岭分割
mask1(square(1):square(2),square(3):square(4)) = object8; % 把分割结果合并在一张图上，最后再使用边缘检测的方法检测边缘

for i = 1:m
    for j = 1:n
        if mask1(i,j) == 1
            mask2(i,j) = 1; % 分割边缘结果保存在mask2中
        end
    end
end


waitbar(num/N,timebar);
end
toc

mask2_2 = mask2;
% imwrite(mask2_2,'result11/2022_6_1_original_1-proposed_without_golobal.bmp')
% [~,N2] = bwlabel(mask2);

%% 调用全局分割函数globalsg来分割整幅图片，保留全局信息，把细胞细长部分也考虑进去
% 
waitbar(0,timebar,"优化分割中，请等待")
mask2_3 = globalsg(original);
% imwrite(mask2_3,'result9/11_original_3-proposed_golal.bmp')
mask2 = imadd(mask2,mask2_3);

% imwrite(mask2,'result11/2022_6_1_original_1-proposed_fuse.tif')

%% watershed
% 分水岭分割很关键，有了这个函数，不管是哪种分割函数，都能够进行有效分割，只不过是细胞分割形状的差异，即ROI的区别
waitbar(0.5,timebar,"优化分割中，请等待")
threshold2 = 5; % 扩展极小值变换的阈值
d1 = -bwdist(~L); %  距离变换，适用细胞核图像作为掩膜
mask = imextendedmin(d1,threshold2);
d2 = imimposemin(d1,mask);
l = watershed(d2); % 分水岭变换
object9 = mask2;
object9(l==0) = 0;
% imwrite(object9,'result3/object_watershed.tif')
object9_1 = imerode(object9,strel("disk",1)); % 使粘连细胞分割更加明显，但是也会造成细胞原始图像大小减小的问题
object10 = bwareaopen(object9_1,1400); % 去除小杂散点。
waitbar(1,timebar,"优化分割中，请等待")
toc
% 细胞核原本数据为97个，第11组数据面积限制设置为3800时刚好个数为97个，但是会把一些细胞也给去除；为3000时为100个；为3400时为99个，粘连细胞核也成功分开了；为1400时个数为102个；
% 细胞核原本为234，第8组数据设置为1400,N为238，多了4个；
% imwrite(object10,'result11/2022_6_1_original_1-proposed_watershed.bmp')

%% 边缘检测

edge1 = edge(object10,'Canny');
% imwrite(edge1,'result3/object_edge1.tif');
result2 = labeloverlay(result,edge1,'Colormap',[1 0 0],'Transparency',0); % 把分割边缘结果叠加在原图上,最后结果保存在result上

% [L2,N2] = bwlabel(object10);
% figure,imshow(result2),title(strcat('分割结果叠加在原图上,实际细胞个数为',num2str(N),',分割细胞个数为',num2str(N2)),FontSize=18);

% imwrite(result2,'result13/2022_6_1_original_4_overlay.bmp');
result3 = labeloverlay(fluo11,edge1,'Transparency',0); % 把分割结果叠加在荧光图片上
result3_2 = labeloverlay(fluo22,edge1,'Transparency',0); % 把分割结果叠加在荧光图片上
result3_3 = labeloverlay(fluo33,edge1,'Transparency',0); % 把分割结果叠加在荧光图片上
result3_4 = labeloverlay(fluo44,edge1,'Transparency',0); % 把分割结果叠加在荧光图片上
result3_5 = labeloverlay(gfluointensity11,edge1,'Transparency',0); % 把分割结果叠加在荧光图片上
result3_51 = labeloverlay(gfluointensity1,edge1,'Transparency',0); % 把分割结果叠加在灰度荧光图片上,绿色荧光图片不免先
%% 计数荧光点

[L2,N2] = bwlabel(object10); % 给明场细胞编号
count = result2;
count2_1 = result3; % 只有这样才能在同一张图现实多个结果啊
count2_2 = result3_2;
count2_3 = result3_3;
count2_4 = result3_4;
count2_5 = result3_5;
count2_51 = result3_51;
%——————保存计数结果到表格
   result4 = [] ;
   cellnumber = [1:N2]';
   result4 = cellnumber;
%——————

timebar = waitbar(0,timebar,"荧光点计数和强度面积计算中，请等待");

for number = 1:N2

    L3 = L2;
    x2 = 0;
    y2 = 0;
    sum2 = 0;
   for i = 1:m
       for j = 1:n
           if L3(i,j) == number % 把属于该标签细胞的范围置1
               x2 = x2 + j*L3(i,j);
               y2 = y2 + i*L3(i,j);
               sum2 = sum2 + L2(i,j);
               L3(i,j) = 1;
           else
               L3(i,j) = 0; % 把不属于该标签范围的置0
           end
       end
   end
   average_x2 = round(x2/sum2); % 获取x轴细胞质心坐标
   average_y2 = round(y2/sum2); % 获取y轴细胞质心坐标
   L33 = im2uint8(L3); % 转换为uint8位整型，方便与fluo图像相乘
   L33(L33 == 255) = 1; % 转换为8位整型时，1变成了255，需要把其返回成1
   multi1 = immultiply(fluo1,L33); % 该细胞和荧光图像相乘，获取特定细胞的荧光点
   multi2 = immultiply(fluo2,L33);
   multi3 = immultiply(fluo3,L33);
   multi4 = immultiply(fluo4,L33);
   gfluointensity_area = immultiply(gfluointensity1,L33);




%%% 调用函数来计算荧光点个数
fluonumbers1 = fluocount(multi1,threshold3);
fluonumbers2 = fluocount(multi2,threshold3);
fluonumbers3 = fluocount(multi3,threshold3);
fluonumbers4 = fluocount(multi4,threshold3);
%%%

%%% 求荧光强度图片的面积
g_area = countintensity(gfluointensity_area,threshold4);




   if fluonumbers1 > 0 % 加一个判定条件，只显示荧光点个数不为0的结果

       text1 = num2str(fluonumbers1,'%03d'); % 数字信息
%        count = insertText(count,[average_x2-23,average_y2],text,FontSize=24); % 把结果显示在明场细胞图片count上的每个细胞质心位置
       %    figure,imshow(object11),title('荧光点计数',FontSize=18);
       count2_1 = insertText(count2_1,[average_x2-23,average_y2],text1,FontSize=24); % 把结果显示在荧光图片count上的每个细胞质心位置

   end

 if fluonumbers2 > 0 % 加一个判定条件，只显示荧光点个数不为0的结果

       text2 = num2str(fluonumbers2,'%03d'); % 数字信息
%        count = insertText(count,[average_x2-23,average_y2],text,FontSize=24); % 把结果显示在明场细胞图片count上的每个细胞质心位置
       %    figure,imshow(object11),title('荧光点计数',FontSize=18);
       count2_2 = insertText(count2_2,[average_x2-23,average_y2],text2,FontSize=24); % 把结果显示在荧光图片count上的每个细胞质心位置

 end

  if fluonumbers3 > 0 % 加一个判定条件，只显示荧光点个数不为0的结果

       text3 = num2str(fluonumbers3,'%03d'); % 数字信息
%        count = insertText(count,[average_x2-23,average_y2],text,FontSize=24); % 把结果显示在明场细胞图片count上的每个细胞质心位置
       %    figure,imshow(object11),title('荧光点计数',FontSize=18);
       count2_3 = insertText(count2_3,[average_x2-23,average_y2],text3,FontSize=24); % 把结果显示在荧光图片count上的每个细胞质心位置

  end

   if fluonumbers4 > 0 % 加一个判定条件，只显示荧光点个数不为0的结果

       text4 = num2str(fluonumbers4,'%03d'); % 数字信息
%        count = insertText(count,[average_x2-23,average_y2],text,FontSize=24); % 把结果显示在明场细胞图片count上的每个细胞质心位置
       %    figure,imshow(object11),title('荧光点计数',FontSize=18);
       count2_4 = insertText(count2_4,[average_x2-23,average_y2],text4,FontSize=24); % 把结果显示在荧光图片count上的每个细胞质心位置

   end

   if g_area > 0

       text5 = num2str(g_area,'%.2f');
       count2_5 = insertText(count2_5,[average_x2-23,average_y2],text5,FontSize=24); % 把结果显示在荧光图片count上的每个细胞质心位置
       count2_51 = insertText(count2_51,[average_x2-23,average_y2],text5,FontSize=24); % 把结果显示在荧光图片count上的每个细胞质心位置

   end

   %————保存计数结果
   result4(number,2) = fluonumbers1;
   result4(number,3) = fluonumbers2;
   result4(number,4) = fluonumbers3;
   result4(number,5) = fluonumbers4;
   result4(number,6) = g_area;

   %————

waitbar(number/N2,timebar);
end

%% 求荧光点和荧光强度值的平均值

result4(N2+1,2) = mean2(result4(:,2));
result4(N2+1,3) = mean2(result4(:,3));
result4(N2+1,4) = mean2(result4(:,4));
result4(N2+1,5) = mean2(result4(:,5));
result4(N2+1,6) = mean2(result4(:,6));


waitbar(1,timebar,'完成')
toc
delete(timebar);

figure,imshow(result2),title(strcat('分割结果叠加在原图上,实际细胞个数为',num2str(N),',分割细胞个数为',num2str(N2)),FontSize=18);
% figure,imshow(count),title('显示在明场细胞上荧光点计数',FontSize=18);
figure,imshow(count2_1),title('显示在荧光图片上的红色荧光点计数fluo1',FontSize=18);
figure,imshow(count2_2),title('显示在荧光图片上的青色荧光点计数fluo2',FontSize=18);
figure,imshow(count2_3),title('显示在荧光图片上的紫色荧光点计数fluo3',FontSize=18);
figure,imshow(count2_4),title('显示在荧光图片上的黄色荧光点计数fluo4',FontSize=18);
figure,imshow(count2_5),title('显示在绿色荧光强度图片上的面积强度gfluointensity',FontSize=18);
figure,imshow(count2_51),title('显示在绿色灰度荧光强度图片上的面积强度gfluointensity',FontSize=18);

columnname = {'细胞编号','fluo1红色荧光点个数','fluo2青色荧光点个数','fluo3紫色荧光点个数','fluo4黄色荧光点个数',...
    'gfluointensity绿色荧光强度'};
cellnumbers = table(result4(:,1),result4(:,2),result4(:,3),result4(:,4),result4(:,5),result4(:,6),'VariableNames',columnname);
writetable(cellnumbers,'result.xls')

% writematrix(result4,'mydata.xls')

% imwrite(result2,'result6/11_original2.tif');
% imwrite(count,'result3/result11_original_c.tif');
% imwrite(count2,'result3/result11_y-fluo.tif');







