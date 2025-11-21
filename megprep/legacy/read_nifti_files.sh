#!/bin/bash  

# 读取 demo_mri.txt 文件，并进行处理  
while IFS= read -r line; do  
    # 提取被试 ID (sub_id) 和文件路径 (nii_files)  
    sub_id=$(echo "$line" | awk -F : '{print $1}' | xargs)  # 取出被试 ID  
    nii_files=$(echo "$line" | awk -F : '{print $2}' | sed "s/[][]//g" | xargs)  # 删除方括号并去掉多余空格  

    # 将 NIfTI 文件路径按逗号分开，循环输出每个路径  
    IFS=',' read -ra files <<< "$nii_files"  # 读取到数组  
    for nii_file in "${files[@]}"; do  
        echo "$sub_id $nii_file" | xargs  # 输出被试 ID 和对应的 NIfTI 路径  
    done  
done < demo_mri.txt  
