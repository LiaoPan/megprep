## 使用方法
$ conda activate megprep


## 前期依赖 库安装
mne
osl-ephys

nextflow
streamlit

Freesurfer Environment:(https://hub.docker.com/r/freesurfer/freesurfer/)
$ docker pull freesurfer/freesurfer | timeout
$ docker pull brainlife/freesurfer:7.3.2

FSL Environment
$ docker pull brainlife/fsl


docker pull brainlife/brainstorm:220526

https://www.cnblogs.com/Chary/p/18096678 docker 代理配置
$ sudo mkdir -p /etc/systemd/system/docker.service.d
$ vi /etc/systemd/system/docker.service.d/http-proxy.conf

[Service]
Environment="HTTP_PROXY=http://100.122.141.118:7890/"
Environment="HTTPS_PROXY=http://100.122.141.118:7890/"
Environment="NO_PROXY=localhost,127.0.0.1,.example.com"

重启docker生效：
sudo systemctl daemon-reload
sudo systemctl restart docker

测试方法：
$ curl -x http://100.122.141.118:7890 https://registry-1.docker.io/v2/  


### osl-ephys Installation
git clone https://github.com/OHBA-analysis/osl-ephys.git
cd osl-ephys
conda env create -f envs/linux.yml
conda activate osle
pip install -e .

### streamlit Installation
$ pip install streamlit
$ streamlit hello
Visit URL: http://100.114.213.66:8501/

MPA: https://docs.streamlit.io/get-started/tutorials/create-a-multipage-app
https://digitalhumanities.hkust.edu.hk/tutorials/dive-deeper-into-python-and-streamlit-to-create-website-an-advanced-guide-with-demo-code-and-slides/


### NextFlow Installation
Nextflow can be used on any POSIX-compatible system (Linux, macOS, etc), and on Windows through WSL.
It requires Bash 3.2 (or later) and Java 17 (or later, up to 23) to be installed. 

https://www.nextflow.io/docs/latest/install.html 


$ source "/home/liaopan/.sdkman/bin/sdkman-init.sh"

环境变量：
$HOME/.local/bin/nextflow

nextflow命令行：
$ nextflow info

reports
$ nextflow run tutorial.nf -with-report

$ nextflow run <pipeline> -with-trace

updates
$ nextflow self-update

temporarily switch to a specific version of NextFlow.
$ NXF_VER=23.10.0 nextflow info

### Execute the script
$ nextflow run tutorial.nf

### 修改部分代码，缓存部分结果
$ nextflow run tutorial.nf -resume


Nextflow REPL console
The Nextflow console is a REPL (read-eval-print loop) environment that allows one to quickly test part of a script or segments of Nextflow code in an interactive manner. This can be particularly useful to quickly evaluate channels and operators behaviour and prototype small snippets that can be included in your pipeline scripts.

$ nextflow console

Use the CTRL+R keyboard shortcut to run (⌘+Ron the Mac) and to evaluate your cod


### pipeline parameters
$ nextflow run tutorial.nf --str 'Bonjour le monde'
params.str 替换这个str的内容。


### 日志指定
nextflow -log custom.log 

### Launching a remote project
In other words if a Nextflow project is hosted, 
for example, in a GitHub repository at the address http://github.com/foo/bar, 
it can be executed by entering the following command in your shell terminal:

$ nextflow run foo/bar
or
$ nextflow run http://github.com/foo/bar

### tips
The pipeline results are cached by default in the directory $PWD/work. 
Depending on your script, this folder can take up a lot of disk space.
It’s a good idea to clean this folder periodically, 
as long as you know you won’t need to resume any pipeline runs.


### Listing available projects
$ nextflow list

### Showing project information
nextflow info <project name>

### Pulling or updating a project
The pull command allows you to download a project from a GitHub repository or to update it if that repository has already been downloaded. For example:

$ nextflow pull nextflow-io/hello

$ nextflow pull https://github.com/nextflow-io/hello

### view the project code
$ nextflow view nextflow-io/hello
### clone the project into a local directory
$ nextflow clone nextflow-io/hello target-dir

### delete the project
$ nextflow drop nextflow-io/hello

### nf-core tools: 管理python packages
nf-core is a community effort to collect a curated set of analysis pipelines built using Nextflow. The pipelines continue to come on in leaps and bounds and nf-core tools is a python package for helping with developing nf-core pipelines. It includes options for listing, creating, and even downloading pipelines for offline usage.
$ conda install nf-core
$ pip install nf-core

$ nf-core list


## DEBUG
docker
$ newgrp docker 启动新的shell，让其当前会话主组为docker，以便正常调用docker


recon-all -sd /data/liaopan/datasets/Holmes_cn/smri_ -all -i /data/liaopan/datasets/Holmes_cn/preprocessed/work/88/b6badefb16320b991dab2998622ee7/sub-009/sub-009.nii.gz -s sub-009
ERROR: SUBJECTS_DIR /data/liaopan/datasets/Holmes_cn/smri_ does not exist.


## Docker 转换singularity
$ singularity build deepprep_25.1.0.beta.1.sif docker-daemon:pbfslab/deepprep:25.1.0.beta.1


## DeepPrep.nf 修改地方记录:
process anat_get_t1w_file_in_bids 添加config参数，方便过滤特定结构像
deepprep.common.config 中新增参数mri_import_config



## mesa-18.3.6 安装
$sudo yum install llvm-devel libX11-devel libxcb-devel libXxf86vm-devel libXext-devel libXdamage-devel libXfixes-devel libxshmfence-devel expat-devel
$ wget https://archive.mesa3d.org/older-versions/18.x/mesa-18.3.6.tar.xz
$ tar xf mesa-18.3.6.tar.xz
$ ./configure --with-platforms=x11 --with-gallium-drivers=swrast --with-dri-drivers=swrast --disable-egl
$ make -j 4

https://github.com/mne-tools/mne-python/issues/7977


conda install -c conda-forge libgl-devel mesalib 

## REF
- https://seqera.io/blog/nextflow-developer-environment/
- 