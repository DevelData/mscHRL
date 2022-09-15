conda create --name mscHRL python=3.7 -y
eval "$(conda shell.bash hook)"
conda activate mscHRL
conda install numpy -y
yes | pip install gym[all]==0.25.1
yes | pip install pybullet
yes | pip install mujoco-py<2.2,>=2.1
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch -y
conda install matplotlib -y