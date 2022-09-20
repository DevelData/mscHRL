# mscHRL
Hierarchical Reinforcement Learning using SAC and HIDIO for MSc project.


How to run the code (best results will be on Linux OS with pip and conda installed):
1. Install all dependencies by running: bash ./hidio/mscHRL_env.sh
2. conda activate mscHRL
3. cd ../hidio/hidio_scratch/hidio_and_sac/
4. Edit the main files 'sac_main.py' or 'hidio_main.py' to change hyperparameters and/or environments
5. python [name of file]
6. If you want to transfer network parameters, use the transfer.py file. The non-default arguments in this file can be configured from the terminal.